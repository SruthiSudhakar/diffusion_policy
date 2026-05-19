"""
Simple paper figures: BC stochasticity & best-of-K headroom.

Single-panel plots, each makes one point:
    bars.pdf         -- single-shot vs best-of-5 success rate (headline)
    passk.pdf        -- pass@K curve with K=1 and K=5 annotated
    traces.pdf       -- K=5 trajectories for one (run, cycle) example
    traces_grid.pdf  -- grid of many (task x cycle) trace examples,
                        showing divergence is not a one-off

Usage:
    # single-root mode (bars, passk, traces):
    python paper_figs_simple.py --root /path/to/<ckpt_stem>/

    # multi-task grid (traces_grid):
    python paper_figs_simple.py --grid
"""
import pathlib
import random
import sys

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


JOINT_LABELS = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'gripper']


def style():
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 11,
        'pdf.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ---------- run discovery & status parsing ----------

def parse_status(name: str):
    parts = name.split('_')
    if len(parts) < 2:
        return None
    has_poor = any('poorcritic' in p for p in parts)
    idx = 2 if has_poor else 1
    if idx >= len(parts):
        return None
    sf = parts[idx]
    if len(sf) != 2 or sf[0] not in 'sf' or sf[1] not in 'sf':
        return None
    return {
        'prefix': parts[0],
        'pick': sf[0] == 's',
        'place': sf[1] == 's',
        'has_vlm': 'VLM' in parts,
        'has_poorcritic': has_poor,
    }


def discover_runs(root: pathlib.Path):
    out = []
    for p in root.rglob('actions_000000.npy'):
        s = parse_status(p.parent.name)
        if s is None or s['has_poorcritic']:
            continue
        out.append((p.parent, s))
    out.sort(key=lambda x: str(x[0]))
    return out


# ---------- figure 1: bars ----------

def fig_bars(runs, K_show, out_path):
    pick = np.array([s['pick'] for _, s in runs])
    place = np.array([s['place'] for _, s in runs])
    joint = pick & place
    p1 = float(joint.mean())
    pK = 1 - (1 - p1) ** K_show

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ['1 try', f'best of {K_show}'],
        [p1, pK],
        color=['#888888', '#2ca02c'],
        width=0.55,
    )
    for b, v in zip(bars, [p1, pK]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f'{100 * v:.0f}%', ha='center', va='bottom',
                fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('success rate')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title('Best-of-K headroom\n(upper bound, single-rollout dataset)',
                 fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[bars]   wrote {out_path}  '
          f'(n={len(runs)}, p1={p1:.2f}, p@{K_show}={pK:.2f})')


# ---------- figure 2: pass@K curve ----------

def fig_passk(runs, out_path):
    pick = np.array([s['pick'] for _, s in runs])
    place = np.array([s['place'] for _, s in runs])
    joint = pick & place
    p = float(joint.mean())
    Ks = np.arange(1, 11)
    y = 1 - (1 - p) ** Ks

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ks, y, color='#2ca02c', linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='white',
            markeredgecolor='#2ca02c', markeredgewidth=2)

    # annotate K=1 and K=5
    ax.annotate(f'{100*y[0]:.0f}%  (single try)',
                xy=(1, y[0]), xytext=(1.4, y[0] - 0.10),
                fontsize=11,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.annotate(f'{100*y[4]:.0f}%  (best of 5)',
                xy=(5, y[4]), xytext=(5.2, y[4] - 0.18),
                fontsize=11,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.set_xticks(Ks)
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('K (samples per scene)')
    ax.set_ylabel('success rate')
    ax.set_title('Pass@K: success if any of K rollouts succeeds')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[passk]  wrote {out_path}  '
          f'(n={len(runs)}, p={p:.2f}, p@5={y[4]:.2f})')


# ---------- figure 3: trajectory traces ----------

def pick_example(runs, min_k=5):
    """Pick a (run, cycle, dof) with high sample spread."""
    best = None
    for run_dir, _ in runs:
        files = sorted(run_dir.glob('actions_[0-9]*.npy'))
        for af in files:
            cyc = int(af.stem.split('_')[-1])
            a = np.load(af)  # (K, n_t, 7)
            if a.shape[0] < min_k:
                continue
            # score = max sample std anywhere in this cycle
            score = a.std(axis=0).max()
            if best is None or score > best[0]:
                best = (score, run_dir, cyc, af)
    if best is None:
        return None
    _, run_dir, cyc, af = best
    a = np.load(af)
    # pick the DOF with the highest peak std for this cycle
    std_t = a.std(axis=0)  # (n_t, 7)
    dof = int(std_t.max(axis=0).argmax())
    return run_dir, cyc, dof, a


def fig_traces(runs, out_path):
    pick = pick_example(runs)
    if pick is None:
        print('[traces] no cycle data', file=sys.stderr)
        return
    run_dir, cyc, dof, a = pick
    K, n_t, _ = a.shape
    ts = np.arange(n_t)
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in range(K):
        ax.plot(ts, a[s, :, dof],
                color=cmap(s % 10), linewidth=2.0,
                label=f'sample {s + 1}')
    ax.set_xlabel('timestep within action chunk')
    ax.set_ylabel(f'{JOINT_LABELS[dof]} value')
    ax.set_title(f'{K} samples from the same observation\n'
                 f'(cycle {cyc} of {run_dir.name}, {JOINT_LABELS[dof]})')
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[traces] wrote {out_path}  '
          f'(run={run_dir.name}, cycle={cyc}, dof={JOINT_LABELS[dof]})')


# ---------- figure 4: traces grid across tasks ----------

# Hard-coded task roots with the most data (auto-detected earlier).
TASK_ROOTS = {
    'pnp_lego':
        'data/jgd/2026.05.06/23.17.10_train_diffusion_unet_hybrid_pnp_lego_image/'
        'checkpoints/epoch=0200-train_loss=0.0116',
    # 'bag_plate':
    #     'data/jgd/2026.05.14/15.07.10_train_diffusion_unet_hybrid_bag_plate_image_only/'
    #     'checkpoints/archive/epoch=0300-train_loss=0.0208',
    'push_bowl':
        'data/jgd/2026.05.15/18.07.44_train_diffusion_unet_hybrid_push_bowl_image_only/'
        'checkpoints/epoch=0250-train_loss=0.0265',
}


def collect_high_spread_cycles(root: pathlib.Path, min_k=4):
    """For each cycle under any run dir, return (spread, run_dir, cyc, dof, actions).

    Discovery here is loose: any directory containing actions_*.npy. We
    do not require a parseable s/f label because tasks use slightly
    different naming conventions, and the grid only cares about action
    spread, not success.
    """
    entries = []
    seen_runs = set()
    for af in root.rglob('actions_[0-9]*.npy'):
        run_dir = af.parent
        seen_runs.add(run_dir)
        a = np.load(af)  # (K, n_t, 7)
        if a.shape[0] < min_k:
            continue
        std_t = a.std(axis=0)  # (n_t, 7)
        dof = int(std_t.max(axis=0).argmax())
        spread = float(std_t[:, dof].max())
        cyc = int(af.stem.split('_')[-1])
        entries.append((spread, run_dir, cyc, dof, a))
    return entries


def fig_variance_progression(out_path, base_dir=pathlib.Path('.'),
                             n_buckets=5):
    """One panel per task. X = rollout progression (5 buckets, start->end).
    Y = average sample variance across all rollouts. 7 lines (joints) per panel.

    Aggregation: for each (run, cycle) we compute the std across samples then
    average over chunk timesteps -> a (7,) vector. Each cycle is bucketed by
    its position in its rollout (cycle_index / total_cycles). Vectors within
    a bucket are averaged across all (run, cycle) of that task.
    """
    tasks_data = {}  # task -> (n_buckets, 7)
    for task, rel_root in TASK_ROOTS.items():
        root = base_dir / rel_root
        if not root.is_dir():
            print(f'[progression] skipping {task}: {root} missing',
                  file=sys.stderr)
            continue

        # Group action files by run dir.
        per_run: dict[pathlib.Path, list[pathlib.Path]] = {}
        for af in root.rglob('actions_[0-9]*.npy'):
            per_run.setdefault(af.parent, []).append(af)

        bucket_vecs: list[list[np.ndarray]] = [[] for _ in range(n_buckets)]
        for files in per_run.values():
            files = sorted(files)
            N = len(files)
            if N == 0:
                continue
            for i, af in enumerate(files):
                a = np.load(af)  # (K, n_t, 7)
                if a.shape[0] < 2:
                    continue
                std_per_dof = a.std(axis=0).mean(axis=0)  # (7,)
                if N > 1:
                    b = min(n_buckets - 1,
                            int(n_buckets * i / N))
                else:
                    b = 0
                bucket_vecs[b].append(std_per_dof)

        avgs = np.full((n_buckets, 7), np.nan)
        for b in range(n_buckets):
            if bucket_vecs[b]:
                avgs[b] = np.mean(bucket_vecs[b], axis=0)
        tasks_data[task] = avgs

    if not tasks_data:
        sys.exit('[progression] no task data found')

    n_tasks = len(tasks_data)
    fig, axes = plt.subplots(1, n_tasks,
                             figsize=(4.2 * n_tasks, 4),
                             sharey=False)
    if n_tasks == 1:
        axes = [axes]
    cmap = plt.get_cmap('tab10')

    task_display = {
        'pnp_lego': 'PnP Lego To Bowl',
        'push_bowl': 'Push Bowl',
    }
    task_xmax = {
        'pnp_lego': 240,
        'push_bowl': 96,
    }

    handles = []
    labels = []
    for ax, (task, avgs) in zip(axes, tasks_data.items()):
        xmax = task_xmax.get(task, n_buckets)
        x = np.linspace(0, xmax, n_buckets)
        for d in range(7):
            line, = ax.plot(x, avgs[:, d],
                            color=cmap(d), linewidth=2.0,
                            marker='o', markersize=6,
                            label=JOINT_LABELS[d])
            if ax is axes[0]:
                handles.append(line)
                labels.append(JOINT_LABELS[d])
        ax.set_title(task_display.get(task, task), fontsize=12)
        ax.set_xticks(np.linspace(0, xmax, n_buckets))
        ax.set_xlabel('rollout cycle')
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel('avg sample variance')

    # Legend below the plots.
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=len(JOINT_LABELS),
               fontsize=10, frameon=False,
               title='joint')
    fig.suptitle(
        'Sample variance averaged across all rollouts, '
        'over 5 stages of the rollout',
        y=1.03, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.08, 1.0, 1.0])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    summary = ', '.join(
        f'{t}={int(np.sum(~np.isnan(v[:, 0])))}/{n_buckets} buckets'
        for t, v in tasks_data.items()
    )
    print(f'[progression] wrote {out_path}  ({summary})')


def fig_traces_grid(out_path, n_per_task=6, base_dir=pathlib.Path('.'),
                    seed=0):
    rng = random.Random(seed)
    tasks_data = {}
    for task, rel_root in TASK_ROOTS.items():
        root = base_dir / rel_root
        if not root.is_dir():
            print(f'[grid] skipping {task}: {root} not found', file=sys.stderr)
            continue
        entries = collect_high_spread_cycles(root)
        if not entries:
            print(f'[grid] skipping {task}: no high-K cycles', file=sys.stderr)
            continue
        # Take from the top half by spread, then random-sample so the
        # picks are not all the same extreme example.
        entries.sort(key=lambda e: e[0], reverse=True)
        top = entries[: max(2 * n_per_task, len(entries) // 2)]
        rng.shuffle(top)
        tasks_data[task] = top[:n_per_task]

    if not tasks_data:
        sys.exit('[grid] no task data found')

    n_rows = len(tasks_data)
    n_cols = n_per_task
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.7 * n_cols, 1.6 * n_rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    cmap = plt.get_cmap('tab10')

    for r, (task, examples) in enumerate(tasks_data.items()):
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(examples):
                _spread, run_dir, cyc, dof, a = examples[c]
                ts = np.arange(a.shape[1])
                for s in range(a.shape[0]):
                    ax.plot(ts, a[s, :, dof],
                            color=cmap(s % 10),
                            linewidth=1.2, alpha=0.9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.04, 0.94, JOINT_LABELS[dof],
                        transform=ax.transAxes,
                        fontsize=8, va='top', ha='left',
                        color='#444',
                        bbox=dict(boxstyle='round,pad=0.18',
                                  facecolor='white',
                                  edgecolor='none', alpha=0.7))
            else:
                ax.axis('off')
            if c == 0:
                ax.set_ylabel(task, fontsize=11, rotation=90,
                              labelpad=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Multiple samples from the same observation '
        f'(every panel = one cycle, {n_per_task} cycles per task)',
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    summary = ', '.join(f'{t}={len(v)}' for t, v in tasks_data.items())
    print(f'[grid]   wrote {out_path}  ({summary})')


# ---------- CLI ----------

@click.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False),
              default=None,
              help='Checkpoint-stem directory (parent of run dirs). '
                   'Required unless --grid.')
@click.option('--grid', is_flag=True,
              help='Also produce traces_grid figure across multiple tasks.')
@click.option('--out-dir', type=click.Path(), default=None,
              help='Output directory (default: <root>/paper_figs '
                   'or ./paper_figs for --grid).')
@click.option('--k-show', type=int, default=5,
              help='K to highlight in the bars figure (default 5).')
@click.option('--n-per-task', type=int, default=6,
              help='# example cycles per task in the grid (default 6).')
@click.option('--format', 'fmt', type=click.Choice(['pdf', 'png']),
              default='pdf')
def main(root, grid, out_dir, k_show, n_per_task, fmt):
    style()
    if root is None and not grid:
        sys.exit('provide --root, --grid, or both')

    if root is not None:
        root = pathlib.Path(root)
        single_out = pathlib.Path(out_dir) if out_dir else (root / 'paper_figs')
        single_out.mkdir(parents=True, exist_ok=True)
        runs = discover_runs(root)
        if not runs:
            sys.exit(f'no labeled runs found under {root}')
        fig_bars(runs, k_show, single_out / f'bars.{fmt}')
        fig_passk(runs, single_out / f'passk.{fmt}')
        fig_traces(runs, single_out / f'traces.{fmt}')

    if grid:
        grid_out = pathlib.Path(out_dir) if out_dir else pathlib.Path('paper_figs')
        grid_out.mkdir(parents=True, exist_ok=True)
        fig_variance_progression(grid_out / f'variance_progression.{fmt}')
        fig_traces_grid(grid_out / f'traces_grid.{fmt}',
                        n_per_task=n_per_task)


if __name__ == '__main__':
    main()
