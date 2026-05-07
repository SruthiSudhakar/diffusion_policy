"""
Visualize per-cycle std-across-samples for a run dir produced by
run_diffusion_policy_blocking.py.

Each row of the output figure = one cycle:
    [ frame_NNNNNN.jpg ]   [ std(t) per DOF, t=0..n_act-1 ]

Run:
    python visualize_run_stats.py <run_dir>
        [--out <png_path>]   # default: <run_dir>/sample_variance.png
"""
import pathlib
import sys

import click
import matplotlib.pyplot as plt
import numpy as np


JOINT_LABELS = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'gripper']


@click.command()
@click.argument('run_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--out', 'out_path', type=click.Path(), default=None,
              help='Output PNG path (default: <run_dir>/sample_variance.png).')
def main(run_dir, out_path):
    run_dir = pathlib.Path(run_dir)
    action_files = sorted(run_dir.glob('actions_*.npy'))
    if not action_files:
        sys.exit(f'no actions_*.npy under {run_dir}')

    obs_dir = run_dir / 'observations'
    frame_dir = obs_dir if obs_dir.is_dir() else run_dir

    cycles = []
    for af in action_files:
        idx = int(af.stem.split('_')[1])
        ff = frame_dir / f'frame_{idx:06d}.jpg'
        if not ff.exists():
            print(f'skip cycle {idx}: missing {ff}')
            continue
        actions = np.load(af)        # (N, n_act, 7)
        cycles.append((idx, ff, actions))
    if not cycles:
        sys.exit('no cycle had both image and actions')

    n_cycles = len(cycles)
    n_act = cycles[0][2].shape[1]
    n_dof = cycles[0][2].shape[2]

    # Shared y-limit so std magnitudes are comparable across cycles.
    all_std_max = max(np.std(a, axis=0).max() for _, _, a in cycles)
    y_top = float(all_std_max) * 1.1 + 1e-6

    fig, axes = plt.subplots(
        n_cycles, 2,
        figsize=(12, 2.4 * n_cycles),
        gridspec_kw={'width_ratios': [1.0, 1.4]},
    )
    if n_cycles == 1:
        axes = axes[None, :]

    cmap = plt.get_cmap('tab10')

    for row, (idx, ff, actions) in enumerate(cycles):
        N = actions.shape[0]
        std_t = actions.std(axis=0)              # (n_act, 7)

        ax_img = axes[row, 0]
        img = plt.imread(str(ff))
        ax_img.imshow(img)
        ax_img.set_title(f'cycle {idx} obs ({ff.name})', fontsize=9)
        ax_img.axis('off')

        ax_std = axes[row, 1]
        ts = np.arange(n_act)
        for d in range(n_dof):
            ax_std.plot(ts, std_t[:, d], color=cmap(d % 10),
                        label=JOINT_LABELS[d] if d < len(JOINT_LABELS) else f'd{d}',
                        linewidth=1.4)
        ax_std.set_xlim(0, n_act - 1)
        ax_std.set_ylim(0, y_top)
        ax_std.set_xlabel('timestep within chunk')
        ax_std.set_ylabel(f'std across {N} samples')
        ax_std.set_title(f'cycle {idx}: per-DOF sample std vs t', fontsize=9)
        ax_std.grid(True, alpha=0.3)
        if row == 0:
            ax_std.legend(loc='upper left', fontsize=7, ncol=2)

    fig.tight_layout()
    out_path = pathlib.Path(out_path) if out_path else (run_dir / 'sample_variance.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'wrote {out_path}  ({n_cycles} cycles, n_act={n_act}, n_dof={n_dof}, '
          f'shared y_top={y_top:.4f})')


if __name__ == '__main__':
    main()
