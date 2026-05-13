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
    action_files = sorted(run_dir.glob('actions_[0-9]*.npy'))
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

    # Split arm joints (0..n_dof-2) from the gripper (last DOF) onto separate
    # y-axes. The gripper's std at open/close transitions can be ~30x larger
    # than typical arm-joint std, which would otherwise squash the arm panel.
    arm_dofs = list(range(n_dof - 1))
    grip_dof = n_dof - 1

    arm_std_max = max(np.std(a, axis=0)[:, arm_dofs].max() for _, _, a in cycles)
    grip_std_max = max(np.std(a, axis=0)[:, grip_dof].max() for _, _, a in cycles)
    y_top_arm = float(arm_std_max) * 1.1 + 1e-6
    y_top_grip = float(grip_std_max) * 1.1 + 1e-6

    fig, axes = plt.subplots(
        n_cycles, 3,
        figsize=(15, 2.4 * n_cycles),
        gridspec_kw={'width_ratios': [1.0, 1.2, 1.0]},
    )
    if n_cycles == 1:
        axes = axes[None, :]

    cmap = plt.get_cmap('tab10')

    for row, (idx, ff, actions) in enumerate(cycles):
        N = actions.shape[0]
        std_t = actions.std(axis=0)              # (n_act, 7)
        ts = np.arange(n_act)

        ax_img = axes[row, 0]
        img = plt.imread(str(ff))
        ax_img.imshow(img)
        ax_img.set_title(f'cycle {idx} obs ({ff.name})', fontsize=9)
        ax_img.axis('off')

        ax_arm = axes[row, 1]
        for d in arm_dofs:
            ax_arm.plot(ts, std_t[:, d], color=cmap(d % 10),
                        label=JOINT_LABELS[d] if d < len(JOINT_LABELS) else f'd{d}',
                        linewidth=1.4)
        ax_arm.set_xlim(0, n_act - 1)
        ax_arm.set_ylim(0, y_top_arm)
        ax_arm.set_xlabel('timestep within chunk')
        ax_arm.set_ylabel(f'std across {N} samples (rad)')
        ax_arm.set_title(f'cycle {idx}: arm sample std vs t', fontsize=9)
        ax_arm.grid(True, alpha=0.3)
        if row == 0:
            ax_arm.legend(loc='upper left', fontsize=7, ncol=2)

        ax_grip = axes[row, 2]
        grip_label = (JOINT_LABELS[grip_dof]
                      if grip_dof < len(JOINT_LABELS) else f'd{grip_dof}')
        ax_grip.plot(ts, std_t[:, grip_dof], color=cmap(grip_dof % 10),
                     label=grip_label, linewidth=1.4)
        ax_grip.set_xlim(0, n_act - 1)
        ax_grip.set_ylim(0, y_top_grip)
        ax_grip.set_xlabel('timestep within chunk')
        ax_grip.set_ylabel(f'std across {N} samples')
        ax_grip.set_title(f'cycle {idx}: gripper sample std vs t', fontsize=9)
        ax_grip.grid(True, alpha=0.3)
        if row == 0:
            ax_grip.legend(loc='upper left', fontsize=7)

    fig.tight_layout()
    out_path = pathlib.Path(out_path) if out_path else (run_dir / 'sample_variance.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'wrote {out_path}  ({n_cycles} cycles, n_act={n_act}, n_dof={n_dof}, '
          f'arm y_top={y_top_arm:.4f}, gripper y_top={y_top_grip:.4f})')


if __name__ == '__main__':
    main()
