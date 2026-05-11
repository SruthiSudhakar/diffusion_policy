"""Sanity test for the diversity selector in run_diffusion_policy_blocking.

The test builds a synthetic batch with five distinct arm-joint centroids,
each replicated 10 times (50 total candidates), and adds large random noise
to the gripper joint of every candidate. With JOINT_WEIGHTS_FOR_DIVERSITY
downweighting the gripper, greedy FPS should ignore the gripper noise and
recover one representative from each of the five centroid groups.

Run:
    cd /home/cvlabusers/Appaji/diffusion_policy
    python tests/test_select_diverse_indices.py
"""
import importlib.util
import pathlib
import sys

import numpy as np


def _load_selector():
    """Load select_diverse_indices without importing torch-heavy deps.

    The run script does `import torch` etc. at module level, which we don't
    need here; load just the symbols we need by exec'ing the relevant slice.
    """
    src = pathlib.Path(__file__).resolve().parent.parent / 'run_diffusion_policy_blocking.py'
    text = src.read_text()
    start = text.index('# Per-joint diversity weights')
    end = text.index('@click.command()')
    snippet = text[start:end]
    g = {'np': np}
    exec(snippet, g)
    return g['select_diverse_indices'], g['JOINT_WEIGHTS_FOR_DIVERSITY']


def main():
    select_diverse_indices, JOINT_WEIGHTS = _load_selector()

    rng = np.random.default_rng(0)
    M, T, D = 50, 8, 7
    n_centroids = 5
    per = M // n_centroids  # 10 per centroid

    # Five well-separated centroids in arm-joint space (joints 0-5).
    # Joint 6 (gripper) is zero in the centroid; we'll add big noise per-sample.
    centroids = rng.uniform(-1.0, 1.0, size=(n_centroids, D)).astype(np.float32)
    centroids[:, 6] = 0.0
    # Force them to be visibly far apart by scaling the arm-joint columns.
    centroids[:, :6] *= 2.0

    arr = np.zeros((M, T, D), dtype=np.float32)
    for c in range(n_centroids):
        for r in range(per):
            i = c * per + r
            # Constant arm pose across the chunk + small jitter.
            arm_pose = centroids[c, :6] + 0.01 * rng.standard_normal(6).astype(np.float32)
            arr[i, :, :6] = arm_pose[None, :]
            # Big random gripper noise that varies per timestep AND per sample.
            arr[i, :, 6] = 1.5 * rng.standard_normal(T).astype(np.float32)

    # Run with downweighted gripper.
    keep = select_diverse_indices(arr, k=n_centroids, seed_idx=0)
    keep_groups = sorted({i // per for i in keep})
    print(f'gripper-downweighted keep={keep}  groups={keep_groups}')
    assert keep_groups == [0, 1, 2, 3, 4], (
        f'Expected one sample per centroid group [0,1,2,3,4], got {keep_groups}. '
        'The gripper-downweighting did not actually filter out gripper noise.'
    )
    assert keep[0] == 0, f'seed_idx=0 should be kept first; got keep[0]={keep[0]}'

    # Counterfactual: with uniform joint weights (no gripper downweight),
    # the gripper noise dominates and FPS should NOT cleanly recover the
    # centroid groups. Use this as a contrast — we don't strictly assert
    # failure (sometimes random luck), but we report it.
    uniform_weights = np.ones(D, dtype=np.float32)
    keep_unweighted = select_diverse_indices(arr, k=n_centroids,
                                             joint_weights=uniform_weights,
                                             seed_idx=0)
    unweighted_groups = sorted({i // per for i in keep_unweighted})
    print(f'uniform-weights keep={keep_unweighted}  groups={unweighted_groups}')

    # Determinism check: same input -> same output.
    keep2 = select_diverse_indices(arr, k=n_centroids, seed_idx=0)
    assert keep == keep2, 'select_diverse_indices is not deterministic'

    # k == M edge case: should return a permutation of [0, M-1].
    keep_all = select_diverse_indices(arr, k=M, seed_idx=0)
    assert sorted(keep_all) == list(range(M)), 'k==M did not return all indices'

    # k == 1 edge case: returns just [seed_idx].
    keep_one = select_diverse_indices(arr, k=1, seed_idx=7)
    assert keep_one == [7], f'k=1 should return [seed_idx]; got {keep_one}'

    print('All sanity checks passed.')


if __name__ == '__main__':
    sys.exit(main())
