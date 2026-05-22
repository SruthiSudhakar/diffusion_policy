"""
Convert the UprightBottle real-robot dataset (with paired cam0/cam1 frames) to
a diffusion_policy zarr ReplayBuffer.

Source layout (per episode):
    <name>.npy              dict(
        trajectory[T,7],            # follower (measured) -> state
        leader_trajectory[T,7],     # leader (commanded)  -> action
        timestamps[T], frequency,
        image_paths_cam0[T],        # paths into <name>_frames_cam0/
        image_paths_cam1[T],        # paths into <name>_frames_cam1/
        camera_serials[2])
    <name>_frames_cam0/     frame_000000.jpg ...
    <name>_frames_cam1/     frame_000000.jpg ...

Output zarr structure:
    meta/episode_ends: int64 (N_episodes,)
    data/image1: uint8  (T_total, H, W, 3)   # cam0
    data/image2: uint8  (T_total, H, W, 3)   # cam1
    data/state:  float32 (T_total, 7)        # follower trajectory (observed)
    data/action: float32 (T_total, 7)        # leader trajectory (commanded)
"""
import argparse
import os
import re
import sys
import pathlib
import numpy as np

# The .npy files were saved with numpy >= 2.0, which references numpy._core.
# Map the legacy numpy.core into that namespace so numpy 1.x can unpickle them.
import numpy
sys.modules.setdefault('numpy._core', numpy.core)
sys.modules.setdefault('numpy._core.multiarray', numpy.core.multiarray)
sys.modules.setdefault('numpy._core.numeric', numpy.core.numeric)

from PIL import Image
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def load_episode(npy_path: pathlib.Path):
    return np.load(npy_path, allow_pickle=True).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,
        default='/proj/vondrick3/datasets/expert_data_jgd_UprightBottle')
    parser.add_argument('--dst', type=str,
        default=str(REPO_ROOT / 'data' / 'upright_bottle' / 'replay.zarr'))
    parser.add_argument('--image-size', type=int, nargs=2, default=[640, 360],
        help='target (W, H) for resized RGB frames')
    parser.add_argument('--subsample', type=int, default=2,
        help='take every Nth frame (30 Hz / 2 = 15 Hz)')
    parser.add_argument('--success-only', action='store_true', default=True)
    parser.add_argument('--include-failures', dest='success_only',
        action='store_false')
    parser.add_argument('--prefixes', type=str, nargs='+', default=None,
        help='if given, include all <prefix>_<num>_*.npy episodes for each '
             'prefix (overrides --success-only). e.g. bag glass remote')
    parser.add_argument('--max-episodes', type=int, default=None,
        help='for smoke-testing: only convert this many episodes')
    parser.add_argument('--episode-min', type=int, default=None,
        help='inclusive lower bound on episode number')
    parser.add_argument('--episode-max', type=int, default=None,
        help='inclusive upper bound on episode number')
    args = parser.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    target_w, target_h = args.image_size

    if dst.exists():
        raise SystemExit(f'destination already exists: {dst}\n'
            f'remove it first if you want to rebuild')
    dst.parent.mkdir(parents=True, exist_ok=True)

    if args.prefixes:
        prefix_group = '|'.join(re.escape(p) for p in args.prefixes)
        ep_re = re.compile(rf'^(?:{prefix_group})_(\d+)_')

        def ep_num(p):
            m = ep_re.match(p.name)
            return int(m.group(1)) if m else None

        candidates = []
        for pref in args.prefixes:
            candidates.extend(src.glob(f'{pref}_*.npy'))
        all_npy = sorted(
            (p for p in candidates if ep_num(p) is not None),
            key=lambda p: (p.name.split('_', 1)[0], ep_num(p)),
        )
    else:
        prefix = 'success_' if args.success_only else ''
        ep_re = re.compile(rf'^{prefix}(\d+)_') if prefix else re.compile(r'^(\d+)_')

        def ep_num(p):
            m = ep_re.match(p.name)
            return int(m.group(1)) if m else None

        all_npy = sorted(
            (p for p in src.glob(f'{prefix}*.npy') if ep_num(p) is not None),
            key=ep_num,
        )
    if args.episode_min is not None or args.episode_max is not None:
        lo = args.episode_min if args.episode_min is not None else -10**9
        hi = args.episode_max if args.episode_max is not None else 10**9
        all_npy = [p for p in all_npy if lo <= ep_num(p) <= hi]
    npy_files = []
    skipped = []
    for p in all_npy:
        frames_cam0 = p.with_name(p.stem + '_frames_cam0')
        frames_cam1 = p.with_name(p.stem + '_frames_cam1')
        if frames_cam0.is_dir() and frames_cam1.is_dir():
            npy_files.append(p)
        else:
            skipped.append(p.name)
    if skipped:
        print(f'skipping {len(skipped)} episodes with missing frames dirs: {skipped}')
    if args.max_episodes is not None:
        npy_files = npy_files[:args.max_episodes]
    if args.prefixes:
        print(f'found {len(npy_files)} episodes (prefixes={args.prefixes}, '
              f'episode range=[{args.episode_min}, {args.episode_max}])')
    else:
        print(f'found {len(npy_files)} episodes (success_only={args.success_only}, '
              f'episode range=[{args.episode_min}, {args.episode_max}])')

    import zarr
    store = zarr.DirectoryStore(str(dst))
    buffer = ReplayBuffer.create_empty_zarr(storage=store)

    img_chunks = (1, target_h, target_w, 3)

    for npy_path in tqdm(npy_files, desc='episodes'):
        ep = load_episode(npy_path)
        action_traj = ep['leader_trajectory']
        state_traj = ep['trajectory']
        image_paths_cam0 = ep['image_paths_cam0']
        image_paths_cam1 = ep['image_paths_cam1']
        T = len(action_traj)
        assert len(state_traj) == T, f'{npy_path.name}: state/action len mismatch'
        assert len(image_paths_cam0) == T, f'{npy_path.name}: cam0/traj len mismatch'
        assert len(image_paths_cam1) == T, f'{npy_path.name}: cam1/traj len mismatch'

        idx = np.arange(0, T, args.subsample)
        action_sub = action_traj[idx].astype(np.float32)
        state_sub = state_traj[idx].astype(np.float32)

        images_cam0 = np.empty((len(idx), target_h, target_w, 3), dtype=np.uint8)
        images_cam1 = np.empty((len(idx), target_h, target_w, 3), dtype=np.uint8)
        for k, t in enumerate(idx):
            for img_arr, paths in (
                    (images_cam0, image_paths_cam0),
                    (images_cam1, image_paths_cam1)):
                img_path = src / paths[t]
                with Image.open(img_path) as im:
                    im = im.convert('RGB').resize((target_w, target_h), Image.LANCZOS)
                    img_arr[k] = np.asarray(im, dtype=np.uint8)

        buffer.add_episode(
            data={
                'image1': images_cam0,
                'image2': images_cam1,
                'state': state_sub,
                'action': action_sub,
            },
            chunks={'image1': img_chunks, 'image2': img_chunks},
        )

    print(f'done: n_episodes={buffer.n_episodes} n_steps={buffer.n_steps}')
    print(f'zarr at: {dst}')


if __name__ == '__main__':
    main()
