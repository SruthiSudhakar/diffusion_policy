"""
Convert the PnPRedLegoToBrownBowl real-robot dataset to a diffusion_policy
zarr ReplayBuffer.

Source layout (per episode):
    <name>.npy              dict(trajectory[T,7], timestamps[T], frequency, image_paths[T])
    <name>_frames/          frame_000000.jpg ... (1280x720)

Output zarr structure:
    meta/episode_ends: int64 (N_episodes,)
    data/image:  uint8  (T_total, H, W, 3)
    data/action: float32 (T_total, 7)

Run with the jgdrobodiff env:
    /proj/vondrick3/sruthi/miniconda3/envs/jgdrobodiff/bin/python \
        scripts_pnp_lego/convert_pnp_to_zarr.py
"""
import argparse
import os
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
        default='/proj/vondrick3/datasets/VLMjgd/PnPRedLegoToBrownBowl')
    parser.add_argument('--dst', type=str,
        default=str(REPO_ROOT / 'data' / 'pnp_red_lego_to_brown_bowl' / 'replay.zarr'))
    parser.add_argument('--image-size', type=int, nargs=2, default=[640, 360],
        help='target (W, H) for resized RGB frames')
    parser.add_argument('--subsample', type=int, default=3,
        help='take every Nth frame (30 Hz / 3 = 10 Hz)')
    parser.add_argument('--success-only', action='store_true', default=True)
    parser.add_argument('--include-failures', dest='success_only',
        action='store_false')
    parser.add_argument('--max-episodes', type=int, default=None,
        help='for smoke-testing: only convert this many episodes')
    args = parser.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    target_w, target_h = args.image_size

    if dst.exists():
        raise SystemExit(f'destination already exists: {dst}\n'
            f'remove it first if you want to rebuild')
    dst.parent.mkdir(parents=True, exist_ok=True)

    prefix = 'success_' if args.success_only else ''
    all_npy = sorted(p for p in src.glob(f'{prefix}*.npy'))
    npy_files = []
    skipped = []
    for p in all_npy:
        frames_dir = p.with_name(p.stem + '_frames')
        if frames_dir.is_dir():
            npy_files.append(p)
        else:
            skipped.append(p.name)
    if skipped:
        print(f'skipping {len(skipped)} episodes with no frames dir: {skipped}')
    if args.max_episodes is not None:
        npy_files = npy_files[:args.max_episodes]
    print(f'found {len(npy_files)} episodes (success_only={args.success_only})')

    import zarr
    store = zarr.DirectoryStore(str(dst))
    buffer = ReplayBuffer.create_empty_zarr(storage=store)

    img_chunks = (1, target_h, target_w, 3)

    for npy_path in tqdm(npy_files, desc='episodes'):
        ep = load_episode(npy_path)
        traj = ep['trajectory']
        image_paths = ep['image_paths']
        T = len(traj)
        assert len(image_paths) == T, f'{npy_path.name}: trajectory/image len mismatch'

        idx = np.arange(0, T, args.subsample)
        traj_sub = traj[idx].astype(np.float32)

        images = np.empty((len(idx), target_h, target_w, 3), dtype=np.uint8)
        for k, t in enumerate(idx):
            img_path = src / image_paths[t]
            with Image.open(img_path) as im:
                im = im.convert('RGB').resize((target_w, target_h), Image.LANCZOS)
                images[k] = np.asarray(im, dtype=np.uint8)

        buffer.add_episode(
            data={'image': images, 'action': traj_sub},
            chunks={'image': img_chunks},
        )

    print(f'done: n_episodes={buffer.n_episodes} n_steps={buffer.n_steps}')
    print(f'zarr at: {dst}')


if __name__ == '__main__':
    main()
