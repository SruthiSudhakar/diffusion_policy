"""
Convert the UprightBottle real-robot dataset (with paired cam0/cam1 frames) to
a diffusion_policy zarr ReplayBuffer.

Source layout (per episode):
    <name>.npy              dict(
        trajectory[T,7],            # ignored
        leader_trajectory[T,7],     # used for both state and action
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
    data/state:  float32 (T_total, 7)        # leader_trajectory
    data/action: float32 (T_total, 7)        # leader_trajectory
"""
import argparse
import json
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

DEFAULT_LABEL_TO_PROMPT = {
    'box': 'pick up the box',
    'glass': 'pick up the glass',
    'remote': 'pick up the remote',
}


def load_episode(npy_path: pathlib.Path):
    return np.load(npy_path, allow_pickle=True).item()


def encode_label_prompts(label_to_prompt, model_name, device='cpu'):
    """Run each unique prompt through a CLIP text encoder once.

    Returns {label: np.ndarray of shape (D,) float32}.
    """
    import torch
    from transformers import CLIPTokenizer, CLIPTextModel
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    # use_safetensors=True avoids torch.load(weights_only=...) which doesn't
    # exist on torch<2.0 (this env has torch 1.12).
    model = CLIPTextModel.from_pretrained(
        model_name, use_safetensors=True).to(device).eval()
    out = {}
    with torch.no_grad():
        for label, prompt in label_to_prompt.items():
            tok = tokenizer([prompt], padding=True, return_tensors='pt').to(device)
            emb = model(**tok).pooler_output[0].cpu().numpy().astype(np.float32)
            out[label] = emb
    del model
    return out


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
    parser.add_argument('--clip-model', type=str,
        default='openai/clip-vit-base-patch32',
        help='HF model id for the CLIP text encoder used to build the '
             'per-episode text conditioning. Default is 512-dim.')
    parser.add_argument('--label-map', type=str, default=None,
        help='JSON dict mapping {label: prompt}. Overrides the default '
             'box/glass/remote -> "pick up the X" map.')
    parser.add_argument('--no-text-cond', action='store_true', default=False,
        help='skip CLIP text-embed computation (matches the pre-text zarr layout)')
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
        ep_re = re.compile(rf'^({prefix_group})_(\d+)_')

        def ep_num(p):
            m = ep_re.match(p.name)
            return int(m.group(2)) if m else None

        def ep_label(p):
            m = ep_re.match(p.name)
            return m.group(1) if m else None

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

        def ep_label(p):
            return None

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

    text_cond_enabled = bool(args.prefixes) and not args.no_text_cond
    if text_cond_enabled:
        if args.label_map is not None:
            label_to_prompt = json.loads(args.label_map)
        else:
            label_to_prompt = {p: DEFAULT_LABEL_TO_PROMPT.get(p, p)
                               for p in args.prefixes}
        missing = [p for p in args.prefixes if p not in label_to_prompt]
        if missing:
            raise SystemExit(f'label_map missing prompts for prefixes: {missing}')
        print(f'CLIP text encoder: {args.clip_model}')
        for k, v in label_to_prompt.items():
            print(f'  {k!r:>10} -> {v!r}')
        prompt_embeds = encode_label_prompts(label_to_prompt, args.clip_model)
        text_embed_dim = next(iter(prompt_embeds.values())).shape[0]
        print(f'CLIP text embed dim: {text_embed_dim}')
    else:
        prompt_embeds = {}
        text_embed_dim = 0

    import zarr
    store = zarr.DirectoryStore(str(dst))
    buffer = ReplayBuffer.create_empty_zarr(storage=store)

    img_chunks = (1, target_h, target_w, 3)
    episode_labels = []
    episode_text_embeds = []

    for npy_path in tqdm(npy_files, desc='episodes'):
        ep = load_episode(npy_path)
        traj = ep['leader_trajectory']
        image_paths_cam0 = ep['image_paths_cam0']
        image_paths_cam1 = ep['image_paths_cam1']
        T = len(traj)
        assert len(image_paths_cam0) == T, f'{npy_path.name}: cam0/traj len mismatch'
        assert len(image_paths_cam1) == T, f'{npy_path.name}: cam1/traj len mismatch'

        idx = np.arange(0, T, args.subsample)
        traj_sub = traj[idx].astype(np.float32)
        state_sub = traj_sub.copy()

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
                'action': traj_sub,
            },
            chunks={'image1': img_chunks, 'image2': img_chunks},
        )
        if text_cond_enabled:
            label = ep_label(npy_path)
            if label is None or label not in prompt_embeds:
                raise RuntimeError(
                    f'{npy_path.name}: could not derive a prompt label '
                    f'(label={label!r}, known={list(prompt_embeds)})')
            episode_labels.append(label)
            episode_text_embeds.append(prompt_embeds[label])

    if text_cond_enabled:
        label_arr = np.array(episode_labels)            # fixed-width unicode
        embed_arr = np.stack(episode_text_embeds).astype(np.float32)
        buffer.update_meta({
            'episode_labels': label_arr,
            'episode_text_embed': embed_arr,
        })
        from collections import Counter
        counts = Counter(episode_labels)
        print(f'text conditioning: episode_text_embed{tuple(embed_arr.shape)}, '
              f'label counts={dict(counts)}')

    print(f'done: n_episodes={buffer.n_episodes} n_steps={buffer.n_steps}')
    print(f'zarr at: {dst}')


if __name__ == '__main__':
    main()
