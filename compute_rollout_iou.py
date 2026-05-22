"""
End-to-end rollout IoU evaluator + method comparison.

For every rollout subdir under --root:
  1. Walk rollout_image2.mp4 frame-by-frame.
  2. Detect the white bowl in each frame (classical CV; see detect_bowl below).
  3. Compute IoU with the global target_mask.png.
  4. Track the max-IoU frame per video.

Outputs (all in --out):
  <rollout_id>_max_iou_mask.png   binary bowl mask at the max-IoU frame
  <rollout_id>_max_iou_vis.png    overlay visualization at that frame
  rollout_iou_summary.csv         per-rollout summary, sorted by best_iou desc

Immediately after the CSV is written, the script also produces a method
comparison (VLM vs nonVLM, split by the `_VLM` suffix on rollout_id) under
<out>/analysis/:
  method_stats.csv / .txt      per-method summary stats
  paired_iou.csv               wide table of (trial_id, iou_vlm, iou_nonvlm, diff)
  iou_hist.png                 overlaid histograms
  iou_box.png                  box + jittered points
  iou_paired_scatter.png       VLM vs nonVLM scatter w/ y=x line
  iou_ecdf.png                 empirical CDFs

Usage:
python compute_rollout_iou.py \
    --root data/jgd/2026.05.18/15.12.41_train_diffusion_unet_hybrid_push_bowl_image_only_trajectory/checkpoints/epoch=0250-train_loss=0.0247 \
    --target-mask target_mask_image1.png \
    --table-mask  table_mask_image1.png \
    --video-name  rollout_image1.mp4 \
    --out  ./iou_outputs_image1

python compute_rollout_iou.py \
    --root data/jgd/2026.05.18/15.12.41_train_diffusion_unet_hybrid_push_bowl_image_only_trajectory/checkpoints/epoch=0250-train_loss=0.0247 \
    --target-mask target_mask_image2.png \
    --table-mask  table_mask_image2.png \
    --video-name  rollout_image2.mp4 \
    --out  ./iou_outputs_image2
"""

import argparse
import csv
import glob
import os
from multiprocessing import Pool, cpu_count

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# Bowl detector (classical CV, no training)
# ============================================================================

# Strict "core blue": uniquely the wrapped candies (bright, saturated blue).
CORE_BLUE_LO = np.array([90, 100, 150], dtype=np.uint8)
CORE_BLUE_HI = np.array([115, 255, 255], dtype=np.uint8)

# Broader blue: only used adjacent to the core seed.
WIDE_BLUE_LO = np.array([85, 40, 110], dtype=np.uint8)
WIDE_BLUE_HI = np.array([120, 255, 255], dtype=np.uint8)

# White: bowl rim and interior (low saturation, high value).
WHITE_LO = np.array([0, 0, 165], dtype=np.uint8)
WHITE_HI = np.array([180, 70, 255], dtype=np.uint8)

# Minimum strict-blue pixels to call the bowl "present".
MIN_CORE_PIXELS = 120

# Tight bowl rim: white pixels at most this far from candy contents (px).
RIM_THICKNESS = 13

# Final outward dilation applied to the fitted ellipse (px).
ELLIPSE_PAD = 4


def _largest_cluster(mask: np.ndarray, link_radius: int = 30) -> np.ndarray:
    """Return the largest connected blob plus any smaller blobs whose centroid
    lies within `link_radius` of the main blob's bounding box."""
    n, lbls, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    out = (lbls == main).astype(np.uint8) * 255

    mx = stats[main, cv2.CC_STAT_LEFT] + stats[main, cv2.CC_STAT_WIDTH] / 2
    my = stats[main, cv2.CC_STAT_TOP] + stats[main, cv2.CC_STAT_HEIGHT] / 2
    for i in range(1, n):
        if i == main:
            continue
        ix = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
        iy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
        if (ix - mx) ** 2 + (iy - my) ** 2 < link_radius * link_radius:
            out[lbls == i] = 255
    return out


def detect_bowl(img_bgr: np.ndarray,
                fixed_circle_mask: np.ndarray,
                rim_thickness: int = RIM_THICKNESS,
                ellipse_pad: int = ELLIPSE_PAD) -> tuple[np.ndarray, bool]:
    """
    Args:
        img_bgr: HxWx3 BGR uint8 image.
        fixed_circle_mask: HxW uint8.  Non-zero where the fixed white table
            circle lives.
        rim_thickness: pixels.  Max distance from a candy pixel at which we
            still trust a white pixel as bowl rim.
        ellipse_pad: px of outward padding around the fitted bowl ellipse.

    Returns:
        (bowl_mask, found)
        bowl_mask: HxW uint8 in {0, 255} covering rim + interior + contents.
        found: False iff effectively no bowl is present in the frame.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Strict "core blue" seed — unmistakably the candies.
    core = cv2.inRange(hsv, CORE_BLUE_LO, CORE_BLUE_HI)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if int(core.sum() // 255) < MIN_CORE_PIXELS:
        return np.zeros((h, w), dtype=np.uint8), False
    core = _largest_cluster(core, link_radius=60)

    # Wider blue restricted to a neighborhood of the seed (candy edges).
    wide = cv2.inRange(hsv, WIDE_BLUE_LO, WIDE_BLUE_HI)
    seed_region = cv2.dilate(
        core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    blue_full = cv2.bitwise_and(wide, seed_region)
    blue_full = cv2.morphologyEx(
        blue_full, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # White mask.
    white = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Tight bowl-rim mask: white pixels within `rim_thickness` of candy contents.
    k = 2 * rim_thickness + 1
    near = cv2.dilate(blue_full,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    rim = cv2.bitwise_and(white, near)

    # Fit an ellipse to (candy seed + tight rim) — the bowl shape.
    bowl_evidence = cv2.bitwise_or(rim, blue_full)
    bowl_evidence = cv2.morphologyEx(
        bowl_evidence, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    pts = cv2.findNonZero(bowl_evidence)
    if pts is None or len(pts) < 20:
        return np.zeros((h, w), dtype=np.uint8), False

    if len(pts) >= 5:
        ellipse = cv2.fitEllipse(pts)
        (cx, cy), (a, b), ang = ellipse
        ellipse = ((cx, cy),
                   (a + 2 * ellipse_pad, b + 2 * ellipse_pad),
                   ang)
        bowl = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(bowl, ellipse, 255, thickness=cv2.FILLED)
    else:
        (cx, cy), r = cv2.minEnclosingCircle(pts)
        bowl = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(bowl, (int(cx), int(cy)), int(r) + ellipse_pad, 255, -1)

    bowl = cv2.bitwise_or(bowl, blue_full)  # always include candies
    return bowl, True


# ============================================================================
# Per-video processing (with multiprocessing workers)
# ============================================================================

# Worker-side globals populated by _worker_init so masks are not reserialized
# per task.
_TARGET_MASK = None
_TABLE_MASK = None


def _binarize(mask_gray: np.ndarray) -> np.ndarray:
    return (mask_gray > 127).astype(np.uint8) * 255


def _resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == target_hw:
        return mask
    h, w = target_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a > 0
    b_b = b > 0
    inter = int(np.logical_and(a_b, b_b).sum())
    union = int(np.logical_or(a_b, b_b).sum())
    if union == 0:
        return 0.0
    return inter / union


def _make_vis(frame_bgr: np.ndarray,
              bowl_mask: np.ndarray,
              target_mask: np.ndarray,
              label: str) -> np.ndarray:
    overlay = frame_bgr.copy()
    overlay[bowl_mask > 0] = (0, 0, 255)  # red bowl
    vis = cv2.addWeighted(frame_bgr, 0.55, overlay, 0.45, 0)
    tgt_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, tgt_contours, -1, (0, 255, 0), 2)
    bowl_contours, _ = cv2.findContours(bowl_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, bowl_contours, -1, (255, 255, 255), 1)
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    return vis


def _worker_init(target_mask: np.ndarray, table_mask: np.ndarray):
    global _TARGET_MASK, _TABLE_MASK
    _TARGET_MASK = target_mask
    _TABLE_MASK = table_mask
    cv2.setNumThreads(1)


def _process_one(arg) -> dict:
    idx, total, subdir, video_name, out_dir = arg
    rollout_id = os.path.basename(subdir.rstrip('/'))
    video_path = os.path.join(subdir, video_name)
    if not os.path.isfile(video_path):
        return {
            'rollout_id': rollout_id, 'num_frames': 0, 'frames_with_bowl': 0,
            'best_iou': 0.0, 'best_frame_idx': -1, 'video_path': video_path,
            'status': 'missing_video',
        }

    res = process_video(video_path, _TARGET_MASK, _TABLE_MASK)

    status = 'ok'
    if res['best_mask'] is not None and res['best_frame_idx'] >= 0:
        mask_path = os.path.join(out_dir, f'{rollout_id}_max_iou_mask.png')
        vis_path = os.path.join(out_dir, f'{rollout_id}_max_iou_vis.png')
        cv2.imwrite(mask_path, res['best_mask'])
        label = (f"{rollout_id}  iou={res['best_iou']:.3f}  "
                 f"frame={res['best_frame_idx']}/{res['num_frames']}")
        vis = _make_vis(res['best_frame'], res['best_mask'],
                        res['target_resized'], label)
        cv2.imwrite(vis_path, vis)
    else:
        status = 'no_bowl_detected'

    print(f'[{idx}/{total}] {rollout_id}: iou={res["best_iou"]:.4f} '
          f'frame={res["best_frame_idx"]}/{res["num_frames"]} '
          f'found_in={res["frames_with_bowl"]}', flush=True)

    return {
        'rollout_id': rollout_id,
        'num_frames': res['num_frames'],
        'frames_with_bowl': res['frames_with_bowl'],
        'best_iou': round(res['best_iou'], 6),
        'best_frame_idx': res['best_frame_idx'],
        'video_path': video_path,
        'status': status,
    }


def process_video(video_path: str,
                  target_mask: np.ndarray,
                  table_mask: np.ndarray) -> dict:
    """Walk one video; return summary dict including best-frame artifacts."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'num_frames': 0, 'frames_with_bowl': 0,
            'best_iou': 0.0, 'best_frame_idx': -1,
            'best_frame': None, 'best_mask': None,
            'target_resized': target_mask, 'table_resized': table_mask,
        }

    best_iou = -1.0
    best_idx = -1
    best_frame = None
    best_mask = None
    frames_with_bowl = 0
    n_frames = 0

    tgt_resized = None
    tbl_resized = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1

        if tgt_resized is None:
            h, w = frame.shape[:2]
            tgt_resized = _resize_mask(target_mask, (h, w))
            tbl_resized = _resize_mask(table_mask, (h, w))

        bowl, found = detect_bowl(frame, tbl_resized)
        if found:
            frames_with_bowl += 1

        iou = _iou(bowl, tgt_resized)
        if iou > best_iou:
            best_iou = iou
            best_idx = n_frames - 1
            best_frame = frame.copy()
            best_mask = bowl.copy()

    cap.release()

    return {
        'num_frames': n_frames,
        'frames_with_bowl': frames_with_bowl,
        'best_iou': max(best_iou, 0.0),
        'best_frame_idx': best_idx,
        'best_frame': best_frame,
        'best_mask': best_mask,
        'target_resized': tgt_resized if tgt_resized is not None else target_mask,
        'table_resized': tbl_resized if tbl_resized is not None else table_mask,
    }


# ============================================================================
# Method comparison: VLM vs nonVLM
# ============================================================================

def _per_method_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby('method'):
        iou = sub['best_iou'].to_numpy()
        bowl_vis_frac = (sub['frames_with_bowl'] /
                         sub['num_frames'].replace(0, np.nan)).to_numpy()
        if 'status' in sub.columns:
            n_no_bowl = int((sub['status'] == 'no_bowl_detected').sum())
        else:
            n_no_bowl = int((sub['frames_with_bowl'] == 0).sum())
        rows.append({
            'method': method,
            'count': len(iou),
            'mean': float(np.mean(iou)),
            'median': float(np.median(iou)),
            'std': float(np.std(iou, ddof=1)) if len(iou) > 1 else 0.0,
            'min': float(np.min(iou)),
            'max': float(np.max(iou)),
            'q25': float(np.quantile(iou, 0.25)),
            'q75': float(np.quantile(iou, 0.75)),
            'n_zero_iou': int((iou == 0.0).sum()),
            'n_no_bowl': n_no_bowl,
            'mean_bowl_visible_frac': float(np.nanmean(bowl_vis_frac)),
        })
    out = pd.DataFrame(rows).set_index('method')
    return out.reindex([m for m in ['VLM', 'nonVLM'] if m in out.index])


def _paired_table(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index='trial_id', columns='method',
                          values='best_iou', aggfunc='first')
    cols_needed = {'VLM', 'nonVLM'}
    if not cols_needed.issubset(wide.columns):
        return pd.DataFrame(columns=['trial_id', 'iou_vlm', 'iou_nonvlm', 'diff'])
    wide = wide.dropna(subset=['VLM', 'nonVLM'])
    wide['diff'] = wide['VLM'] - wide['nonVLM']
    return wide.rename(columns={'VLM': 'iou_vlm',
                                'nonVLM': 'iou_nonvlm'}).reset_index()


def _plot_hist(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 21)
    for method, color in [('VLM', 'tab:blue'), ('nonVLM', 'tab:orange')]:
        vals = df.loc[df.method == method, 'best_iou'].to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.55, label=f'{method} (n={len(vals)})',
                color=color, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('best_iou')
    ax.set_ylabel('count')
    ax.set_title('Per-rollout max bowl-target IoU by method')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_box(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    methods = [m for m in ['VLM', 'nonVLM'] if (df.method == m).any()]
    data = [df.loc[df.method == m, 'best_iou'].to_numpy() for m in methods]
    ax.boxplot(data, labels=methods, widths=0.5, showmeans=True,
               meanprops={'marker': 'D', 'markerfacecolor': 'k',
                          'markeredgecolor': 'k', 'markersize': 6})
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0, 0.04, size=len(vals))
        color = 'tab:blue' if methods[i - 1] == 'VLM' else 'tab:orange'
        ax.scatter(np.full_like(vals, i) + jitter, vals, alpha=0.55,
                   s=22, color=color, edgecolor='none')
    ax.set_ylabel('best_iou')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('best_iou by method (box + jittered points; diamond = mean)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_paired_scatter(paired: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1,
            label='y = x')
    if len(paired):
        x = paired['iou_nonvlm'].to_numpy()
        y = paired['iou_vlm'].to_numpy()
        ax.scatter(x, y, s=36, alpha=0.7, edgecolor='black', linewidth=0.4)
        i_max = int(paired['diff'].abs().idxmax())
        ax.annotate(paired.loc[i_max, 'trial_id'],
                    (paired.loc[i_max, 'iou_nonvlm'],
                     paired.loc[i_max, 'iou_vlm']),
                    textcoords='offset points', xytext=(6, 6), fontsize=8,
                    color='tab:red')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel('nonVLM best_iou')
    ax.set_ylabel('VLM best_iou')
    n_vlm = int((paired['diff'] > 0).sum()) if len(paired) else 0
    n_non = int((paired['diff'] < 0).sum()) if len(paired) else 0
    n_tie = int((paired['diff'] == 0).sum()) if len(paired) else 0
    ax.set_title(f'Paired by trial_id (n={len(paired)})\n'
                 f'VLM wins: {n_vlm}   nonVLM wins: {n_non}   ties: {n_tie}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_ecdf(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, color in [('VLM', 'tab:blue'), ('nonVLM', 'tab:orange')]:
        vals = np.sort(df.loc[df.method == method, 'best_iou'].to_numpy())
        if len(vals) == 0:
            continue
        ys = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, ys, where='post', label=f'{method} (n={len(vals)})',
                color=color, linewidth=2)
    ax.set_xlabel('best_iou')
    ax.set_ylabel('ECDF')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Empirical CDF of best_iou by method')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_analysis(csv_path: str, out_dir: str) -> None:
    """Read the summary CSV and emit VLM-vs-nonVLM stats + plots."""
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df['method'] = np.where(df['rollout_id'].str.endswith('_VLM'),
                            'VLM', 'nonVLM')
    df['trial_id'] = df['rollout_id'].str.split('_', n=1).str[0]

    stats = _per_method_stats(df)
    stats.to_csv(os.path.join(out_dir, 'method_stats.csv'))
    with open(os.path.join(out_dir, 'method_stats.txt'), 'w') as f:
        f.write(stats.round(4).to_string())
        f.write('\n')

    paired = _paired_table(df)
    paired.to_csv(os.path.join(out_dir, 'paired_iou.csv'), index=False)

    counts = df.groupby('trial_id')['method'].nunique()
    unpaired = counts[counts < 2].index.tolist()

    _plot_hist(df, os.path.join(out_dir, 'iou_hist.png'))
    _plot_box(df, os.path.join(out_dir, 'iou_box.png'))
    _plot_paired_scatter(paired, os.path.join(out_dir, 'iou_paired_scatter.png'))
    _plot_ecdf(df, os.path.join(out_dir, 'iou_ecdf.png'))

    print('=' * 72)
    print('Per-method stats (best_iou):')
    print('=' * 72)
    print(stats.round(4).to_string())
    print()
    print('=' * 72)
    print(f'Paired comparison  (n={len(paired)} matched trial_ids)')
    print('=' * 72)
    if len(paired):
        diff = paired['diff'].to_numpy()
        print(f'mean(VLM - nonVLM)   = {diff.mean():+.4f}')
        print(f'median(VLM - nonVLM) = {np.median(diff):+.4f}')
        print(f'VLM > nonVLM in {int((diff > 0).sum())} pairs   '
              f'(ties: {int((diff == 0).sum())},  '
              f'nonVLM > VLM in {int((diff < 0).sum())})')
        try:
            from scipy.stats import wilcoxon
            nonzero = diff[diff != 0]
            if len(nonzero) > 0:
                stat, p = wilcoxon(nonzero)
                print(f'Wilcoxon signed-rank (non-zero pairs n={len(nonzero)}): '
                      f'W={stat:.2f}, p={p:.4g}')
        except ImportError:
            print('(scipy not installed -> skipping Wilcoxon p-value)')
    if unpaired:
        print(f'Warning: unpaired trial_ids skipped from paired test: {unpaired}')
    print(f'\nAnalysis outputs written to: {out_dir}')


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='parent dir containing rollout subdirs')
    ap.add_argument('--target-mask', required=True,
                    help='path to goal-region binary mask png')
    ap.add_argument('--table-mask', required=True,
                    help='path to fixed table-circle mask (input to detect_bowl)')
    ap.add_argument('--video-name', default='rollout_image2.mp4',
                    help='video filename to look for in each subdir')
    ap.add_argument('--out', required=True,
                    help='output dir for per-rollout artifacts + summary csv')
    ap.add_argument('--limit', type=int, default=0,
                    help='if >0, only process the first N rollouts (smoke test)')
    ap.add_argument('--workers', type=int, default=0,
                    help='parallel worker processes (0 = min(cpu_count, num_rollouts))')
    ap.add_argument('--skip-analysis', action='store_true',
                    help='if set, skip the VLM-vs-nonVLM analysis at the end')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    target_raw = cv2.imread(args.target_mask, cv2.IMREAD_GRAYSCALE)
    table_raw = cv2.imread(args.table_mask, cv2.IMREAD_GRAYSCALE)
    assert target_raw is not None, f'could not read target mask {args.target_mask}'
    assert table_raw is not None, f'could not read table mask {args.table_mask}'
    target_mask = _binarize(target_raw)
    table_mask = _binarize(table_raw)

    subdirs = sorted(
        d for d in glob.glob(os.path.join(args.root, '*'))
        if os.path.isdir(d)
    )
    if args.limit > 0:
        subdirs = subdirs[:args.limit]

    tasks = [
        (i + 1, len(subdirs), subdir, args.video_name, args.out)
        for i, subdir in enumerate(subdirs)
    ]

    n_workers = args.workers if args.workers > 0 else min(cpu_count(), len(tasks))
    n_workers = max(1, n_workers)
    print(f'Processing {len(tasks)} rollouts with {n_workers} workers...',
          flush=True)

    rows = []
    with Pool(processes=n_workers,
              initializer=_worker_init,
              initargs=(target_mask, table_mask)) as pool:
        for row in pool.imap_unordered(_process_one, tasks):
            rows.append(row)

    rows.sort(key=lambda r: r['best_iou'], reverse=True)
    csv_path = os.path.join(args.out, 'rollout_iou_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(
            f,
            fieldnames=['rollout_id', 'num_frames', 'frames_with_bowl',
                        'best_iou', 'best_frame_idx', 'video_path', 'status'])
        wr.writeheader()
        wr.writerows(rows)

    print(f'\nDone. {len(rows)} rollouts -> {csv_path}')

    if not args.skip_analysis:
        print()
        run_analysis(csv_path, os.path.join(args.out, 'analysis'))


if __name__ == '__main__':
    main()
