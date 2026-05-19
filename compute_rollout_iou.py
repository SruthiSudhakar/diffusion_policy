"""Compute per-rollout max-IoU between the bowl footprint and a fixed target
mask for every rollout subdir under a checkpoint directory.

Three bowl detectors that use a painted bowl reference:
  * --detector orb      : ORB feature matching + RANSAC homography
                          (RECOMMENDED for textured bowls — robust to
                          brightness/scale and rejects uniform-colour
                          distractors like the target plate).
  * --detector template : masked template matching; useful when ORB finds
                          too few features (e.g. a plain bowl).
  * --detector hsv      : legacy saturation gate, no painted template
                          required.

Painted-template workflow:
  1. python paint_target_mask.py --image <ref_frame.jpg> --out bowl_mask.png
     (paint over the bowl on a clean frame, save).
  2. Pass --bowl-template-frame <ref_frame.jpg> --bowl-template-mask bowl_mask.png
     here. With --detector orb the script warps the painted shape into each
     frame via the recovered homography and uses the warped shape as the
     per-frame bowl footprint.


python compute_rollout_iou.py \
--root <epoch_dir> \
--video image2 \
--target-mask target_mask.png \
--detector orb \
--bowl-template-frame data/jgd/2026.05.18/15.12.41_train_diffusion_unet_hybrid_push_bowl_image_only_trajectory/checkpoints/epoch=0250-train_loss=0.0247/37jgd_f_2026-05-19_16-42-24_VLM/observations/image2/frame_000000.jpg \
--bowl-template-mask bowl_mask.png \
--bowl-mask template \
--bowl-roi 180,100,540,310 \
--skip-frames 0 --stride 1 \
--save-overlay --workers 8

"""


import argparse
import csv
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def parse_triplet(s: str, name: str) -> Tuple[int, int, int]:
    parts = s.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f'{name} expects "a,b,c", got {s!r}')
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def parse_target(s: str) -> Tuple[int, int, int]:
    return parse_triplet(s, '--target')


def parse_hsv(s: str) -> Tuple[int, int, int]:
    return parse_triplet(s, 'hsv bound')


def parse_scales(s: str) -> Tuple[float, ...]:
    """Parse the --template-scales spec. Accepts either a comma list of
    explicit scales ("0.7,1.0,1.3") or a linspace range "min:max:n"
    ("0.7:1.3:7")."""
    if ':' in s:
        parts = s.split(':')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f'--template-scales range expects "min:max:n", got {s!r}')
        lo = float(parts[0]); hi = float(parts[1]); n = int(parts[2])
        if n < 1 or hi < lo or lo <= 0:
            raise argparse.ArgumentTypeError(
                f'bad scale range {s!r} (need 0<lo<=hi, n>=1)')
        if n == 1:
            return (lo,)
        return tuple(float(x) for x in np.linspace(lo, hi, n))
    parts = [float(p) for p in s.split(',') if p.strip()]
    if not parts or any(p <= 0 for p in parts):
        raise argparse.ArgumentTypeError(f'bad --template-scales {s!r}')
    return tuple(parts)


def build_target_mask(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, thickness=-1)
    return m


def auto_detect_target(
    frame_bgr: np.ndarray,
    search_bbox: Tuple[int, int, int, int],
    v_min: int = 130,
    s_max: int = 90,
    min_area: int = 300,
) -> Optional[Tuple[int, int, int]]:
    """Detect the white target plate as the largest desaturated bright blob
    inside `search_bbox`. Returns (cx, cy, r_equivalent) or None.
    `r_equivalent` is derived from blob area: r = sqrt(area/pi)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    x1, y1, x2, y2 = search_bbox
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    mask = np.zeros((H, W), dtype=np.uint8)
    sub_v = hsv[y1:y2, x1:x2, 2]
    sub_s = hsv[y1:y2, x1:x2, 1]
    mask[y1:y2, x1:x2] = ((sub_v > v_min) & (sub_s < s_max)).astype(
        np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    a = int(stats[best, cv2.CC_STAT_AREA])
    if a < min_area:
        return None
    cx, cy = centroids[best]
    r = int(round(float(np.sqrt(a / np.pi))))
    return int(round(cx)), int(round(cy)), r


def build_orb_state(
    template_bgr: np.ndarray,
    template_mask: np.ndarray,
    nfeatures: int = 2000,
) -> Optional[dict]:
    """Detect ORB keypoints inside the painted bowl region of the template.
    Returns a dict carrying the detector + template keypoints/descriptors,
    or None if the template is too featureless to use ORB."""
    orb = cv2.ORB_create(
        nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
        edgeThreshold=15, fastThreshold=10)
    kp, des = orb.detectAndCompute(template_bgr, mask=template_mask)
    if des is None or len(kp) < 4:
        return None
    return dict(
        orb=orb, kp=kp, des=des,
        tpl_h=int(template_bgr.shape[0]),
        tpl_w=int(template_bgr.shape[1]),
        tpl_mask=template_mask)


def detect_bowl_orb(
    frame_bgr: np.ndarray,
    orb_state: dict,
    min_inliers: int,
    ratio_test: float = 0.75,
    ransac_reproj: float = 5.0,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[np.ndarray, Tuple[float, float], float, int]]:
    """Locate the bowl via ORB feature matching + RANSAC homography.

    The bowl's distinctive interior pattern (e.g. coloured decoration on a
    light bowl) produces many ORB keypoints; uniform-coloured distractors
    such as a target plate produce none of the matched descriptors and are
    therefore ignored. Homography natively handles scale and small pose
    changes, so this detector does not need --template-scales.

    Returns (warped_bowl_mask, centroid, score, n_inliers) or None.
    `score` is the inlier ratio (RANSAC inliers / Lowe-good matches), in
    [0, 1]; 1.0 = every good match is geometrically consistent."""
    orb = orb_state['orb']
    kp_t = orb_state['kp']
    des_t = orb_state['des']
    tpl_mask = orb_state['tpl_mask']

    H, W = frame_bgr.shape[:2]
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255
    else:
        roi_mask = None

    kp_f, des_f = orb.detectAndCompute(frame_bgr, mask=roi_mask)
    if des_f is None or len(kp_f) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_t, des_f, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < ratio_test * b.distance:
            good.append(a)
    if len(good) < max(4, min_inliers):
        return None

    pts_t = np.float32([kp_t[mm.queryIdx].pt for mm in good]).reshape(-1, 1, 2)
    pts_f = np.float32([kp_f[mm.trainIdx].pt for mm in good]).reshape(-1, 1, 2)
    try:
        Hm, inl_mask = cv2.findHomography(pts_t, pts_f, cv2.RANSAC,
                                          ransac_reproj)
    except cv2.error:
        return None
    if Hm is None or inl_mask is None:
        return None
    n_inl = int(inl_mask.sum())
    if n_inl < min_inliers:
        return None

    warped = cv2.warpPerspective(
        tpl_mask, Hm, (W, H), flags=cv2.INTER_NEAREST)
    blob = (warped > 0).astype(np.uint8) * 255
    if int((blob > 0).sum()) == 0:
        return None
    ys, xs = np.where(blob > 0)
    cx = float(xs.mean()); cy = float(ys.mean())
    score = float(n_inl) / float(len(good))
    return blob, (cx, cy), score, n_inl


def _largest_blob(
    mask: np.ndarray, min_area: int
) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if stats[best, cv2.CC_STAT_AREA] < min_area:
        return None
    blob = (labels == best).astype(np.uint8) * 255
    cx, cy = centroids[best]
    return blob, (float(cx), float(cy))


def detect_bowl_hsv(
    frame_bgr: np.ndarray,
    hsv_low: Tuple[int, int, int],
    hsv_high: Tuple[int, int, int],
    min_area: int,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(hsv_low, dtype=np.uint8)
    high = np.array(hsv_high, dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    if roi is not None:
        x1, y1, x2, y2 = roi
        keep = np.zeros_like(mask)
        keep[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, keep)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _largest_blob(mask, min_area)


def load_bowl_template(
    frame_path: pathlib.Path,
    mask_path: pathlib.Path,
    pad: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a reference frame and a painted bowl mask, then crop both to the
    tight bbox of the mask (with `pad` pixels of margin). Returns
    (template_bgr, template_mask_uint8) where template_mask is 0/255 of the
    bowl footprint at the template resolution."""
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'cannot read bowl template frame {frame_path}')
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f'cannot read bowl template mask {mask_path}')
    if m.shape[:2] != img.shape[:2]:
        raise RuntimeError(
            f'bowl template mask {m.shape} does not match frame '
            f'{img.shape[:2]}')
    m = (m > 0).astype(np.uint8) * 255
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        raise RuntimeError(f'bowl template mask {mask_path} is empty')
    H, W = img.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(W, int(xs.max()) + 1 + pad)
    y2 = min(H, int(ys.max()) + 1 + pad)
    template_bgr = img[y1:y2, x1:x2].copy()
    template_mask = m[y1:y2, x1:x2].copy()
    return template_bgr, template_mask


def precompute_scaled_templates(
    template_bgr: np.ndarray,
    template_mask: np.ndarray,
    scales: Tuple[float, ...],
) -> list:
    """Pre-resize the bowl template to each scale once, so the per-frame
    matcher just does K masked-correlations. Mask is resampled with NEAREST
    and re-binarised to stay 0/255."""
    out = []
    bh, bw = template_bgr.shape[:2]
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            tpl = template_bgr.copy()
            tplm = template_mask.copy()
        else:
            nh = max(2, int(round(bh * s)))
            nw = max(2, int(round(bw * s)))
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
            tpl = cv2.resize(template_bgr, (nw, nh), interpolation=interp)
            m = cv2.resize(
                template_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            tplm = ((m > 127).astype(np.uint8)) * 255
        out.append((tpl, tplm, float(s)))
    return out


def _expand_to_fit(a1: int, a2: int, need: int, max_: int) -> Tuple[int, int]:
    """Expand the range [a1, a2) to at least `need` pixels wide, clamped to
    [0, max_]. If the available space is too small, returns the maximum
    possible range (caller decides whether to skip)."""
    cur = a2 - a1
    if cur >= need:
        return a1, a2
    grow = need - cur
    new_a1 = max(0, a1 - grow // 2)
    new_a2 = min(max_, new_a1 + need)
    if new_a2 - new_a1 < need:
        new_a1 = max(0, new_a2 - need)
    return new_a1, new_a2


def detect_bowl_template(
    frame_bgr: np.ndarray,
    scaled_templates: list,
    min_score: float,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[np.ndarray, Tuple[float, float], float, float]]:
    """Locate the bowl via masked template matching at multiple scales.

    Uses TM_SQDIFF_NORMED with a binary mask so only painted bowl pixels in
    the template contribute. SQDIFF is preferred over CCORR_NORMED here
    because CCORR_NORMED is a magnitude-normalised inner product and is
    biased toward high-energy patches (e.g. a brightly-coloured target
    plate); SQDIFF measures pixel-wise differences and discriminates
    between similarly-shaped but differently-textured regions.

    Returns (bowl_blob, centroid, match_score, best_scale) or None.
    `match_score` is reported as (1 - sqdiff_normed), clipped to [0, 1],
    so 1.0 is a perfect match — same direction as the old CCORR score."""
    H, W = frame_bgr.shape[:2]
    if roi is None:
        sx1, sy1, sx2, sy2 = 0, 0, W, H
    else:
        sx1, sy1, sx2, sy2 = roi
        sx1 = max(0, sx1); sy1 = max(0, sy1)
        sx2 = min(W, sx2); sy2 = min(H, sy2)

    best: Optional[Tuple[float, int, int, np.ndarray, float]] = None
    for tpl, tplm, scale in scaled_templates:
        th, tw = tpl.shape[:2]
        if tw > W or th > H:
            continue
        rx1, rx2 = _expand_to_fit(sx1, sx2, tw, W)
        ry1, ry2 = _expand_to_fit(sy1, sy2, th, H)
        if (rx2 - rx1) < tw or (ry2 - ry1) < th:
            continue
        sub = frame_bgr[ry1:ry2, rx1:rx2]
        res = cv2.matchTemplate(
            sub, tpl, cv2.TM_SQDIFF_NORMED, mask=tplm)
        if not np.all(np.isfinite(res)):
            res = np.where(
                np.isfinite(res), res, np.float32(1e9)).astype(np.float32)
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        sqd = float(min_val)
        # lower sqdiff = better; convert to a higher-is-better score in [0,1].
        sc = max(0.0, min(1.0, 1.0 - sqd))
        if best is None or sc > best[0]:
            best = (sc, int(min_loc[0]) + rx1, int(min_loc[1]) + ry1, tplm,
                    scale)
    if best is None:
        return None
    score, tlx, tly, tplm, scale = best
    if score < min_score:
        return None
    th, tw = tplm.shape[:2]
    blob = np.zeros((H, W), dtype=np.uint8)
    blob[tly:tly + th, tlx:tlx + tw] = tplm
    ys, xs = np.where(tplm > 0)
    cx = float(xs.mean() + tlx)
    cy = float(ys.mean() + tly)
    return blob, (cx, cy), score, scale


def detect_bowl_bright(
    frame_bgr: np.ndarray,
    thresh: int,
    min_area: int,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
    """Threshold grayscale brightness. Under indoor lighting the bowl rim is
    not actually white, but the bowl's interior has specular highlights from
    metallic/glossy contents that exceed any pixel on the (dimmer) target
    plate — so a brightness gate is the cleanest bowl discriminator here."""
    gr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gr > thresh).astype(np.uint8) * 255
    if roi is not None:
        x1, y1, x2, y2 = roi
        keep = np.zeros_like(mask)
        keep[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, keep)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return _largest_blob(mask, min_area)


def bowl_mask_circle(
    h: int, w: int, centroid: Tuple[float, float], r: int
) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (int(round(centroid[0])), int(round(centroid[1]))),
               r, 255, thickness=-1)
    return m


def stamp_template_at_centroid(
    template_mask: np.ndarray,
    template_local_centroid: Tuple[float, float],
    target_centroid: Tuple[float, float],
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Place `template_mask` into a (frame_h, frame_w) canvas such that the
    mask's centroid lands on `target_centroid`. Used when the detector only
    gives us a centroid (e.g. HSV blue-pattern blob) but we want to score
    IoU using the painted bowl shape rather than the raw detected pixels."""
    out = np.zeros((frame_h, frame_w), dtype=np.uint8)
    th, tw = template_mask.shape[:2]
    tcx, tcy = template_local_centroid
    dcx, dcy = target_centroid
    tlx = int(round(dcx - tcx))
    tly = int(round(dcy - tcy))
    src_x0 = max(0, -tlx); src_y0 = max(0, -tly)
    dst_x0 = max(0, tlx); dst_y0 = max(0, tly)
    w_cp = min(tw - src_x0, frame_w - dst_x0)
    h_cp = min(th - src_y0, frame_h - dst_y0)
    if w_cp <= 0 or h_cp <= 0:
        return out
    out[dst_y0:dst_y0 + h_cp, dst_x0:dst_x0 + w_cp] = \
        template_mask[src_y0:src_y0 + h_cp, src_x0:src_x0 + w_cp]
    return out


def bowl_mask_convex(blob: np.ndarray, dilate_r: int) -> np.ndarray:
    ys, xs = np.where(blob > 0)
    if len(xs) < 3:
        return blob.copy()
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)
    out = np.zeros_like(blob)
    cv2.fillConvexPoly(out, hull, 255)
    if dilate_r > 0:
        k = 2 * dilate_r + 1
        out = cv2.dilate(out, np.ones((k, k), np.uint8))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = (a > 0)
    b_b = (b > 0)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def save_overlay(
    frame_bgr: np.ndarray,
    bowl_mask: np.ndarray,
    target_mask: np.ndarray,
    iou_val: float,
    frame_idx: int,
    out_path: pathlib.Path,
) -> None:
    overlay = frame_bgr.copy()
    red = np.zeros_like(overlay)
    red[..., 2] = 255
    bowl_b = bowl_mask > 0
    overlay[bowl_b] = (0.5 * overlay[bowl_b] + 0.5 * red[bowl_b]).astype(
        np.uint8)
    contours, _ = cv2.findContours(
        target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.putText(
        overlay, f'IoU={iou_val:.3f} f={frame_idx}', (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), overlay)


def score_rollout(
    video_path: pathlib.Path,
    target_xyr: Optional[Tuple[int, int, int]],
    bowl_radius: int,
    detector: str,
    bright_thresh: int,
    hsv_low: Tuple[int, int, int],
    hsv_high: Tuple[int, int, int],
    bowl_mask_mode: str,
    min_bowl_area: int,
    stride: int,
    skip_frames: int,
    auto_target: bool,
    target_search_bbox: Optional[Tuple[int, int, int, int]],
    target_detect_frame: int,
    bowl_roi: Optional[Tuple[int, int, int, int]],
    save_overlay_path: Optional[pathlib.Path],
    target_mask_path: Optional[pathlib.Path] = None,
    bowl_template_frame_path: Optional[pathlib.Path] = None,
    bowl_template_mask_path: Optional[pathlib.Path] = None,
    template_min_score: float = 0.0,
    template_scales: Tuple[float, ...] = (1.0,),
    orb_min_inliers: int = 10,
    orb_nfeatures: int = 2000,
    show_progress: bool = True,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'failed to open {video_path}')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scaled_templates: list = []
    orb_state: Optional[dict] = None
    stamp_mask: Optional[np.ndarray] = None
    stamp_local_centroid: Optional[Tuple[float, float]] = None
    if (bowl_mask_mode == 'template'
            and detector in ('hsv', 'bright')):
        if bowl_template_frame_path is None or bowl_template_mask_path is None:
            raise RuntimeError(
                '--bowl-mask template (with --detector hsv/bright) requires '
                '--bowl-template-frame and --bowl-template-mask: the painted '
                'shape is stamped at the detected centroid each frame')
        _tpl_bgr, _tpl_mask = load_bowl_template(
            bowl_template_frame_path, bowl_template_mask_path)
        ys_t, xs_t = np.where(_tpl_mask > 0)
        if len(xs_t) == 0:
            raise RuntimeError('bowl template mask is empty')
        stamp_mask = _tpl_mask
        stamp_local_centroid = (float(xs_t.mean()), float(ys_t.mean()))
    if detector == 'template':
        if bowl_template_frame_path is None or bowl_template_mask_path is None:
            raise RuntimeError(
                '--detector template requires --bowl-template-frame and '
                '--bowl-template-mask')
        tpl_bgr, tpl_mask = load_bowl_template(
            bowl_template_frame_path, bowl_template_mask_path)
        scaled_templates = precompute_scaled_templates(
            tpl_bgr, tpl_mask, template_scales)
    elif detector == 'orb':
        if bowl_template_frame_path is None or bowl_template_mask_path is None:
            raise RuntimeError(
                '--detector orb requires --bowl-template-frame and '
                '--bowl-template-mask')
        tpl_bgr, tpl_mask = load_bowl_template(
            bowl_template_frame_path, bowl_template_mask_path)
        orb_state = build_orb_state(
            tpl_bgr, tpl_mask, nfeatures=orb_nfeatures)
        if orb_state is None:
            raise RuntimeError(
                f'too few ORB features in bowl template '
                f'{bowl_template_frame_path} under mask '
                f'{bowl_template_mask_path}; the bowl may be too small or '
                f'too featureless for feature matching')

    if target_mask_path is not None:
        loaded = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            raise RuntimeError(f'cannot read target mask {target_mask_path}')
        if loaded.shape != (h, w):
            raise RuntimeError(
                f'target mask shape {loaded.shape} does not match video '
                f'{(h, w)} for {video_path}')
        target_mask = (loaded > 0).astype(np.uint8) * 255
        ys, xs = np.where(target_mask > 0)
        if len(xs) == 0:
            raise RuntimeError(f'target mask {target_mask_path} is empty')
        cx_t = int(round(float(xs.mean())))
        cy_t = int(round(float(ys.mean())))
        r_t = int(round(float(np.sqrt(len(xs) / np.pi))))
    elif auto_target:
        if target_search_bbox is None:
            raise RuntimeError(
                'auto_target=True requires --target-search-bbox')
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_detect_frame)
        ok, ref = cap.read()
        if not ok:
            raise RuntimeError(
                f'cannot read target-detect frame {target_detect_frame} '
                f'from {video_path}')
        det = auto_detect_target(ref, target_search_bbox)
        if det is None:
            raise RuntimeError(
                f'auto target detection failed in {video_path} at frame '
                f'{target_detect_frame}')
        cx_t, cy_t, r_t = det
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        target_mask = build_target_mask(h, w, cx_t, cy_t, r_t)
    else:
        if target_xyr is None:
            raise RuntimeError('--target required when --auto-target not set')
        cx_t, cy_t, r_t = target_xyr
        target_mask = build_target_mask(h, w, cx_t, cy_t, r_t)

    max_iou = 0.0
    max_iou_frame = -1
    max_iou_centroid: Tuple[float, float] = (float('nan'), float('nan'))
    best_overlay: Optional[Tuple[np.ndarray, np.ndarray]] = None
    detected = 0
    seen = 0
    frame_idx = 0

    pbar = tqdm(total=n_total, desc=video_path.parent.name, leave=False,
                disable=not show_progress)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pbar.update(1)
        if frame_idx < skip_frames or frame_idx % stride != 0:
            frame_idx += 1
            continue
        seen += 1

        if detector == 'bright':
            det = detect_bowl_bright(
                frame, bright_thresh, min_bowl_area, bowl_roi)
            if det is None:
                frame_idx += 1
                continue
            blob, centroid = det
        elif detector == 'hsv':
            det = detect_bowl_hsv(
                frame, hsv_low, hsv_high, min_bowl_area, bowl_roi)
            if det is None:
                frame_idx += 1
                continue
            blob, centroid = det
        elif detector == 'template':
            det_t = detect_bowl_template(
                frame, scaled_templates, template_min_score, bowl_roi)
            if det_t is None:
                frame_idx += 1
                continue
            blob, centroid, _, _ = det_t
        elif detector == 'orb':
            det_o = detect_bowl_orb(
                frame, orb_state, orb_min_inliers, roi=bowl_roi)
            if det_o is None:
                frame_idx += 1
                continue
            blob, centroid, _, _ = det_o
        else:
            raise ValueError(detector)
        detected += 1

        if bowl_mask_mode == 'circle':
            bm = bowl_mask_circle(h, w, centroid, bowl_radius)
        elif bowl_mask_mode == 'convex':
            bm = bowl_mask_convex(blob, dilate_r=max(0, bowl_radius // 4))
        elif bowl_mask_mode == 'template':
            if detector in ('template', 'orb'):
                # `blob` is the warped painted shape produced by the
                # feature/template matcher.
                bm = blob
            else:
                # HSV/bright detectors return a blob of detected pixels (e.g.
                # the bowl's blue pattern) that is smaller than the bowl
                # footprint. Stamp the painted shape at the detected centroid.
                bm = stamp_template_at_centroid(
                    stamp_mask, stamp_local_centroid, centroid, h, w)
        else:
            raise ValueError(bowl_mask_mode)

        score = iou(bm, target_mask)
        if score > max_iou:
            max_iou = score
            max_iou_frame = frame_idx
            max_iou_centroid = centroid
            if save_overlay_path is not None:
                best_overlay = (frame.copy(), bm.copy())

        frame_idx += 1
    pbar.close()
    cap.release()

    if save_overlay_path is not None and best_overlay is not None:
        save_overlay(
            best_overlay[0], best_overlay[1], target_mask,
            max_iou, max_iou_frame, save_overlay_path)

    return dict(
        num_frames=n_total,
        frames_seen=seen,
        frames_with_bowl_detected=detected,
        max_iou=max_iou,
        max_iou_frame=max_iou_frame,
        max_iou_cx=max_iou_centroid[0],
        max_iou_cy=max_iou_centroid[1],
        target_cx=cx_t,
        target_cy=cy_t,
        target_r=r_t,
    )


def method_of(name: str) -> str:
    return 'VLM' if name.endswith('_VLM') else 'baseline'


def _report_row(row: dict) -> None:
    if '_error' in row:
        tqdm.write(f'[err] {row["rollout"]}: {row["_error"]}')
        return
    tqdm.write(
        f'{row["rollout"]}  method={row["method"]}  '
        f'max_iou={row["max_iou"]:.3f}  '
        f'frame={row["max_iou_frame"]}  '
        f'det={row["frames_with_bowl_detected"]}/{row["frames_seen"]}')


def _worker(task: dict) -> dict:
    """Process-pool entry point. Returns either a stats row or an error row.
    Limit OpenCV/BLAS internal threads so per-worker CPU stays bounded."""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    name = task['name']
    try:
        stats = score_rollout(
            task['video_path'],
            target_xyr=task['target_xyr'],
            bowl_radius=task['bowl_radius'],
            detector=task['detector'],
            bright_thresh=task['bright_thresh'],
            hsv_low=task['hsv_low'],
            hsv_high=task['hsv_high'],
            bowl_mask_mode=task['bowl_mask_mode'],
            min_bowl_area=task['min_bowl_area'],
            stride=task['stride'],
            skip_frames=task['skip_frames'],
            auto_target=task['auto_target'],
            target_search_bbox=task['target_search_bbox'],
            target_detect_frame=task['target_detect_frame'],
            bowl_roi=task['bowl_roi'],
            save_overlay_path=task['save_overlay_path'],
            target_mask_path=task.get('target_mask_path'),
            bowl_template_frame_path=task.get('bowl_template_frame_path'),
            bowl_template_mask_path=task.get('bowl_template_mask_path'),
            template_min_score=task.get('template_min_score', 0.0),
            template_scales=task.get('template_scales', (1.0,)),
            orb_min_inliers=task.get('orb_min_inliers', 10),
            orb_nfeatures=task.get('orb_nfeatures', 2000),
            show_progress=False,
        )
        return dict(rollout=name, method=method_of(name), **stats)
    except Exception as e:
        return dict(rollout=name, method=method_of(name), _error=repr(e))


def summarise(rows, key='max_iou'):
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        bucket[r['method']].append(r[key])
    print(f'\n=== per-method summary on {key} ===')
    print(f'{"method":<10} {"n":>4} {"mean":>7} {"median":>7} '
          f'{"std":>7} {"min":>7} {"max":>7}')
    for m, vals in sorted(bucket.items()):
        a = np.array(vals, dtype=np.float64)
        print(f'{m:<10} {len(a):>4d} {a.mean():>7.3f} '
              f'{np.median(a):>7.3f} {a.std():>7.3f} '
              f'{a.min():>7.3f} {a.max():>7.3f}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, type=pathlib.Path,
                    help='checkpoint epoch dir containing rollout subdirs')
    ap.add_argument('--video', choices=('image1', 'image2'), default='image1',
                    help='which mp4 to read; image1 = fixed scene cam, '
                         'image2 = wrist cam (camera moves, not suitable for '
                         'fixed-target IoU)')
    ap.add_argument('--target', type=parse_target, default=None,
                    metavar='xc,yc,r',
                    help='manual target circle (skip with --auto-target)')
    ap.add_argument('--target-mask', type=pathlib.Path, default=None,
                    help='path to a binary PNG mask (uint8, >0 = target) at '
                         'the same resolution as the video; overrides '
                         '--target and --auto-target')
    ap.add_argument('--auto-target', action='store_true',
                    help='auto-detect the white target plate per rollout in a '
                         'reference frame (recommended; camera position '
                         'varies slightly between rollouts)')
    ap.add_argument('--target-search-bbox', type=str, default=None,
                    metavar='x1,y1,x2,y2',
                    help='where to search for the target plate in the '
                         'reference frame (required with --auto-target)')
    ap.add_argument('--target-detect-frame', type=int, default=120,
                    help='frame index to detect target from; should be after '
                         'camera has stabilised (default 120)')
    ap.add_argument('--skip-frames', type=int, default=100,
                    help='skip first N frames (camera-stabilisation transient '
                         'in image1)')
    ap.add_argument('--bowl-radius', type=int, default=0,
                    help='bowl footprint radius in pixels (only used when '
                         '--bowl-mask circle); ignored for --bowl-mask template')
    ap.add_argument('--detector',
                    choices=('bright', 'hsv', 'template', 'orb'),
                    default='hsv',
                    help='bowl cue: hsv=saturation gate on bowl-content '
                         'hue; bright=specular gray>thr; '
                         'template=masked template match against a painted '
                         'bowl reference; orb=ORB feature matching against '
                         'a painted bowl reference (RECOMMENDED for textured '
                         'bowls — robust to brightness/scale changes and '
                         'rejects uniform-colour distractors like a plate)')
    ap.add_argument('--bowl-template-frame', type=pathlib.Path, default=None,
                    help='reference frame the bowl was painted on '
                         '(required with --detector template)')
    ap.add_argument('--bowl-template-mask', type=pathlib.Path, default=None,
                    help='painted bowl mask at the reference-frame resolution '
                         '(required with --detector template); produce one '
                         'with paint_target_mask.py')
    ap.add_argument('--template-min-score', type=float, default=0.0,
                    help='reject template matches below this normalised '
                         'similarity score, where score = 1 - sqdiff_normed '
                         'in [0,1] (1.0 = perfect match). Useful to drop '
                         'frames where the bowl is occluded.')
    ap.add_argument('--template-scales', type=parse_scales, default=(1.0,),
                    help='scales at which to match the bowl template. '
                         'Format: comma list "0.7,0.85,1.0,1.15,1.3" or '
                         'linspace range "min:max:n" e.g. "0.7:1.3:7". '
                         'Default 1.0 (single scale). Use a range when the '
                         'bowl visibly grows/shrinks across frames. '
                         'Ignored when --detector orb.')
    ap.add_argument('--orb-min-inliers', type=int, default=10,
                    help='minimum RANSAC inlier matches needed to accept an '
                         'ORB detection; raise for stricter detection, '
                         'lower if the bowl is small or partly occluded')
    ap.add_argument('--orb-nfeatures', type=int, default=2000,
                    help='max ORB features per image (template and frame)')
    ap.add_argument('--bright-thresh', type=int, default=160,
                    help='grayscale threshold for --detector bright')
    ap.add_argument('--bowl-roi', type=str, default=None,
                    metavar='x1,y1,x2,y2',
                    help='optional ROI to restrict bowl search (excludes '
                         'stray specular highlights e.g. on the camera lens)')
    ap.add_argument('--hsv-low', type=parse_hsv, default=(80, 60, 40),
                    metavar='H,S,V')
    ap.add_argument('--hsv-high', type=parse_hsv, default=(130, 255, 255),
                    metavar='H,S,V')
    ap.add_argument('--bowl-mask', choices=('circle', 'convex', 'template'),
                    default='circle',
                    help='shape of the bowl footprint used for IoU. '
                         "'template' uses the painted bowl shape translated "
                         'to the matched location (requires --detector '
                         'template).')
    ap.add_argument('--min-bowl-area', type=int, default=5)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--save-overlay', action='store_true')
    ap.add_argument('--out-csv', type=pathlib.Path, default=None)
    ap.add_argument('--workers', type=int,
                    default=min(8, (os.cpu_count() or 2)),
                    help='parallel rollout workers (default min(8, cpu_count))')
    args = ap.parse_args()

    root: pathlib.Path = args.root
    if not root.is_dir():
        print(f'--root not a directory: {root}', file=sys.stderr)
        return 2

    video_name = f'rollout_{args.video}.mp4'
    out_csv = args.out_csv or (root / 'iou_scores.csv')

    bowl_roi: Optional[Tuple[int, int, int, int]] = None
    if args.bowl_roi:
        parts = [int(p) for p in args.bowl_roi.split(',')]
        if len(parts) != 4:
            print(f'--bowl-roi expects 4 ints, got {args.bowl_roi!r}',
                  file=sys.stderr)
            return 2
        bowl_roi = (parts[0], parts[1], parts[2], parts[3])

    target_search_bbox: Optional[Tuple[int, int, int, int]] = None
    if args.target_search_bbox:
        parts = [int(p) for p in args.target_search_bbox.split(',')]
        if len(parts) != 4:
            print(f'--target-search-bbox expects 4 ints, got '
                  f'{args.target_search_bbox!r}', file=sys.stderr)
            return 2
        target_search_bbox = (parts[0], parts[1], parts[2], parts[3])

    needs_template = (args.detector in ('template', 'orb')
                       or args.bowl_mask == 'template')
    if needs_template:
        if args.bowl_template_frame is None or args.bowl_template_mask is None:
            print('--bowl-template-frame and --bowl-template-mask are '
                  'required (used by --detector template/orb, and by '
                  '--bowl-mask template to stamp the painted shape at the '
                  'detected centroid)', file=sys.stderr)
            return 2
        if not args.bowl_template_frame.exists():
            print(f'--bowl-template-frame not found: '
                  f'{args.bowl_template_frame}', file=sys.stderr)
            return 2
        if not args.bowl_template_mask.exists():
            print(f'--bowl-template-mask not found: '
                  f'{args.bowl_template_mask}', file=sys.stderr)
            return 2
    if args.bowl_mask == 'circle' and args.bowl_radius <= 0:
        print('--bowl-radius is required when --bowl-mask circle',
              file=sys.stderr)
        return 2

    if args.target_mask is not None:
        if not args.target_mask.exists():
            print(f'--target-mask not found: {args.target_mask}',
                  file=sys.stderr)
            return 2
    elif not args.auto_target and args.target is None:
        print('one of --target, --target-mask, or --auto-target is required',
              file=sys.stderr)
        return 2
    elif args.auto_target and target_search_bbox is None:
        print('--target-search-bbox is required with --auto-target',
              file=sys.stderr)
        return 2

    rollout_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not rollout_dirs:
        print(f'no subdirs in {root}', file=sys.stderr)
        return 2

    tasks = []
    for d in rollout_dirs:
        vp = d / video_name
        if not vp.exists():
            print(f'[skip] {d.name}: missing {video_name}')
            continue
        tasks.append(dict(
            name=d.name,
            video_path=vp,
            target_xyr=args.target,
            bowl_radius=args.bowl_radius,
            detector=args.detector,
            bright_thresh=args.bright_thresh,
            hsv_low=args.hsv_low,
            hsv_high=args.hsv_high,
            bowl_mask_mode=args.bowl_mask,
            min_bowl_area=args.min_bowl_area,
            stride=args.stride,
            skip_frames=args.skip_frames,
            auto_target=args.auto_target,
            target_search_bbox=target_search_bbox,
            target_detect_frame=args.target_detect_frame,
            bowl_roi=bowl_roi,
            save_overlay_path=(d / 'iou_max_frame.png')
                              if args.save_overlay else None,
            target_mask_path=args.target_mask,
            bowl_template_frame_path=args.bowl_template_frame,
            bowl_template_mask_path=args.bowl_template_mask,
            template_min_score=args.template_min_score,
            template_scales=args.template_scales,
            orb_min_inliers=args.orb_min_inliers,
            orb_nfeatures=args.orb_nfeatures,
        ))

    if not tasks:
        print(f'no rollouts with {video_name} under {root}', file=sys.stderr)
        return 2

    rows = []
    workers = max(1, args.workers)
    print(f'scoring {len(tasks)} rollouts with {workers} worker(s)')
    if workers == 1:
        # In-process path: keep the per-rollout frame progress bars visible.
        for task in tqdm(tasks, desc='rollouts'):
            row = _worker(task)
            _report_row(row)
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs),
                          desc='rollouts'):
                row = f.result()
                _report_row(row)
                rows.append(row)

    rows = [r for r in rows if '_error' not in r]
    rows.sort(key=lambda r: r['rollout'])

    if not rows:
        print('no rollouts scored', file=sys.stderr)
        return 1

    fieldnames = ['rollout', 'method', 'num_frames', 'frames_seen',
                  'frames_with_bowl_detected', 'max_iou', 'max_iou_frame',
                  'max_iou_cx', 'max_iou_cy',
                  'target_cx', 'target_cy', 'target_r']
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f'\nwrote {out_csv} ({len(rows)} rows)')

    summarise(rows, key='max_iou')
    return 0


if __name__ == '__main__':
    sys.exit(main())
