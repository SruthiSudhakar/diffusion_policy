"""
Compute MSE / PSNR / SSIM between the winning generated video's last frame
and the ground-truth observation at each timestep of a VLM-guided rollout,
with proper FOV alignment recovered from the video-model's preprocessing.

Accepts either:
  * a single rollout dir (one containing videogen/ and observations/), or
  * a parent dir, in which case every descendant dir whose name ends in
    "VLM" and that contains a videogen/ subdir is processed. An aggregate
    summary is written to <parent>/mse_results_aggregate.json.

For a rollout dir laid out as:
  <rollout_dir>/
    videogen/
      <prefix>_NNNNNN.jpg     # conditioning image fed to the video model (native res)
      <prefix>_NNNNNN/
        0.mp4 ... 9.mp4
        ranking.json          # has "winner_idx"
      ...
    observations/
      frame_000000.jpg        # initial state (pre-execution)
      frame_000001.jpg        # state after videogen step 000000 executed
      ...

Mapping: videogen step N -> observation frame_{N+1:06d}.jpg.

Why alignment matters
---------------------
HunyuanVideo's `load_video` in the server (generate.py) does, for each
sample, in order:

  1. scale = max((video_width + margin) / W_in, (video_height + margin) / H_in)
  2. bilinear resize to (W_in * scale, H_in * scale)
  3. RANDOM crop to (video_width, video_height)

The random crop offset is seeded per sample, so different candidate videos
within the same step have different crops. To compare a video frame to the
real-world observation, we have to apply the same resize+crop to the obs.

We recover the crop offset by template-matching the winner video's first
frame against the resized conditioning image (which is what the server
loaded). Match scores are typically >0.99 in practice. We then apply that
same crop to the post-execution observation and compute MSE against the
winner video's last frame -- both at native 848x480 (or whatever
--video_width / --video_height is set to).

Outputs in <rollout_dir>:
  mse_results.json        per-step MSE + crop offsets + mean/std
  mse_vis/step_NNNNNN.png native-res [pred | aligned GT | abs diff] panel

Usage:
  python compute_rollout_winner_mse.py <rollout_dir> [--no-vis]
       [--video-width 848] [--video-height 480] [--margin 40]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


STEP_DIR_RE = re.compile(r"(.*)_(\d{6})$")


def list_step_dirs(videogen_dir: Path):
    out = []
    for p in sorted(videogen_dir.iterdir()):
        if not p.is_dir():
            continue
        m = STEP_DIR_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(2)), m.group(1), p))
    out.sort(key=lambda x: x[0])
    return out


def read_video_frames(video_path: Path):
    """Return (first_rgb, last_rgb) from an mp4."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        ok, first = cap.read()
        if not ok or first is None:
            raise RuntimeError(f"could not read first frame from {video_path}")
        nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last = first
        if nb > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, nb - 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                last = frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    last = frame
        return (cv2.cvtColor(first, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(last,  cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def server_resize_dims(W, H, video_w, video_h, margin):
    """Replicates generate.py's resize_dims: pre-crop dims after max-scale resize."""
    scale = max((video_w + margin) / W, (video_h + margin) / H)
    return (int(round(W * scale)), int(round(H * scale)))  # (w, h)


def server_resize(img_rgb, video_w, video_h, margin):
    new_w, new_h = server_resize_dims(img_rgb.shape[1], img_rgb.shape[0],
                                      video_w, video_h, margin)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def find_crop_offset(resized_cond_rgb, video_first_rgb):
    """Template-match the video's first frame inside resized_cond. Returns
    ((x_off, y_off), score)."""
    cg = cv2.cvtColor(resized_cond_rgb, cv2.COLOR_RGB2GRAY)
    vg = cv2.cvtColor(video_first_rgb,  cv2.COLOR_RGB2GRAY)
    res = cv2.matchTemplate(cg, vg, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_loc, float(max_val)


# ---------- metrics (operate on float32 RGB images in [0, 1]) ----------

_SSIM_KSIZE = 11
_SSIM_SIGMA = 1.5
_SSIM_C1 = (0.01) ** 2  # K1=0.01, L=1.0
_SSIM_C2 = (0.03) ** 2  # K2=0.03, L=1.0


def _gauss(x):
    return cv2.GaussianBlur(x, (_SSIM_KSIZE, _SSIM_KSIZE), _SSIM_SIGMA,
                            borderType=cv2.BORDER_REPLICATE)


def ssim_per_channel(a, b):
    """Wang-et-al. SSIM with an 11x11 Gaussian window (sigma=1.5). Expects
    float32 inputs in [0, 1]. Returns per-channel mean SSIM and the RGB-avg."""
    per = []
    for c in range(a.shape[-1]):
        x = a[..., c]
        y = b[..., c]
        mu_x = _gauss(x)
        mu_y = _gauss(y)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sig_x2 = _gauss(x * x) - mu_x2
        sig_y2 = _gauss(y * y) - mu_y2
        sig_xy = _gauss(x * y) - mu_xy
        num = (2 * mu_xy + _SSIM_C1) * (2 * sig_xy + _SSIM_C2)
        den = (mu_x2 + mu_y2 + _SSIM_C1) * (sig_x2 + sig_y2 + _SSIM_C2)
        # Crop the boundary region the Gaussian "saw" outside the image to match
        # scikit-image's default (gaussian_weights=True) behavior.
        pad = _SSIM_KSIZE // 2
        per.append(float(num[pad:-pad, pad:-pad].mean()
                         / den[pad:-pad, pad:-pad].mean()))
    return per, float(np.mean(per))


def compute_metrics(pred_uint8, gt_uint8):
    """pred, gt: HxWx3 uint8. Returns dict with MSE/PSNR/SSIM on [0,1] floats."""
    p = pred_uint8.astype(np.float32) / 255.0
    g = gt_uint8.astype(np.float32) / 255.0
    diff = p - g
    mse = float((diff ** 2).mean())
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    ssim_rgb, ssim_mean = ssim_per_channel(p, g)
    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim_mean,
        "ssim_per_channel_rgb": ssim_rgb,
    }


def save_side_by_side(pred, gt, diff, step_idx, winner_idx, metrics,
                      crop_xy, match_score, out_path):
    h, w = pred.shape[:2]
    pad = 16
    label_h = 28
    total_w = w * 3 + pad * 4
    total_h = h + pad * 2 + label_h * 2
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    for i, panel in enumerate([pred, gt, diff]):
        canvas.paste(Image.fromarray(panel.astype(np.uint8)),
                     (pad + i * (w + pad), pad))
    draw = ImageDraw.Draw(canvas)
    labels = [
        f"winner pred last frame (idx={winner_idx})",
        "aligned GT obs",
        "abs diff",
    ]
    for i, txt in enumerate(labels):
        draw.text((pad + i * (w + pad), pad + h + 4), txt, fill=(0, 0, 0))
    draw.text((pad, total_h - label_h * 2 + 4),
              f"step {step_idx:06d}  MSE={metrics['mse']:.5f}  "
              f"PSNR={metrics['psnr_db']:.2f}dB  SSIM={metrics['ssim']:.4f}  "
              f"crop=({crop_xy[0]},{crop_xy[1]})  match={match_score:.4f}",
              fill=(0, 0, 0))
    canvas.save(out_path)


def _stat(xs):
    return ({"mean": float(np.mean(xs)), "std": float(np.std(xs))}
            if xs else {"mean": None, "std": None})


def _frames_dir_has_jpgs(d: Path) -> bool:
    return d.is_dir() and any(d.glob("frame_*.jpg"))


def pick_obs_dir(rollout_dir: Path, cond_rgb_full: np.ndarray) -> Path:
    """Locate the observations dir holding frame_NNNNNN.jpg.

    Supports two layouts:
      1. <rollout>/observations/frame_NNNNNN.jpg          (single camera)
      2. <rollout>/observations/<camera>/frame_NNNNNN.jpg (multi-camera)

    For layout 2, picks the camera whose frame_000000.jpg has the lowest MSE
    against the first conditioning image (the conditioning is what was fed to
    the video model, so the camera that matches it is the right one). Raises
    FileNotFoundError if no candidate is found.
    """
    obs_root = rollout_dir / "observations"
    if not obs_root.is_dir():
        raise FileNotFoundError(f"{obs_root} not found")

    if _frames_dir_has_jpgs(obs_root):
        return obs_root

    # Multi-camera layout: choose the subdir whose frame_000000.jpg best matches cond.
    candidates = [d for d in sorted(obs_root.iterdir())
                  if _frames_dir_has_jpgs(d)]
    if not candidates:
        raise FileNotFoundError(
            f"no frame_*.jpg found directly under {obs_root} or its subdirs")
    if len(candidates) == 1:
        return candidates[0]

    best, best_mse = None, float("inf")
    for d in candidates:
        f0 = d / "frame_000000.jpg"
        if not f0.is_file():
            continue
        cam = np.array(Image.open(f0).convert("RGB"))
        # Resize cond to camera size for an apples-to-apples MSE.
        cond_r = cv2.resize(cond_rgb_full, (cam.shape[1], cam.shape[0]),
                            interpolation=cv2.INTER_AREA)
        m = float(((cond_r.astype(np.float32) - cam.astype(np.float32)) ** 2).mean())
        if m < best_mse:
            best, best_mse = d, m
    if best is None:
        raise FileNotFoundError(
            f"no camera under {obs_root} has frame_000000.jpg")
    print(f"  obs camera: {best.name} (cond->cam MSE={best_mse:.1f} "
          f"vs {len(candidates)} candidate(s))")
    return best


def process_rollout(rollout_dir: Path, args) -> dict:
    """Run per-step MSE/PSNR/SSIM for one rollout dir. Writes mse_results.json
    (and optional mse_vis/) into the rollout dir. Returns the results dict.

    Raises FileNotFoundError if required subdirs are missing so callers can
    record the error and continue when iterating many rollouts.
    """
    videogen_dir = rollout_dir / "videogen"
    if not videogen_dir.is_dir():
        raise FileNotFoundError(f"{videogen_dir} not found")

    step_dirs = list_step_dirs(videogen_dir)
    if not step_dirs:
        raise FileNotFoundError(f"no step dirs matching *_NNNNNN under {videogen_dir}")

    # Pick the observation dir using the first step's conditioning image as the
    # reference (it's what the videogen pipeline saw).
    first_idx, first_prefix, _ = step_dirs[0]
    first_cond = videogen_dir / f"{first_prefix}_{first_idx:06d}.jpg"
    if not first_cond.is_file():
        raise FileNotFoundError(
            f"first-step conditioning image {first_cond.name} not found")
    obs_dir = pick_obs_dir(rollout_dir,
                           np.array(Image.open(first_cond).convert("RGB")))

    vis_dir = rollout_dir / "mse_vis"
    if not args.no_vis:
        vis_dir.mkdir(exist_ok=True)

    per_step = []
    skipped = []
    mses, psnrs, ssims = [], [], []
    b_mses, b_psnrs, b_ssims = [], [], []
    rel_mses, psnr_gains, ssim_gains = [], [], []

    for step_idx, prefix, step_dir in step_dirs:
        ranking_path = step_dir / "ranking.json"
        if not ranking_path.is_file():
            skipped.append({"step_idx": step_idx, "reason": "ranking.json missing"})
            print(f"[step {step_idx:06d}] SKIP (no ranking.json)")
            continue
        with open(ranking_path) as f:
            ranking = json.load(f)
        winner_idx = int(ranking["winner_idx"])

        video_path = step_dir / f"{winner_idx}.mp4"
        if not video_path.is_file():
            skipped.append({"step_idx": step_idx,
                            "reason": f"winner video {video_path.name} missing"})
            print(f"[step {step_idx:06d}] SKIP (winner mp4 missing)")
            continue

        cond_path = videogen_dir / f"{prefix}_{step_idx:06d}.jpg"
        if not cond_path.is_file():
            skipped.append({"step_idx": step_idx,
                            "reason": f"conditioning image {cond_path.name} missing"})
            print(f"[step {step_idx:06d}] SKIP (no conditioning jpg)")
            continue

        gt_path = obs_dir / f"frame_{step_idx + 1:06d}.jpg"
        if not gt_path.is_file():
            skipped.append({"step_idx": step_idx,
                            "reason": f"no matching ground-truth frame {gt_path.name}"})
            print(f"[step {step_idx:06d}] SKIP (no gt frame: {gt_path.name})")
            continue

        cond_rgb = np.array(Image.open(cond_path).convert("RGB"))
        rcond = server_resize(cond_rgb, args.video_width, args.video_height, args.margin)

        first_rgb, last_rgb = read_video_frames(video_path)
        if last_rgb.shape[:2] != (args.video_height, args.video_width):
            skipped.append({
                "step_idx": step_idx,
                "reason": (f"video shape {last_rgb.shape[:2]} != "
                           f"({args.video_height}, {args.video_width})"),
            })
            print(f"[step {step_idx:06d}] SKIP (unexpected video size {last_rgb.shape[:2]})")
            continue

        (x_off, y_off), score = find_crop_offset(rcond, first_rgb)
        if score < args.match_warn:
            print(f"[step {step_idx:06d}] WARN low template-match score {score:.3f}")

        gt_rgb = np.array(Image.open(gt_path).convert("RGB"))
        rgt = server_resize(gt_rgb, args.video_width, args.video_height, args.margin)
        if rgt.shape[:2] != rcond.shape[:2]:
            print(f"[step {step_idx:06d}] WARN resized obs shape {rgt.shape[:2]} "
                  f"!= resized cond {rcond.shape[:2]}")
        gt_crop = rgt[y_off:y_off + args.video_height,
                      x_off:x_off + args.video_width]

        metrics = compute_metrics(last_rgb, gt_crop)
        # "Do nothing" baseline: video first frame vs aligned GT. This is the
        # error you'd get if the model just emitted the conditioning image as
        # both the start and the end of the video (no predicted motion).
        baseline = compute_metrics(first_rgb, gt_crop)
        if baseline["mse"] > 0:
            rel_mse = metrics["mse"] / baseline["mse"]
        else:
            rel_mse = 0.0 if metrics["mse"] == 0 else float("inf")
        psnr_gain_db = metrics["psnr_db"] - baseline["psnr_db"]
        ssim_gain = metrics["ssim"] - baseline["ssim"]

        diff_uint8 = np.clip(
            np.abs(last_rgb.astype(np.int16) - gt_crop.astype(np.int16)),
            0, 255).astype(np.uint8)

        per_step.append({
            "step_idx": step_idx,
            "winner_idx": winner_idx,
            "video_path": str(video_path),
            "cond_path": str(cond_path),
            "gt_path": str(gt_path),
            "crop_offset_xy": [int(x_off), int(y_off)],
            "template_match_score": score,
            **metrics,
            "baseline_mse":     baseline["mse"],
            "baseline_psnr_db": baseline["psnr_db"],
            "baseline_ssim":    baseline["ssim"],
            "relative_mse":     rel_mse,
            "psnr_gain_db":     psnr_gain_db,
            "ssim_gain":        ssim_gain,
        })
        mses.append(metrics["mse"])
        psnrs.append(metrics["psnr_db"])
        ssims.append(metrics["ssim"])
        b_mses.append(baseline["mse"])
        b_psnrs.append(baseline["psnr_db"])
        b_ssims.append(baseline["ssim"])
        rel_mses.append(rel_mse)
        psnr_gains.append(psnr_gain_db)
        ssim_gains.append(ssim_gain)
        print(f"[step {step_idx:06d}] winner={winner_idx}  "
              f"crop=({x_off:3d},{y_off:3d})  match={score:.3f}  "
              f"MSE={metrics['mse']:.5f}  PSNR={metrics['psnr_db']:5.2f}dB  "
              f"SSIM={metrics['ssim']:.4f}  "
              f"rel_MSE={rel_mse:5.3f}  ΔPSNR={psnr_gain_db:+5.2f}dB")

        if not args.no_vis:
            save_side_by_side(
                last_rgb, gt_crop, diff_uint8,
                step_idx, winner_idx, metrics,
                (x_off, y_off), score,
                vis_dir / f"step_{step_idx:06d}.png",
            )

    stats = {
        "mse":           _stat(mses),
        "psnr":          _stat(psnrs),
        "ssim":          _stat(ssims),
        "baseline_mse":  _stat(b_mses),
        "baseline_psnr": _stat(b_psnrs),
        "baseline_ssim": _stat(b_ssims),
        "relative_mse":  _stat(rel_mses),
        "psnr_gain_db":  _stat(psnr_gains),
        "ssim_gain":     _stat(ssim_gains),
    }

    if mses:
        print(f"  evaluated steps: {len(mses)}  "
              f"MSE={stats['mse']['mean']:.5f}  "
              f"PSNR={stats['psnr']['mean']:.2f}dB  "
              f"SSIM={stats['ssim']['mean']:.4f}  "
              f"rel_MSE={stats['relative_mse']['mean']:.3f}  "
              f"ΔPSNR={stats['psnr_gain_db']['mean']:+.2f}dB")
    else:
        print("  no steps were evaluated")
    if skipped:
        print(f"  skipped: {len(skipped)} step(s)")

    results = {
        "rollout_dir": str(rollout_dir),
        "obs_dir": str(obs_dir),
        "video_width": args.video_width,
        "video_height": args.video_height,
        "margin": args.margin,
        "value_range": "metrics computed on float32 RGB in [0, 1]",
        "ssim_window": {"kernel": _SSIM_KSIZE, "sigma": _SSIM_SIGMA,
                        "K1": 0.01, "K2": 0.03, "data_range": 1.0,
                        "per_channel_then_avg": True},
        "baseline_explanation": ("baseline = MSE/PSNR/SSIM between the video's "
                                 "FIRST frame and the aligned GT post-exec obs "
                                 "(model emits no predicted motion)"),
        "per_step": per_step,
        "num_steps_evaluated": len(mses),
        "skipped": skipped,
        **stats,
    }
    out_json = rollout_dir / "mse_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    return results


# Outcome-classification table for the rollout-name suffixes used across
# pnp_lego ({ss,sf,fs,ff} with optional "poorcritic_" prefix) and
# push_bowl / stacking (digit-prefixed {s,f,ish,sisg}, e.g. "4s", "3ish").
_OUTCOME = {
    "ss": "success", "s": "success",
    "ff": "failure", "f": "failure",
    "sf": "partial", "fs": "partial",
    "ish": "partial", "sisg": "partial",
}


def classify_rollout(name: str) -> str:
    """Map a rollout dir name to one of {success, failure, partial, unknown}.

    Scans underscore-separated tokens, strips a leading digit run from each,
    and returns the first match in _OUTCOME. Examples:
      18_ss_2026-05-10_22-46-11_VLM    -> success
      18_ff_...                        -> failure
      0_poorcritic_sf_...              -> partial
      1jgd_4s_...                      -> success
      7jgd_3ish_...                    -> partial
      0_poorcritic_2026-05-17_13-51-57_VLM (no code) -> unknown
    """
    for t in name.split("_"):
        stripped = t.lstrip("0123456789")
        if stripped in _OUTCOME:
            return _OUTCOME[stripped]
    return "unknown"


def find_vlm_rollouts(root: Path):
    """Yield all descendant dirs whose name ends in 'VLM' and contain videogen/."""
    out = []
    for p in sorted(root.rglob("*VLM")):
        if p.is_dir() and (p / "videogen").is_dir():
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=str,
                    help="rollout dir (with videogen/+observations/) OR a "
                         "parent dir to recurse for *VLM rollouts")
    ap.add_argument("--no-vis", action="store_true",
                    help="skip writing side-by-side comparison PNGs")
    ap.add_argument("--video-width",  type=int, default=848)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--match-warn", type=float, default=0.9)
    ap.add_argument("--max-success", type=int, default=None,
                    help="(parent-dir mode) cap success-bucket rollouts")
    ap.add_argument("--max-partial", type=int, default=None,
                    help="(parent-dir mode) cap partial-bucket rollouts")
    ap.add_argument("--max-failure", type=int, default=None,
                    help="(parent-dir mode) cap failure-bucket rollouts")
    ap.add_argument("--max-unknown", type=int, default=None,
                    help="(parent-dir mode) cap unknown-bucket rollouts")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        sys.exit(f"error: {input_dir} not found")

    # Single-rollout mode: input_dir itself looks like a rollout.
    if (input_dir / "videogen").is_dir():
        print(f"[rollout] {input_dir}")
        try:
            process_rollout(input_dir, args)
        except FileNotFoundError as e:
            sys.exit(f"error: {e}")
        return

    # Parent mode: find every *VLM rollout under input_dir.
    rollouts = find_vlm_rollouts(input_dir)
    if not rollouts:
        sys.exit(f"error: no *VLM rollout dirs (with videogen/) found under {input_dir}")
    print(f"found {len(rollouts)} rollout(s) under {input_dir}")

    # Optional per-outcome caps. After classifying, take the first N within each
    # bucket (sorted by full path -- deterministic across runs).
    caps = {b: getattr(args, f"max_{b}", None)
            for b in ("success", "partial", "failure", "unknown")}
    if any(c is not None for c in caps.values()):
        buckets = {b: [] for b in caps}
        for r in rollouts:
            buckets[classify_rollout(r.name)].append(r)
        kept = []
        for b, items in buckets.items():
            items_sorted = sorted(items)
            cap = caps[b]
            chosen = items_sorted if cap is None else items_sorted[:cap]
            kept.extend(chosen)
            print(f"  cap {b}: kept {len(chosen)}/{len(items_sorted)}"
                  + (f"  (cap={cap})" if cap is not None else ""))
        rollouts = sorted(kept)
        print(f"after caps: {len(rollouts)} rollouts")

    per_rollout = []
    METRIC_KEYS = ("mse", "psnr", "ssim",
                   "baseline_mse", "baseline_psnr", "baseline_ssim",
                   "relative_mse", "psnr_gain_db", "ssim_gain")
    OUTCOMES = ("success", "partial", "failure", "unknown")
    # Per-step values pooled across rollouts; outer key = outcome bucket
    # ("all" plus one per outcome). Inner key = metric.
    pooled = {b: {k: [] for k in METRIC_KEYS} for b in ("all", *OUTCOMES)}
    rollout_means = {b: {k: [] for k in METRIC_KEYS} for b in ("all", *OUTCOMES)}
    # Map: aggregate metric key -> per-step dict field name in res["per_step"]
    step_field = {
        "mse": "mse", "psnr": "psnr_db", "ssim": "ssim",
        "baseline_mse": "baseline_mse",
        "baseline_psnr": "baseline_psnr_db",
        "baseline_ssim": "baseline_ssim",
        "relative_mse": "relative_mse",
        "psnr_gain_db": "psnr_gain_db",
        "ssim_gain":    "ssim_gain",
    }

    for i, rdir in enumerate(rollouts, 1):
        outcome = classify_rollout(rdir.name)
        print(f"\n[{i}/{len(rollouts)}] [{outcome:7s}] {rdir}")
        try:
            res = process_rollout(rdir, args)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            per_rollout.append({
                "rollout_dir": str(rdir),
                "outcome": outcome,
                "error": str(e),
                "num_steps_evaluated": 0,
            })
            continue

        per_rollout.append({
            "rollout_dir": str(rdir),
            "outcome": outcome,
            "num_steps_evaluated": res["num_steps_evaluated"],
            "num_skipped": len(res["skipped"]),
            **{k: res[k] for k in METRIC_KEYS},
        })
        # Pool this rollout's step values into "all" and its outcome bucket.
        buckets = ("all", outcome)
        for k, field in step_field.items():
            vals = [s[field] for s in res["per_step"]]
            for b in buckets:
                pooled[b][k] += vals
        if res["num_steps_evaluated"]:
            for b in buckets:
                for k in METRIC_KEYS:
                    rollout_means[b][k].append(res[k]["mean"])

    def _bucket_stats(per_bucket):
        return {b: {k: _stat(v) for k, v in metrics.items()}
                for b, metrics in per_bucket.items()}

    # Count rollouts in each outcome bucket (excluding script-level errors).
    outcome_counts = {b: 0 for b in OUTCOMES}
    for r in per_rollout:
        if "error" not in r and r.get("num_steps_evaluated", 0) > 0:
            outcome_counts[r["outcome"]] += 1

    aggregate = {
        "input_dir": str(input_dir),
        "video_width": args.video_width,
        "video_height": args.video_height,
        "margin": args.margin,
        "value_range": "metrics computed on float32 RGB in [0, 1]",
        "baseline_explanation": ("baseline = MSE/PSNR/SSIM between the video's "
                                 "FIRST frame and the aligned GT post-exec obs "
                                 "(model emits no predicted motion); "
                                 "relative_mse = pred_mse / baseline_mse"),
        "outcome_explanation": ("rollout outcome classified from the dir name: "
                                "ss/s=success, ff/f=failure, "
                                "sf/fs/ish/sisg=partial, otherwise unknown"),
        "num_rollouts_found": len(rollouts),
        "num_rollouts_processed": sum(1 for r in per_rollout if "error" not in r),
        "num_rollouts_failed_to_process": sum(1 for r in per_rollout if "error" in r),
        "outcome_counts": outcome_counts,
        "total_steps_evaluated_all": len(pooled["all"]["mse"]),
        "total_steps_evaluated_by_outcome": {
            b: len(pooled[b]["mse"]) for b in OUTCOMES
        },
        "pooled_over_steps":      _bucket_stats(pooled),
        "averaged_over_rollouts": _bucket_stats(rollout_means),
        "per_rollout": per_rollout,
    }
    aggregate["caps_applied"] = caps
    if any(c is not None for c in caps.values()):
        cap_parts = [f"{b}{c}" for b, c in caps.items() if c is not None]
        out_name = "mse_results_aggregate_" + "_".join(cap_parts) + ".json"
    else:
        out_name = "mse_results_aggregate.json"
    out_path = input_dir / out_name
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    def _row(s, key, w):
        v, sd = s[key]["mean"], s[key]["std"]
        if v is None:
            return "       n/a"
        sign = "+" if key in ("psnr_gain_db", "ssim_gain") else ""
        return f"{sign}{v:.{w}f}±{sd:.{w}f}"

    print()
    print(f"rollouts: {aggregate['num_rollouts_processed']} processed, "
          f"{aggregate['num_rollouts_failed_to_process']} failed")
    print(f"by outcome: " +
          "  ".join(f"{b}={outcome_counts[b]}" for b in OUTCOMES))
    print(f"total evaluated steps: {aggregate['total_steps_evaluated_all']}")

    # Print a compact table per aggregation, with one column per outcome bucket.
    headers = ("all", *OUTCOMES)
    rows = [
        ("pred MSE",       "mse",           5),
        ("pred PSNR(dB)",  "psnr",          2),
        ("pred SSIM",      "ssim",          4),
        ("baseline MSE",   "baseline_mse",  5),
        ("baseline PSNR",  "baseline_psnr", 2),
        ("baseline SSIM",  "baseline_ssim", 4),
        ("rel MSE",        "relative_mse",  3),
        ("ΔPSNR(dB)",      "psnr_gain_db",  2),
        ("ΔSSIM",          "ssim_gain",     4),
    ]
    for title, source in [("pooled over all steps:", aggregate["pooled_over_steps"]),
                          ("averaged over rollout means:", aggregate["averaged_over_rollouts"])]:
        print()
        print(title)
        nsteps = {b: len(pooled[b]["mse"]) if title.startswith("pooled")
                  else len(rollout_means[b]["mse"]) for b in headers}
        print("                 " + "".join(f"{b+f' (n={nsteps[b]})':>22s}" for b in headers))
        for label, key, w in rows:
            line = f"  {label:<14s} "
            for b in headers:
                line += f"{_row(source[b], key, w):>22s}"
            print(line)
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
