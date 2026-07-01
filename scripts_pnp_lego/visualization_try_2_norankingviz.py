#!/usr/bin/env python
"""Explainer video for a ``--videogen`` VLM rollout — NO pairwise-ranking viz.

Variant of ``visualization_try_2.py``. Same single continuous timeline and
precise pixel-based execution detection, but the right panel does NOT reveal the
VLM's pairwise comparisons one-by-one. Instead, per step:

  1. The K candidate "future" videos all play together (once, at 0.5x), then
     freeze.
  2. The global VLM score appears below each candidate at once (no one-by-one
     ranking) and the winner is highlighted.
  3. The chosen action plays on the real robot.

Focus alternates between the two panels to direct the eye:
  * While the candidates play + are scored (the paused/generation stretch), the
    Generated Samples are bright and the Real Robot panel is dimmed.
  * While the robot executes, the Real Robot panel is bright and the Generated
    Samples dim.

EXECUTION bursts are played at 1x; PAUSED stretches are sped up. Bursts are
detected directly from the pixels (see ``robust_motion_signal``): per frame, the
count of pixels whose change deviates from the global illumination shift — a
moving arm survives, camera auto-exposure cancels — giving one clean burst/step.

Usage:
python scripts_pnp_lego/visualization_try_2_norankingviz.py \
    --run_dir /proj/vondrick3/sruthi/Appaji/diffusion_policy/data/jgd/realworld_data/jgd/2026.05.19/23.23.33_train_diffusion_unet_hybrid_stacking_image_10hz_wstate/checkpoints/epoch=0500-train_loss=0.0148/1jgd_3s_2026-05-21_15-42-35_VLM
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False


DEFAULT_RUN_DIR = (
    "/proj/vondrick3/sruthi/Appaji/diffusion_policy/data/jgd/realworld_data/jgd/"
    "2026.05.06/23.17.10_train_diffusion_unet_hybrid_pnp_lego_image/checkpoints/"
    "epoch=0150-train_loss=0.0160/0_sf_2026-05-10_14-52-03_VLM"
)

# ---------------------------------------------------------------------------
# Canvas / layout constants (1920x1080, RGB)
# ---------------------------------------------------------------------------
W, H = 1920, 1080
HEADER_H = 78
PAD = 14

# Frame rate the generated candidate clips were written at (HunyuanVideo
# server writes 24fps mp4s).
CAND_SRC_FPS = 24.0

BG = (18, 20, 26)
PANEL_BG = (30, 33, 42)
TEXT = (235, 238, 244)
SUBTLE = (150, 156, 168)
ACCENT = (92, 170, 255)       # blue
WIN = (80, 210, 120)          # green
WIN_DIM = (34, 92, 54)        # dimmed green (winner box "off" blink phase)
LOSE = (210, 80, 90)          # red
BAR_BG = (55, 60, 72)
HILITE = (250, 205, 70)       # yellow: active comparison

# Stacked layout: Real Robot panel on top (full width), Generated Samples grid
# below it (full width). A thin header strip at the very top holds the phase
# narration.
ROBOT_X0, ROBOT_X1 = PAD, W - PAD
ROBOT_Y0 = PAD
ROBOT_Y1 = 642
SAMP_X0, SAMP_X1 = PAD, W - PAD
SAMP_Y0 = ROBOT_Y1 + PAD
SAMP_Y1 = H - PAD
# vertical space reserved under each candidate video for its score bar (the
# "Sample N" label is overlaid on the video itself)
LABEL_H = 36


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
_FONT_CACHE = {}
_FONT_FILES = {
    "regular": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ],
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ],
}


def get_font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font = None
    if _HAVE_PIL:
        for p in _FONT_FILES["bold" if bold else "regular"]:
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


# ---------------------------------------------------------------------------
# Small drawing helpers (RGB uint8 numpy canvas via cv2)
# ---------------------------------------------------------------------------
def fill_rect(img, x0, y0, x1, y1, color):
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness=-1)


def border_rect(img, x0, y0, x1, y1, color, t=3):
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness=t)


def paste(img, sub, x0, y0):
    """Paste sub (RGB) into img at (x0, y0), clipped to bounds."""
    h, w = sub.shape[:2]
    x0, y0 = int(x0), int(y0)
    x1, y1 = x0 + w, y0 + h
    H_, W_ = img.shape[:2]
    if x0 >= W_ or y0 >= H_ or x1 <= 0 or y1 <= 0:
        return
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    x0c = max(0, x0); y0c = max(0, y0)
    x1c = min(W_, x1); y1c = min(H_, y1)
    img[y0c:y1c, x0c:x1c] = sub[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]


def fit_into(frame, box_w, box_h):
    """Resize keeping aspect; return (img, off_x, off_y) to center in box."""
    h, w = frame.shape[:2]
    s = min(box_w / w, box_h / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return r, (box_w - nw) // 2, (box_h - nh) // 2


class TextLayer:
    """Accumulate text draws, flush once per frame via one PIL round-trip."""

    def __init__(self, img):
        self.img = img
        self.items = []

    def add(self, xy, s, size, color=TEXT, bold=False, anchor="la"):
        self.items.append((xy, s, size, color, bold, anchor))

    def flush(self):
        if not self.items:
            return self.img
        if not _HAVE_PIL:
            for (x, y), s, size, color, bold, anchor in self.items:
                cv2.putText(self.img, s, (int(x), int(y) + size),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 32.0,
                            color, 1, cv2.LINE_AA)
            return self.img
        pim = Image.fromarray(self.img)
        d = ImageDraw.Draw(pim)
        for (x, y), s, size, color, bold, anchor in self.items:
            d.text((x, y), s, font=get_font(size, bold), fill=tuple(color),
                   anchor=anchor)
        self.img[:] = np.asarray(pim)
        return self.img


# ---------------------------------------------------------------------------
# Data discovery / loading
# ---------------------------------------------------------------------------
def resolve_rollout(run_dir, camera, override):
    """Find the real-robot rollout video.

    Priority: explicit ``--rollout`` > ``rollout_<camera>.mp4`` > flat
    ``rollout.mp4`` > any ``rollout_*.mp4`` in the dir. Single-camera runs write
    ``rollout.mp4``; two-camera runs write ``rollout_image1.mp4`` /
    ``rollout_image2.mp4``.
    """
    if override:
        return (override if os.path.isabs(override)
                else os.path.join(run_dir, override))
    cands = []
    if camera:
        cands.append(os.path.join(run_dir, f"rollout_{camera}.mp4"))
    cands.append(os.path.join(run_dir, "rollout.mp4"))
    for c in cands:
        if os.path.exists(c):
            return c
    g = sorted(glob.glob(os.path.join(run_dir, "rollout_*.mp4")))
    return g[0] if g else cands[-1]


def discover_steps(run_dir):
    """Sorted videogen step dirs that contain a ranking.json (one per step)."""
    vg = os.path.join(run_dir, "videogen")
    steps = sorted(d for d in glob.glob(os.path.join(vg, "*"))
                   if os.path.isdir(d) and
                   os.path.exists(os.path.join(d, "ranking.json")))
    return steps


def read_all_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_step(step_dir):
    """Return (ranking_dict, list_of_candidate_framelists)."""
    with open(os.path.join(step_dir, "ranking.json")) as f:
        ranking = json.load(f)
    mp4s = sorted(glob.glob(os.path.join(step_dir, "*.mp4")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    cand = [read_all_frames(p) for p in mp4s]
    return ranking, cand


# ---------------------------------------------------------------------------
# Precise execution detection
# ---------------------------------------------------------------------------
def robust_motion_signal(rollout_path, work_w=160, work_h=90, pix_thr=18):
    """Per-frame illumination-robust motion energy.

    For each frame we diff against the previous frame, subtract the *median*
    of that diff (the global brightness shift induced by camera auto-exposure,
    which is spatially uniform), and count the fraction of pixels whose
    residual still exceeds ``pix_thr``. A moving arm is a localised change that
    survives; a global exposure change is cancelled. Returns (signal, fps,
    nframes).
    """
    cap = cv2.VideoCapture(str(rollout_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev = None
    sig = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (work_w, work_h)),
                         cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is None:
            sig.append(0.0)
        else:
            d = g - prev
            d = d - np.median(d)
            sig.append(float(np.count_nonzero(np.abs(d) > pix_thr)) / d.size)
        prev = g
    cap.release()
    return np.asarray(sig), fps, len(sig)


def smooth(x, k):
    k = max(1, int(k))
    if k <= 1:
        return x.astype(float)
    return np.convolve(x, np.ones(k) / k, mode="same")


def find_execution_bursts(sig, fps, n_expected, min_dur=0.25, merge_gap=0.8,
                          verbose=True):
    """Locate execution motion bursts in the robust signal.

    A burst is a contiguous run above an adaptive threshold (sitting a little
    above the still baseline). Short flickers are dropped and near-adjacent
    runs merged. If more bursts than ``n_expected`` survive, the highest-energy
    ``n_expected`` are kept; the start of each is then tightened to the first
    frame where the raw (unsmoothed) signal genuinely rises, so the 1x replay
    begins exactly when the arm starts moving. Returns a list of (start, end)
    inclusive frame indices, time-ordered.
    """
    sm = smooth(sig, int(0.5 * fps))
    base = float(np.percentile(sm, 50))
    hi = float(np.percentile(sm, 98))
    thr = base + 0.15 * max(1e-9, hi - base)
    mask = sm > thr
    if verbose:
        print(f"[motion] base={base:.4f} hi={hi:.4f} thr={thr:.4f}")

    n = len(mask)
    runs = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    # merge near-adjacent runs
    merged = []
    gap = int(merge_gap * fps)
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    min_len = int(min_dur * fps)
    merged = [r for r in merged if (r[1] - r[0] + 1) >= min_len]

    if len(merged) > n_expected:
        merged = sorted(merged,
                        key=lambda r: -float(sm[r[0]:r[1] + 1].sum()))[:n_expected]
        merged = sorted(merged, key=lambda r: r[0])

    # tighten each burst's start onto the first real rise in the raw signal
    raw_thr = base + 0.04 * max(1e-9, hi - base)
    out = []
    for s, e in merged:
        lo = max(0, s - int(0.6 * fps))
        s2 = s
        for f in range(lo, e + 1):
            if sig[f] > raw_thr:
                s2 = f
                break
        out.append((s2, e))
    return out


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------
def base_canvas():
    img = np.empty((H, W, 3), np.uint8)
    img[:] = BG
    return img


def overlay_tag(img, tl, x, y, text, size, color=TEXT, anchor="lt", pad=5):
    """Draw `text` on a darkened plate so it reads over video. `anchor` gives
    which corner (x, y) is: lt=left-top, rt=right-top, rb=right-bottom,
    lb=left-bottom."""
    w = int(len(text) * size * 0.62) + 2 * pad
    h = size + 2 * pad
    rx0 = x - w if anchor in ("rt", "rb") else x
    ry0 = y - h if anchor in ("rb", "lb") else y
    rx1, ry1 = rx0 + w, ry0 + h
    # semi-transparent dark plate behind the text
    H_, W_ = img.shape[:2]
    cx0, cy0 = max(0, rx0), max(0, ry0)
    cx1, cy1 = min(W_, rx1), min(H_, ry1)
    if cx1 > cx0 and cy1 > cy0:
        roi = img[cy0:cy1, cx0:cx1].astype(np.float32) * 0.3
        img[cy0:cy1, cx0:cx1] = roi.astype(np.uint8)
    tl.add((rx0 + pad, ry0 + pad), text, size, color, bold=True, anchor="la")


def overlay_result(frame, success):
    """Stamp the rollout outcome on an already-composed frame: a colored border
    around the whole canvas plus a compact pill (green check + SUCCESS /
    red cross + FAILED) near the TOP of the Real Robot panel, so the center of
    the final frame stays unobstructed."""
    H_, W_ = frame.shape[:2]
    col = WIN if success else LOSE
    label = "SUCCESS" if success else "FAILED"
    bt = 16
    border_rect(frame, bt // 2, bt // 2, W_ - bt // 2, H_ - bt // 2, col, bt)
    cx = W_ // 2
    pw, ph = 340, 78
    cy = ROBOT_Y0 + 26 + ph // 2  # sit near the top, not the middle
    x0, y0, x1, y1 = cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2
    fill_rect(frame, x0, y0, x1, y1, col)
    border_rect(frame, x0, y0, x1, y1, (255, 255, 255), 2)
    white = (255, 255, 255)
    gx = x0 + 52
    if success:
        cv2.line(frame, (gx - 22, cy + 2), (gx - 6, cy + 20), white, 7, cv2.LINE_AA)
        cv2.line(frame, (gx - 6, cy + 20), (gx + 26, cy - 22), white, 7, cv2.LINE_AA)
    else:
        cv2.line(frame, (gx - 20, cy - 20), (gx + 20, cy + 20), white, 7, cv2.LINE_AA)
        cv2.line(frame, (gx - 20, cy + 20), (gx + 20, cy - 20), white, 7, cv2.LINE_AA)
    tl = TextLayer(frame)
    tl.add(((gx + 34 + x1) // 2, cy + 1), label, 40, white, bold=True,
           anchor="mm")
    tl.flush()


def rows_for(K, cols, rows_arrangement):
    """Resolve the per-row sample counts: explicit `rows_arrangement` if it sums
    to K, else rows of `cols`."""
    if rows_arrangement and sum(rows_arrangement) == K:
        return list(rows_arrangement)
    cols = max(1, min(cols, K))
    arr, left = [], K
    while left > 0:
        arr.append(min(cols, left))
        left -= min(cols, left)
    return arr


def grid_geometry(K, cols=3, rows_arrangement=None):
    """Lay the K candidate videos out, label + vote bar reserved underneath.

    `rows_arrangement`, if given, is an explicit per-row count list (e.g.
    [2, 1, 2] for 5 samples) summing to K; otherwise rows of `cols` are used.
    All videos are sized uniformly as large as the tightest row allows while
    keeping the 848:480 aspect ratio, and each row is centered horizontally.
    Returns a list of (thumb_x, thumb_y, thumb_w, thumb_h, cell_x0, cell_w).
    """
    arr = rows_for(K, cols, rows_arrangement)
    R = len(arr)
    avail_w = SAMP_X1 - SAMP_X0
    avail_h = SAMP_Y1 - SAMP_Y0
    gx, gy = 20, 14
    row_h = (avail_h - (R - 1) * gy) / R
    th_box = row_h - LABEL_H
    # uniform video size: limited by the widest row (width) and row height
    max_c = max(arr)
    cell_w_min = (avail_w - (max_c - 1) * gx) / max_c
    s = min(cell_w_min / 848.0, th_box / 480.0)
    tw, th = int(848 * s), int(480 * s)
    geo = []
    k = 0
    for r, c in enumerate(arr):
        row_w = c * tw + (c - 1) * gx
        x_start = SAMP_X0 + (avail_w - row_w) / 2  # center this row
        y0 = SAMP_Y0 + r * (row_h + gy)
        for j in range(c):
            tx = int(x_start + j * (tw + gx))
            geo.append((tx, int(y0), tw, th, tx, tw))
            k += 1
    return geo


def draw_left(img, robot_frame, label, tl, dim=False):
    """Draw the Real Robot video in the full-width top panel. The "Real Robot"
    label is overlaid on the top-left of the video and the status caption
    (speed / "Executing Sample W") on the bottom-right."""
    x0, x1, y0, y1 = ROBOT_X0, ROBOT_X1, ROBOT_Y0, ROBOT_Y1
    r, ox, oy = fit_into(robot_frame, x1 - x0, y1 - y0)
    vx0, vy0 = x0 + ox, y0 + oy
    vx1, vy1 = vx0 + r.shape[1], vy0 + r.shape[0]
    paste(img, r, vx0, vy0)
    overlay_tag(img, tl, vx0 + 8, vy0 + 8, "Real Robot", 30,
                SUBTLE if dim else TEXT, anchor="lt")
    if label:
        overlay_tag(img, tl, vx1 - 8, vy1 - 8, label, 22, SUBTLE, anchor="rb")


DIM = 0.35  # brightness multiplier for the dimmed (out-of-focus) panel


def compose(left_frame, left_label, cand_imgs, geo, phase, phase_color,
            votes, winner_idx, dim_left=False, dim_samples=False,
            winner_blink_on=True, left_bright=None, samp_bright=None):
    """Render one output frame.

    votes           : global VLM score list[int] (len K) shown below each
                      candidate, or None (during generation, before scoring)
    winner_idx      : highlight this cell green, or None
    dim_left        : darken the Real Robot panel (focus is on the samples)
    dim_samples     : darken all candidate videos (focus is on the robot)
    winner_blink_on : when False, draw the winner box in dim green instead of
                      bright green (drives the acceptance blink)
    left_bright /
    samp_bright     : explicit brightness multipliers in [0, 1] that override
                      dim_left / dim_samples (used to crossfade the focus shift
                      smoothly instead of a hard cut)
    """
    img = base_canvas()
    tl = TextLayer(img)

    lb = left_bright if left_bright is not None else (DIM if dim_left else 1.0)
    sb = samp_bright if samp_bright is not None else (DIM if dim_samples else 1.0)

    lf = ((left_frame.astype(np.float32) * lb).astype(np.uint8)
          if lb < 0.999 else left_frame)
    draw_left(img, lf, left_label, tl, dim=lb < 0.9)

    K = len(cand_imgs)
    # bars show each sample's pairwise win count out of the K-1 head-to-head
    # comparisons it took part in, so near-ties read as similar bar lengths.
    win_denom = max(1, K - 1, (max(votes) if votes else 0))
    for k in range(K):
        tx, ty, tw, th, cx0, cw = geo[k]
        is_winner = winner_idx is not None and k == winner_idx

        thumb = cv2.resize(cand_imgs[k], (tw, th), interpolation=cv2.INTER_AREA)
        if sb < 0.999:
            thumb = (thumb.astype(np.float32) * sb).astype(np.uint8)
        paste(img, thumb, tx, ty)

        bcol, bt = (70, 76, 90), 2
        if is_winner:
            bcol, bt = (WIN if winner_blink_on else WIN_DIM), 5
        border_rect(img, tx, ty, tx + tw, ty + th, bcol, bt)

        # "Sample N" overlaid top-left of the video; "SELECTED" top-right
        overlay_tag(img, tl, tx + 6, ty + 6, f"Candidate {k}", 19,
                    WIN if is_winner else TEXT, anchor="lt")
        if is_winner and votes is not None:
            tag_col = WIN if winner_blink_on else WIN_DIM
            overlay_tag(img, tl, tx + tw - 6, ty + 6, "SELECTED", 18, tag_col,
                        anchor="rt")

        # win-count bar underneath the video (number of head-to-head wins)
        if votes is not None:
            by0 = ty + th + 8
            by1 = by0 + 16
            tl.add((tx, by0 - 3), "Wins:", 18, SUBTLE, bold=True)
            bx0, bx1 = tx + 66, tx + tw - 52
            fill_rect(img, bx0, by0, bx1, by1, BAR_BG)
            frac = votes[k] / win_denom
            fcol = WIN if is_winner else ACCENT
            fill_rect(img, bx0, by0, bx0 + int((bx1 - bx0) * frac), by1, fcol)
            tl.add((bx1 + 6, by0 - 3), f"{votes[k]}/{K - 1}", 19, TEXT,
                   bold=True)

    return tl.flush()


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------
class Writer:
    def __init__(self, path, fps):
        self.path = str(path)
        self.fps = fps
        try:
            import imageio
            self.w = imageio.get_writer(self.path, fps=fps, codec="libx264",
                                        quality=8, macro_block_size=None)
            self.kind = "imageio"
        except Exception as e:
            print(f"[writer] imageio unavailable ({e}); using cv2.VideoWriter")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.w = cv2.VideoWriter(self.path, fourcc, fps, (W, H))
            self.kind = "cv2"

    def write(self, rgb):
        if self.kind == "imageio":
            self.w.append_data(rgb)
        else:
            self.w.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.kind == "imageio":
            self.w.close()
        else:
            self.w.release()


# ---------------------------------------------------------------------------
# Right-panel state plan for one (sped-up) pause region
# ---------------------------------------------------------------------------
def build_pause_states(votes_final, winner_idx, K, n_gen, n_score, n_out=None,
                       n_denoise=0, executed=True):
    """Return the right-panel state dicts for one pause region (no ranking viz).

    Two sub-phases: generation (`n_gen` frames — candidates playing, no score)
    then global score (`n_score` frames — the full VLM tally shown below every
    candidate at once with the winner highlighted; no one-by-one pairwise
    reveal). The first `n_denoise` of the generation frames are the diffusion
    denoise dissolve (noise -> first frame) and get their own label. Throughout
    the pause the Real Robot panel is dimmed and the samples are bright. If
    `n_out` is larger than the natural length, the score frame is held to pad up.
    """
    states = []

    # phase 1: generation — diffusion denoise dissolve, then candidates playing
    for g in range(n_gen):
        if g < n_denoise:
            phase = (f"A video diffusion model denoises random noise into "
                     f"{K} candidate futures")
        else:
            phase = f"Generating {K} candidate futures from the policy"
        states.append(dict(
            phase=phase,
            phase_color=ACCENT, votes=None, winner_idx=None,
            dim_left=True, dim_samples=False))

    # phase 2: global score appears below every candidate at once
    tail = ("executing on robot" if executed
            else "run ended before execution")
    for _ in range(n_score):
        states.append(dict(
            phase=f"VLM scores all candidates  ->  Sample {winner_idx} wins "
                  f"(score {votes_final[winner_idx]})  ->  {tail}",
            phase_color=WIN, votes=list(votes_final), winner_idx=winner_idx,
            dim_left=True, dim_samples=False))

    if not states:
        states.append(dict(phase="", phase_color=ACCENT, votes=None,
                           winner_idx=None, dim_left=True, dim_samples=False))
    # hold the last (score) frame to pad up to n_out when requested
    target = len(states) if n_out is None else max(n_out, len(states))
    while len(states) < target:
        states.append(states[-1])
    return states


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def cand_frame(cand, K, t_out, out_fps, speed=0.5, src_fps=CAND_SRC_FPS):
    """Candidate frame for each sample at pause-region output index t_out.

    Plays each generated clip through exactly once at `speed` (0.5 = half
    speed) starting at the top of the pause region, then holds on the last
    frame for the rest of the pause (so the pairwise ranking proceeds over a
    frozen still rather than a looping clip).
    """
    out = []
    for k in range(K):
        n = max(1, len(cand[k]))
        src = min(n - 1, int((t_out / out_fps) * src_fps * speed))
        out.append(cand[k][src])
    return out


def denoise_cand_imgs(cand, K, s, n_steps, seed):
    """Per-candidate frame at discrete diffusion-denoise step `s` (0-based, of
    `n_steps`). Each step crossfades further toward the clean first frame —
    sigma = 1 - (s+1)/n_steps — and the noise is re-sampled every step so the
    static visibly jumps, evoking iterative diffusion sampling. The last step is
    fully clean (sigma=0), so it hands off seamlessly to clip playback.
    """
    sigma = 1.0 - (s + 1) / max(1, n_steps)
    out = []
    for k in range(K):
        target = cand[k][0].astype(np.float32)
        rng = np.random.default_rng(seed + k * 1000 + s)
        noise = rng.integers(0, 256, target.shape, dtype=np.uint8).astype(np.float32)
        mix = (1.0 - sigma) * target + sigma * noise
        out.append(np.clip(mix, 0, 255).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out", default=None,
                    help="output mp4 (default <run_dir>/visualization_try_2.mp4)")
    ap.add_argument("--fps", type=float, default=30.0, help="output fps")
    ap.add_argument("--pause_speedup", type=int, default=10,
                    help="play paused (generation/ranking) stretches this many "
                         "times faster. Execution bursts always play at 1x.")
    ap.add_argument("--cand_speed", type=float, default=0.5,
                    help="playback speed of the generated candidate clips "
                         "(0.5 = half speed). Each clip plays through once then "
                         "freezes on its last frame while ranking proceeds.")
    ap.add_argument("--denoise_sec", type=float, default=2,
                    help="seconds of the diffusion denoise dissolve (random "
                         "noise -> each clip's first frame) shown before the "
                         "candidates play. 0 disables it.")
    ap.add_argument("--denoise_seed", type=int, default=0,
                    help="RNG seed for the per-candidate denoise static "
                         "(reproducible noise).")
    ap.add_argument("--denoise_steps", type=int, default=50,
                    help="number of discrete diffusion timesteps in the denoise "
                         "intro (noise re-sampled each step, shown as a t=N->0 "
                         "countdown). Higher = smoother.")
    ap.add_argument("--gen_hold", type=float, default=0.4,
                    help="seconds the candidate clips stay frozen on their last "
                         "frame after the play-through before the score appears.")
    ap.add_argument("--score_sec", type=float, default=2.5,
                    help="seconds the global VLM score is shown below the "
                         "candidates before the robot executes.")
    ap.add_argument("--winner_blink_hz", type=float, default=3.0,
                    help="blink rate (Hz) of the winner's green box while it is "
                         "being accepted (score phase). 0 disables blinking.")
    ap.add_argument("--winner_blink_count", type=int, default=2,
                    help="number of times the winner box flashes when it first "
                         "appears, then it holds solid green. 0 = blink for the "
                         "whole score phase.")
    ap.add_argument("--outcome", choices=["success", "failure"], default=None,
                    help="if given, hold an end card at the end of the video "
                         "stamping the rollout result (green SUCCESS / red "
                         "FAILED) over the final robot frame.")
    ap.add_argument("--result_sec", type=float, default=2.5,
                    help="seconds to hold the success/failure end card.")
    ap.add_argument("--input_sec", type=float, default=0.6,
                    help="seconds to open on the bright robot input observation "
                         "(undimmed) before it dims and generation begins. "
                         "0 disables the intro.")
    ap.add_argument("--input_frame_idx", type=int, default=60,
                    help="rollout frame to freeze on for the input intro "
                         "(default 60 — skips the dark auto-exposure ramp at "
                         "the very start).")
    ap.add_argument("--content_width", type=int, default=1280,
                    help="display width (px) of the Real Robot video; the whole "
                         "canvas is sized to fit exactly this wide content "
                         "(robot video on top, samples grid below).")
    ap.add_argument("--grid_cols", type=int, default=5,
                    help="columns in the candidate-video grid, used when "
                         "--grid_rows does not apply (e.g. 10 samples -> 5 "
                         "cols = two rows).")
    ap.add_argument("--grid_rows", default="3,2",
                    help="explicit per-row sample counts, comma-separated "
                         "(default '3,2' for 5 samples -> a 3-then-2 pyramid "
                         "under the Real Robot panel). Used only when it sums "
                         "to the number of samples; otherwise falls back to "
                         "--grid_cols. Pass '' to always use cols.")
    ap.add_argument("--exec_pad_pre", type=float, default=0.25,
                    help="seconds of real-time lead-in before each detected "
                         "execution burst")
    ap.add_argument("--exec_pad_post", type=float, default=0.6,
                    help="seconds of real-time settle shown after each burst")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="only render the first N steps (debug)")
    ap.add_argument("--camera", default="image2",
                    help="which camera to show for two-camera runs "
                         "(rollout_<camera>.mp4). Ignored when a flat "
                         "rollout.mp4 exists.")
    ap.add_argument("--rollout", default=None,
                    help="explicit rollout mp4 path (overrides --camera lookup)")
    args = ap.parse_args()

    run_dir = args.run_dir
    rollout_path = resolve_rollout(run_dir, args.camera, args.rollout)
    assert os.path.exists(rollout_path), f"rollout not found: {rollout_path}"
    if args.out:
        out_path = args.out
    else:
        # camera-aware default name so two-camera runs don't clobber each other
        roll_stem = os.path.splitext(os.path.basename(rollout_path))[0]
        suffix = roll_stem[len("rollout"):]  # "" or "_image2"
        out_path = os.path.join(
            run_dir, f"visualization_try_2_norankingviz{suffix}.mp4")

    all_steps = discover_steps(run_dir)
    n_all = len(all_steps)
    assert n_all > 0, "no videogen step folders with ranking.json found"
    steps = all_steps[:args.max_steps] if args.max_steps else all_steps
    n_steps = len(steps)
    print(f"[info] rollout : {rollout_path}")
    print(f"[info] {n_all} decision steps"
          + (f" (rendering first {n_steps})" if n_steps != n_all else ""))

    # ---- precise execution windows from the pixels ----
    # Detect over the FULL rollout (all steps) so burst<->step pairing is
    # correct even when --max_steps truncates rendering.
    print("[info] scanning rollout for execution bursts ...")
    sig, fps_in, nframes = robust_motion_signal(rollout_path)
    bursts = find_execution_bursts(sig, fps_in, n_expected=n_all)
    bursts = bursts[:n_steps]
    n_burst = len(bursts)
    if n_burst != n_steps:
        print(f"[warn] detected {n_burst} execution burst(s) but have {n_steps} "
              f"step(s). Steps {n_burst}..{n_steps - 1} were generated/ranked but "
              f"never executed (run stopped) -> shown as pause-only segments.")

    pad_pre = int(args.exec_pad_pre * fps_in)
    pad_post = int(args.exec_pad_post * fps_in)
    # Build, per step, the [pause_region) and [exec_region] frame spans that
    # together tile the rollout. The first `n_burst` steps each end in a 1x
    # execution; any trailing steps (generated+ranked but never executed
    # because the run stopped) become pause-only segments that split whatever
    # rollout frames remain after the last burst.
    segs = []  # (pause_start, exec_start, exec_end, has_exec)
    prev_end = -1
    for i in range(n_burst):
        bs, be = bursts[i]
        es = max(prev_end + 1, bs - pad_pre)
        ee = be + pad_post
        if i + 1 < n_burst:
            ee = min(ee, bursts[i + 1][0] - 1)
        ee = min(ee, nframes - 1)
        ps = prev_end + 1
        segs.append((ps, es, ee, True))
        prev_end = ee
        print(f"  step {i}: pause f{ps}-{es-1} ({(es-ps)/fps_in:5.1f}s) | "
              f"exec f{es}-{ee} ({(ee-es+1)/fps_in:4.1f}s @1x)")
    n_trail = n_steps - n_burst
    if n_trail > 0:
        # split the remaining rollout tail evenly across the un-executed steps
        rem_lo, rem_hi = prev_end + 1, nframes - 1
        span = max(0, rem_hi - rem_lo + 1)
        for t in range(n_trail):
            ps = rem_lo + t * span // n_trail
            pe = rem_lo + (t + 1) * span // n_trail - 1  # inclusive pause end
            pe = max(ps, min(pe, nframes - 1))
            # exec_start > exec_end => empty exec region
            segs.append((ps, pe + 1, pe, False))
            print(f"  step {n_burst + t}: pause f{ps}-{pe} "
                  f"({(pe - ps + 1)/fps_in:5.1f}s) | no execution (pause-only)")

    fps = args.fps
    rows_arr = None
    if args.grid_rows.strip():
        try:
            rows_arr = [int(x) for x in args.grid_rows.split(",") if x.strip()]
        except ValueError:
            print(f"[warn] could not parse --grid_rows '{args.grid_rows}'; "
                  f"using --grid_cols {args.grid_cols}")
    K0 = max(1, len(glob.glob(os.path.join(steps[0], "*.mp4"))))

    # ---- size the canvas to exactly fit the content (no wasted margins) ----
    # Real Robot video (16:9) on top at content_width; samples grid (848:480)
    # below, the widest row also spanning content_width.
    global W, H, ROBOT_X0, ROBOT_X1, ROBOT_Y0, ROBOT_Y1
    global SAMP_X0, SAMP_X1, SAMP_Y0, SAMP_Y1
    arr = rows_for(K0, args.grid_cols, rows_arr)
    gx, gy = 20, 14
    RW = max(320, args.content_width)
    RH = int(round(RW * 360 / 640))
    max_c = max(arr)
    SW = (RW - (max_c - 1) * gx) // max_c
    SH = int(round(SW * 480 / 848))
    grid_h = len(arr) * (SH + LABEL_H) + (len(arr) - 1) * gy
    W = PAD + RW + PAD
    H = PAD + RH + PAD + grid_h + PAD
    W += W % 2                      # libx264 needs even dimensions
    H += H % 2
    ROBOT_X0, ROBOT_X1, ROBOT_Y0, ROBOT_Y1 = PAD, PAD + RW, PAD, PAD + RH
    SAMP_X0, SAMP_X1 = PAD, PAD + RW
    SAMP_Y0 = ROBOT_Y1 + PAD
    SAMP_Y1 = SAMP_Y0 + grid_h
    print(f"[layout] canvas {W}x{H}, robot {RW}x{RH}, sample {SW}x{SH}, "
          f"rows {arr}")

    writer = Writer(out_path, fps)
    cap = cv2.VideoCapture(str(rollout_path))
    cur_idx = 0
    total_out = 0
    geo = grid_geometry(K0, args.grid_cols, rows_arr)

    for si, step_dir in enumerate(steps):
        ranking, cand = load_step(step_dir)
        K = len(cand)
        geo = grid_geometry(K, args.grid_cols, rows_arr)
        votes_final = ranking.get("votes", [0] * K)
        winner_idx = int(ranking.get("winner_idx", int(np.argmax(votes_final))))
        pairs = ranking.get("pairs", [])
        ps, es, ee, has_exec = segs[si]

        # ---------- PAUSE region: left dimmed/sped up, samples bright ----------
        pause_len = es - ps
        # generation sub-phase lasts one full 0.5x play-through of the longest
        # candidate clip (+ a short frozen hold) so the score only appears once
        # every clip is paused on its last frame.
        max_clip = max((len(c) for c in cand), default=1)
        play_once = int(np.ceil((max_clip - 1) * fps
                                / (CAND_SRC_FPS * max(1e-6, args.cand_speed))))
        # generation = diffusion denoise dissolve -> clip play-through -> hold
        n_denoise = max(0, int(round(args.denoise_sec * fps)))
        n_gen = n_denoise + play_once + int(round(args.gen_hold * fps))
        # Winner-blink timing computed up front: the SCORE PHASE *is* the blink.
        # The green winner appears together with the score bars, flashes `count`
        # times, lands on a one-period solid tail, then execution begins — so
        # there is no bars-only delay before the green and no solid-green hold
        # after the blink.
        blink_period = (max(1, int(round(fps / (2 * args.winner_blink_hz))))
                        if args.winner_blink_hz > 0 else 0)
        blink_halves = 2 * max(0, args.winner_blink_count)  # on+off per blink
        blink_total = (blink_halves + 1) * blink_period      # flashes + tail
        n_score = (blink_total if blink_total > 0
                   else max(1, int(round(args.score_sec * fps))))
        # the paused robot is held still, so just sample it to the narration
        # length — no extra hold-padding that would stretch the score phase.
        narration = n_gen + n_score
        n_out = narration
        states = build_pause_states(votes_final, winner_idx, K, n_gen, n_score,
                                    n_out=n_out, n_denoise=n_denoise,
                                    executed=has_exec)
        n_out = len(states)
        n_steps = max(1, args.denoise_steps)
        # sample exactly n_out source frames across the pause (monotonic,
        # forward-only); reuse the last read when n_out exceeds pause_len.
        left_frames = []
        last_rgb = None
        if pause_len > 0:
            targets = [ps + (i * pause_len) // n_out for i in range(n_out)]
            for s in targets:
                while cur_idx <= s:
                    ok, f = cap.read()
                    if not ok:
                        break
                    last_rgb = cv2.cvtColor(cv2.resize(f, (640, 360)),
                                            cv2.COLOR_BGR2RGB)
                    cur_idx += 1
                left_frames.append(last_rgb if last_rgb is not None
                                   else np.full((360, 640, 3), 40, np.uint8))
        # advance to the execution start so the exec region reads correctly
        while cur_idx < es:
            ok, f = cap.read()
            if not ok:
                break
            cur_idx += 1
        if not left_frames:
            left_frames = [np.full((360, 640, 3), 40, np.uint8)]
        eff_speed = max(1, int(round(pause_len / max(1, n_out))))
        # the winner flashes from the moment the scores appear (= start of the
        # score phase), ending on the solid tail right before execution.
        first_win_t = next((i for i, s in enumerate(states)
                            if s["winner_idx"] is not None), None)
        blink_start = first_win_t if first_win_t is not None else n_out

        # input intro (step 0 only): open on the bright robot observation (the
        # system's input) with the candidates ALREADY present as raw diffusion
        # noise (dimmed, exactly like every other step's out-of-focus samples).
        # Then it hard-switches into generation (robot dims, candidates
        # brighten) — the same cut used between all the later steps. Because the
        # noise candidates are already on screen, nothing "pops up". Use a frame
        # a little way in (default 60) to skip the dark exposure ramp.
        if si == 0 and args.input_sec > 0:
            n_hold = max(1, int(round(args.input_sec * fps)))
            intro_frame = left_frames[0]
            cap2 = cv2.VideoCapture(str(rollout_path))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.input_frame_idx))
            ok2, f2 = cap2.read()
            cap2.release()
            if ok2:
                intro_frame = cv2.cvtColor(cv2.resize(f2, (640, 360)),
                                           cv2.COLOR_BGR2RGB)
            # candidates start as exactly what the generation phase shows first
            # (denoise step 0 noise), so the handoff is seamless.
            if n_denoise > 0:
                intro_cands = denoise_cand_imgs(cand, K, 0, n_steps,
                                                args.denoise_seed)
            else:
                intro_cands = cand_frame(cand, K, 0, fps, speed=args.cand_speed)
            for _ in range(n_hold):
                frame = compose(intro_frame, "Input observation", intro_cands,
                                geo, "", ACCENT, None, None,
                                dim_left=False, dim_samples=True)
                writer.write(frame)
                total_out += 1

        for t in range(n_out):
            st = states[t]
            lf = left_frames[min(t, len(left_frames) - 1)]
            # the green winner is shown for the whole score phase and flashes
            # `count` times from its first frame, then a solid tail before exec.
            blink_on = True
            if (blink_period and blink_halves and st["winner_idx"] is not None
                    and t >= blink_start):
                cyc = (t - blink_start) // blink_period
                if cyc < blink_halves:
                    blink_on = cyc % 2 == 0  # flash; solid tail lands bright
            # stepped diffusion denoise for the first n_denoise frames (with a
            # t=N->0 countdown), then the clips play (cand_frame offset so the
            # play-through starts right after the denoise)
            phase = st["phase"]
            if t < n_denoise:
                s = min(n_steps - 1, int(t / n_denoise * n_steps))
                cand_imgs = denoise_cand_imgs(cand, K, s, n_steps,
                                              args.denoise_seed)
                phase = (f"A video diffusion model denoises random noise into "
                         f"{K} candidate futures   (diffusion step t = "
                         f"{n_steps - s})")
            else:
                cand_imgs = cand_frame(cand, K, t - n_denoise, fps,
                                       speed=args.cand_speed)
            frame = compose(
                lf, f"Generating + scoring — 5x speed up",
                cand_imgs, geo,
                phase, st["phase_color"], st["votes"], st["winner_idx"],
                dim_left=st["dim_left"], dim_samples=st["dim_samples"],
                winner_blink_on=blink_on)
            writer.write(frame)
            total_out += 1

        # ---------- EXEC region: left bright at 1x, samples dimmed ----------
        win_cand_last = [c[-1] for c in cand]
        while cur_idx <= ee:
            ok, f = cap.read()
            if not ok:
                break
            cur_idx += 1
            lf = cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2RGB)
            frame = compose(
                lf, f"Executing Sample {winner_idx}",
                win_cand_last, geo,
                f"Executing chosen action (Sample {winner_idx}) on the real robot",
                WIN, list(votes_final), winner_idx,
                dim_left=False, dim_samples=True)
            writer.write(frame)
            total_out += 1

        del cand
        print(f"[step {si}] done. cumulative out frames={total_out} "
              f"(~{total_out / fps:.1f}s)")

    # ---------- end card: stamp the rollout outcome on the final frame ----------
    if args.outcome is not None:
        success = args.outcome == "success"
        n_end = max(1, int(round(args.result_sec * fps)))
        for _ in range(n_end):
            frame = compose(lf, "", win_cand_last, geo, "", WIN,
                            list(votes_final), winner_idx,
                            dim_left=False, dim_samples=True)
            overlay_result(frame, success)
            writer.write(frame)
            total_out += 1
        print(f"[result] stamped {args.outcome.upper()} end card "
              f"({n_end} frames)")

    cap.release()
    writer.close()
    print(f"[done] wrote {out_path}  "
          f"({total_out} frames, ~{total_out / fps:.1f}s @ {fps:g}fps)")


if __name__ == "__main__":
    sys.exit(main())
