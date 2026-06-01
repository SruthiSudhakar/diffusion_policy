#!/usr/bin/env python
"""Build an explainer video for a ``--videogen`` VLM rollout (take 2).

This renders ONE continuous timeline. The left panel is the *real* robot
rollout (rollout.mp4) played straight through; the right panel shows, for each
decision step, the K candidate "future" videos the generator produced, the VLM
pairwise rankings revealed one-by-one with a running vote tally, and finally the
chosen sample being executed.

The whole thing is driven off a single forward pass through rollout.mp4 with a
two-speed playback:

  * PAUSED stretches (robot holding still while video-gen + VLM ranking happen)
    are played back ``--pause_speedup`` times faster (default 10x). While they
    play, the right panel animates that step's generation -> pairwise ranking ->
    winner reveal, time-stretched to fill exactly the sped-up pause.
  * EXECUTION bursts (robot actually moving the chosen action) are played at
    1x (real time), with the winning sample highlighted on the right.

The execution bursts are detected *precisely* and directly from the pixels, with
no reliance on in-code timestamps or fragile mtime->frame alignment. The trick
(see ``robust_motion_signal``) is to measure, per frame, the number of pixels
whose frame-to-frame change deviates from the *global* illumination shift. A
moving arm is a spatially-localised change; camera auto-exposure is a global
shift that this subtraction cancels. The result is one clean burst per step.

Usage:
  python scripts_pnp_lego/visualization_try_2.py \
      --run_dir /proj/vondrick3/sruthi/Appaji/diffusion_policy/data/jgd/realworld_data/jgd/2026.05.19/23.23.33_train_diffusion_unet_hybrid_stacking_image_10hz_wstate/checkpoints/epoch=0500-train_loss=0.0148/50jgd_4s_2026-05-21_22-52-07_VLM 
      
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
LOSE = (210, 80, 90)          # red
BAR_BG = (55, 60, 72)
HILITE = (250, 205, 70)       # yellow: active comparison

# Left (real robot) panel
LEFT_X0, LEFT_X1 = PAD, 744
# Right region: grid of candidate videos + ranking
RIGHT_X0 = 762
RIGHT_X1 = W - PAD
# vertical space reserved under each candidate video for its label + vote bar
LABEL_H = 52


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
    fill_rect(img, 0, 0, W, HEADER_H, PANEL_BG)
    return img


def grid_geometry(K, cols=3, rows_arrangement=None):
    """Lay the K candidate videos out, label + vote bar reserved underneath.

    `rows_arrangement`, if given, is an explicit per-row count list (e.g.
    [2, 1, 2] for 5 samples) summing to K; otherwise rows of `cols` are used.
    All videos are sized uniformly as large as the tightest row allows while
    keeping the 848:480 aspect ratio, and each row is centered horizontally.
    Returns a list of (thumb_x, thumb_y, thumb_w, thumb_h, cell_x0, cell_w).
    """
    if rows_arrangement and sum(rows_arrangement) == K:
        arr = list(rows_arrangement)
    else:
        cols = max(1, min(cols, K))
        arr, left = [], K
        while left > 0:
            arr.append(min(cols, left))
            left -= min(cols, left)
    R = len(arr)
    top = HEADER_H + PAD
    avail_w = RIGHT_X1 - RIGHT_X0
    avail_h = H - top - PAD
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
        x_start = RIGHT_X0 + (avail_w - row_w) / 2  # center this row
        y0 = top + r * (row_h + gy)
        for j in range(c):
            tx = int(x_start + j * (tw + gx))
            geo.append((tx, int(y0), tw, th, tx, tw))
            k += 1
    return geo


def draw_left(img, left_frame, label, tl):
    box_w = LEFT_X1 - LEFT_X0
    box_h = H - (HEADER_H + PAD) - PAD - 30
    y0 = HEADER_H + PAD
    fill_rect(img, LEFT_X0, y0, LEFT_X1, y0 + box_h, PANEL_BG)
    r, ox, oy = fit_into(left_frame, box_w, box_h)
    paste(img, r, LEFT_X0 + ox, y0 + oy)
    border_rect(img, LEFT_X0, y0, LEFT_X1, y0 + box_h, (70, 76, 90), 2)
    tl.add((LEFT_X0 + 6, y0 + box_h + 4), label, 22, SUBTLE, bold=True)


def compose(left_frame, left_label, cand_imgs, geo, phase, phase_color,
            votes, active_pair, verdict, winner_idx, dim_losers):
    """Render one output frame.

    votes        : running tally list[int] (len K) or None
    active_pair  : (i, j) currently being compared, or None
    verdict      : {idx: 'win'|'lose'} for the active pair, or None
    winner_idx   : highlight this cell green (final), or None
    dim_losers   : dim all non-winner cells
    """
    img = base_canvas()
    tl = TextLayer(img)

    # header: a title box over each panel + the phase narration under the
    # samples title.
    fill_rect(img, LEFT_X0, 8, LEFT_X1, HEADER_H - 8, BAR_BG)
    fill_rect(img, RIGHT_X0, 8, RIGHT_X1, HEADER_H - 8, BAR_BG)
    tl.add(((LEFT_X0 + LEFT_X1) // 2, HEADER_H // 2), "Real Robot", 30, TEXT,
           bold=True, anchor="mm")
    tl.add((RIGHT_X0 + 12, 12), "Generated Samples", 28, TEXT, bold=True)
    tl.add((RIGHT_X0 + 12, 50), phase, 20, phase_color)

    draw_left(img, left_frame, left_label, tl)

    K = len(cand_imgs)
    max_votes = max(votes) if (votes and max(votes) > 0) else 1
    for k in range(K):
        tx, ty, tw, th, cx0, cw = geo[k]
        is_active = active_pair is not None and k in active_pair
        is_winner = winner_idx is not None and k == winner_idx

        thumb = cv2.resize(cand_imgs[k], (tw, th), interpolation=cv2.INTER_AREA)
        if dim_losers and not is_winner:
            thumb = (thumb.astype(np.float32) * 0.4).astype(np.uint8)
        paste(img, thumb, tx, ty)

        bcol, bt = (70, 76, 90), 2
        if is_winner:
            bcol, bt = WIN, 4
        elif is_active:
            bcol, bt = HILITE, 3
        border_rect(img, tx, ty, tx + tw, ty + th, bcol, bt)

        # ---- label + vote bar underneath the video ----
        lab_col = WIN if is_winner else (HILITE if is_active else TEXT)
        uy = ty + th + 5
        tl.add((tx, uy), f"Sample {k}", 21, lab_col, bold=True)

        if verdict is not None and k in verdict:
            v = verdict[k]
            tag = "WIN" if v == "win" else "lose"
            tcol = WIN if v == "win" else LOSE
            tl.add((tx + tw, uy), tag, 20, tcol, bold=True, anchor="ra")

        if votes is not None:
            bx0, bx1 = tx, tx + tw - 36
            by0 = uy + 27
            by1 = by0 + 16
            fill_rect(img, bx0, by0, bx1, by1, BAR_BG)
            frac = votes[k] / max_votes
            fcol = WIN if is_winner else ACCENT
            fill_rect(img, bx0, by0, bx0 + int((bx1 - bx0) * frac), by1, fcol)
            tl.add((bx1 + 6, by0 - 3), str(votes[k]), 19, TEXT, bold=True)

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
def build_pause_states(pairs, votes_final, winner_idx, K, n_gen, n_win,
                       pair_frames, n_out=None, executed=True):
    """Return the list of right-panel state dicts for one pause region.

    The pause is three sub-phases concatenated: generation (`n_gen` frames,
    candidates playing, no votes) -> pairwise ranking (each pair pi shown for
    `pair_frames[pi]` output frames with a growing running tally) -> winner
    reveal (`n_win` frames). The natural length is the sum of those; if `n_out`
    is given and larger, the winner reveal is held to pad up to `n_out`.
    """
    states = []

    # phase 1: generation
    for _ in range(n_gen):
        states.append(dict(
            phase=f"Generating {K} candidate futures from the policy",
            phase_color=ACCENT, votes=None, active_pair=None, verdict=None,
            winner_idx=None, dim_losers=False))

    # phase 2: pairwise ranking with a growing running tally; each pair lingers
    # for its own pair_frames[pi] output frames.
    P = len(pairs)
    if P > 0:
        for pi, pr in enumerate(pairs):
            tally = [0] * K
            for q in range(pi + 1):
                tally[int(pairs[q]["winner"])] += 1
            i, j, w = int(pr["i"]), int(pr["j"]), int(pr["winner"])
            loser = j if w == i else i
            st = dict(
                phase=f"VLM pairwise ranking  ({pi + 1}/{P})   "
                      f"Sample {i} vs Sample {j}  ->  Sample {w} wins",
                phase_color=HILITE, votes=tally, active_pair=(i, j),
                verdict={w: "win", loser: "lose"}, winner_idx=None,
                dim_losers=False)
            for _ in range(max(0, int(pair_frames[pi]))):
                states.append(st)

    # phase 3: winner reveal
    tail = ("executing on robot" if executed
            else "run ended before execution")
    for _ in range(n_win):
        states.append(dict(
            phase=f"VLM chose Sample {winner_idx}  (votes: "
                  f"{votes_final[winner_idx]})  ->  {tail}",
            phase_color=WIN, votes=list(votes_final), active_pair=None,
            verdict=None, winner_idx=winner_idx, dim_losers=True))

    if not states:
        states.append(dict(phase="", phase_color=ACCENT, votes=None,
                           active_pair=None, verdict=None, winner_idx=None,
                           dim_losers=False))
    # hold the last (winner) frame to pad up to n_out when requested
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
    ap.add_argument("--gen_hold", type=float, default=0.4,
                    help="seconds the candidate clips stay frozen on their last "
                         "frame after the play-through before pairwise ranking "
                         "begins.")
    ap.add_argument("--slow_pairs", type=int, default=2,
                    help="number of leading comparisons on the FIRST step that "
                         "get the slow --slow_comparisons duration.")
    ap.add_argument("--slow_comparisons", type=float, default=5.0,
                    help="seconds each of the first --slow_pairs comparisons "
                         "lingers, ON THE FIRST STEP ONLY (default 5s).")
    ap.add_argument("--fast_comparisons", type=float, default=0.35,
                    help="seconds each fast comparison lasts: the remaining "
                         "comparisons on step 1, and ALL comparisons on later "
                         "steps.")
    ap.add_argument("--winner_sec", type=float, default=1.2,
                    help="seconds the winner reveal is shown before execution.")
    ap.add_argument("--grid_cols", type=int, default=3,
                    help="columns in the candidate-video grid, used when "
                         "--grid_rows does not apply (e.g. 10 samples -> 3 "
                         "cols). Fewer columns => bigger, wider videos.")
    ap.add_argument("--grid_rows", default="2,1,2",
                    help="explicit per-row sample counts, comma-separated "
                         "(default '2,1,2' for 5 samples -> big videos). Used "
                         "only when it sums to the number of samples; otherwise "
                         "falls back to --grid_cols. Pass '' to always use cols.")
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
        out_path = os.path.join(run_dir, f"visualization_try_2{suffix}.mp4")

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
    writer = Writer(out_path, fps)
    cap = cv2.VideoCapture(str(rollout_path))
    cur_idx = 0
    total_out = 0
    rows_arr = None
    if args.grid_rows.strip():
        try:
            rows_arr = [int(x) for x in args.grid_rows.split(",") if x.strip()]
        except ValueError:
            print(f"[warn] could not parse --grid_rows '{args.grid_rows}'; "
                  f"using --grid_cols {args.grid_cols}")
    K0 = len(glob.glob(os.path.join(steps[0], "*.mp4")))
    geo = grid_geometry(max(1, K0), args.grid_cols, rows_arr)

    for si, step_dir in enumerate(steps):
        ranking, cand = load_step(step_dir)
        K = len(cand)
        geo = grid_geometry(K, args.grid_cols, rows_arr)
        votes_final = ranking.get("votes", [0] * K)
        winner_idx = int(ranking.get("winner_idx", int(np.argmax(votes_final))))
        pairs = ranking.get("pairs", [])
        ps, es, ee, has_exec = segs[si]

        # ---------- PAUSE region: left sped up, right = gen/rank/winner ----------
        pause_len = es - ps
        # generation sub-phase lasts one full 0.5x play-through of the longest
        # candidate clip (+ a short frozen hold) so ranking only starts once
        # every clip is paused on its last frame.
        max_clip = max((len(c) for c in cand), default=1)
        play_once = int(np.ceil((max_clip - 1) * fps
                                / (CAND_SRC_FPS * max(1e-6, args.cand_speed))))
        n_gen = play_once + int(round(args.gen_hold * fps))
        # explicit per-comparison durations. On the FIRST step the first
        # --slow_pairs comparisons each linger for --slow_comparisons seconds;
        # every other comparison (and all comparisons on later steps) lasts
        # --fast_comparisons seconds.
        slow_f = max(1, int(round(args.slow_comparisons * fps)))
        fast_f = max(1, int(round(args.fast_comparisons * fps)))
        P = len(pairs)
        if si == 0:
            pair_frames = [slow_f if pi < args.slow_pairs else fast_f
                           for pi in range(P)]
        else:
            pair_frames = [fast_f] * P
        n_win = max(1, int(round(args.winner_sec * fps)))
        # narration length drives the pause; if the real pause is longer than
        # 10x of it, we'd be too fast, so take whichever is longer and sample
        # the (static) robot frames to fit.
        narration = n_gen + sum(pair_frames) + n_win
        n_out = max(narration,
                    int(np.ceil(pause_len / max(1, args.pause_speedup))))
        states = build_pause_states(pairs, votes_final, winner_idx, K, n_gen,
                                    n_win, pair_frames, n_out=n_out,
                                    executed=has_exec)
        n_out = len(states)
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
        for t in range(n_out):
            st = states[t]
            lf = left_frames[min(t, len(left_frames) - 1)]
            frame = compose(
                lf, f"Generating + ranking — {eff_speed}x speed up",
                cand_frame(cand, K, t, fps, speed=args.cand_speed), geo,
                st["phase"], st["phase_color"], st["votes"], st["active_pair"],
                st["verdict"], st["winner_idx"], st["dim_losers"])
            writer.write(frame)
            total_out += 1

        # ---------- EXEC region: left at 1x, right = winner executing ----------
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
                WIN, list(votes_final), None, None, winner_idx, True)
            writer.write(frame)
            total_out += 1

        del cand
        print(f"[step {si}] done. cumulative out frames={total_out} "
              f"(~{total_out / fps:.1f}s)")

    cap.release()
    writer.close()
    print(f"[done] wrote {out_path}  "
          f"({total_out} frames, ~{total_out / fps:.1f}s @ {fps:g}fps)")


if __name__ == "__main__":
    sys.exit(main())
