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
# Right region: vertically stacked candidate rows + ranking
RIGHT_X0 = 762
RIGHT_X1 = W - PAD


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


def row_geometry(n_rows):
    """Per-row (y0, thumb_x, thumb_y, thumb_w, thumb_h, text_x)."""
    top = HEADER_H + PAD
    avail = H - top - PAD
    row_h = avail / n_rows
    thumb_h = int(min(row_h - 8, (RIGHT_X1 - RIGHT_X0) * 0.20 * 480 / 848))
    thumb_h = max(40, thumb_h)
    thumb_w = int(round(thumb_h * 848.0 / 480.0))
    thumb_x = RIGHT_X0
    text_x = thumb_x + thumb_w + 16
    geo = []
    for i in range(n_rows):
        y0 = int(top + i * row_h)
        ty = int(y0 + (row_h - thumb_h) / 2)
        geo.append((y0, thumb_x, ty, thumb_w, thumb_h, text_x))
    return geo, row_h


def draw_left(img, left_frame, label, tl):
    box_w = LEFT_X1 - LEFT_X0
    box_h = H - (HEADER_H + PAD) - PAD - 30
    y0 = HEADER_H + PAD
    fill_rect(img, LEFT_X0, y0, LEFT_X1, y0 + box_h, PANEL_BG)
    r, ox, oy = fit_into(left_frame, box_w, box_h)
    paste(img, r, LEFT_X0 + ox, y0 + oy)
    border_rect(img, LEFT_X0, y0, LEFT_X1, y0 + box_h, (70, 76, 90), 2)
    tl.add((LEFT_X0 + 6, y0 + box_h + 4), label, 22, SUBTLE, bold=True)


def compose(left_frame, left_label, cand_imgs, geo, row_h, phase, phase_color,
            votes, active_pair, verdict, winner_idx, dim_losers):
    """Render one output frame.

    votes        : running tally list[int] (len K) or None
    active_pair  : (i, j) currently being compared, or None
    verdict      : {idx: 'win'|'lose'} for the active pair, or None
    winner_idx   : highlight this row green (final), or None
    dim_losers   : dim all non-winner rows
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
    bar_x1 = RIGHT_X1 - 64
    for k in range(K):
        y0, tx, ty, tw, th, txt_x = geo[k]
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

        lab_col = WIN if is_winner else (HILITE if is_active else TEXT)
        tl.add((txt_x, ty + 1), f"Sample {k}", 22, lab_col, bold=True)

        if votes is not None:
            bx0 = txt_x
            by0 = ty + 30
            by1 = min(ty + th, by0 + 22)
            fill_rect(img, bx0, by0, bar_x1, by1, BAR_BG)
            frac = votes[k] / max_votes
            fcol = WIN if is_winner else ACCENT
            fill_rect(img, bx0, by0, bx0 + int((bar_x1 - bx0) * frac), by1, fcol)
            tl.add((bar_x1 + 8, by0 - 2), str(votes[k]), 20, TEXT, bold=True)

        if verdict is not None and k in verdict:
            v = verdict[k]
            tag = "WIN" if v == "win" else "lose"
            tcol = WIN if v == "win" else LOSE
            tl.add((RIGHT_X1 - 6, ty + 1), tag, 20, tcol, bold=True, anchor="ra")

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
def build_pause_states(n_out, pairs, votes_final, winner_idx, K,
                       gen_frac=0.22, rank_frac=0.62, executed=True):
    """Return a list of `n_out` right-panel state dicts for one pause region.

    The pause region is split into three sub-phases whose lengths sum to
    n_out: generation (candidates play, no votes) -> pairwise ranking (pairs
    revealed one-by-one, running tally grows) -> winner reveal. Each output
    frame gets a dict telling `compose` what to draw.
    """
    n_out = max(1, n_out)
    n_gen = max(1, int(round(n_out * gen_frac)))
    n_rank = max(1, int(round(n_out * rank_frac)))
    n_gen = min(n_gen, n_out - 1)
    n_rank = min(n_rank, n_out - n_gen)
    n_win = max(0, n_out - n_gen - n_rank)

    states = []

    # phase 1: generation
    for _ in range(n_gen):
        states.append(dict(
            phase=f"Generating {K} candidate futures from the policy",
            phase_color=ACCENT, votes=None, active_pair=None, verdict=None,
            winner_idx=None, dim_losers=False))

    # phase 2: pairwise ranking with running tally
    P = len(pairs)
    running = [0] * K
    if P > 0:
        # assign each output frame to a pair index (monotonic)
        for t in range(n_rank):
            pi = min(P - 1, int(t * P / n_rank))
            # running tally reflects all pairs up to and including pi
            tally = [0] * K
            for q in range(pi + 1):
                tally[int(pairs[q]["winner"])] += 1
            pr = pairs[pi]
            i, j, w = int(pr["i"]), int(pr["j"]), int(pr["winner"])
            loser = j if w == i else i
            states.append(dict(
                phase=f"VLM pairwise ranking  ({pi + 1}/{P})   "
                      f"Sample {i} vs Sample {j}  ->  Sample {w} wins",
                phase_color=HILITE, votes=tally, active_pair=(i, j),
                verdict={w: "win", loser: "lose"}, winner_idx=None,
                dim_losers=False))
        running = [0] * K
        for pr in pairs:
            running[int(pr["winner"])] += 1
    else:
        for _ in range(n_rank):
            states.append(dict(phase="VLM ranking", phase_color=HILITE,
                               votes=list(votes_final), active_pair=None,
                               verdict=None, winner_idx=None, dim_losers=False))
        running = list(votes_final)

    # phase 3: winner reveal
    tail = ("executing on robot" if executed
            else "run ended before execution")
    for _ in range(n_win):
        states.append(dict(
            phase=f"VLM chose Sample {winner_idx}  (votes: "
                  f"{votes_final[winner_idx]})  ->  {tail}",
            phase_color=WIN, votes=list(votes_final), active_pair=None,
            verdict=None, winner_idx=winner_idx, dim_losers=True))

    # pad/truncate to exactly n_out
    while len(states) < n_out:
        states.append(states[-1] if states else dict(
            phase="", phase_color=ACCENT, votes=None, active_pair=None,
            verdict=None, winner_idx=None, dim_losers=False))
    return states[:n_out]


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def cand_frame(cand, K, t_out, out_fps, speed=0.5, src_fps=24.0):
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
    ap.add_argument("--exec_pad_pre", type=float, default=0.25,
                    help="seconds of real-time lead-in before each detected "
                         "execution burst")
    ap.add_argument("--exec_pad_post", type=float, default=0.6,
                    help="seconds of real-time settle shown after each burst")
    ap.add_argument("--gen_frac", type=float, default=0.30,
                    help="fraction of each sped-up pause spent on the "
                         "'generating candidates' sub-phase")
    ap.add_argument("--rank_frac", type=float, default=0.65,
                    help="fraction of each sped-up pause spent revealing the "
                         "pairwise rankings")
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
    K0 = len(glob.glob(os.path.join(steps[0], "*.mp4")))
    geo, row_h = row_geometry(max(1, K0))

    for si, step_dir in enumerate(steps):
        ranking, cand = load_step(step_dir)
        K = len(cand)
        geo, row_h = row_geometry(K)
        votes_final = ranking.get("votes", [0] * K)
        winner_idx = int(ranking.get("winner_idx", int(np.argmax(votes_final))))
        pairs = ranking.get("pairs", [])
        ps, es, ee, has_exec = segs[si]

        # ---------- PAUSE region: left sped up, right = gen/rank/winner ----------
        pause_len = es - ps
        n_out = max(1, int(np.ceil(pause_len / max(1, args.pause_speedup))))
        states = build_pause_states(n_out, pairs, votes_final, winner_idx, K,
                                    gen_frac=args.gen_frac,
                                    rank_frac=args.rank_frac, executed=has_exec)
        # collect the kept (every-Nth) source frames for the left panel
        left_frames = []
        kept_target = set(ps + k * args.pause_speedup for k in range(n_out))
        while cur_idx <= es - 1:
            ok, f = cap.read()
            if not ok:
                break
            idx = cur_idx
            cur_idx += 1
            if idx in kept_target:
                left_frames.append(cv2.cvtColor(cv2.resize(f, (640, 360)),
                                                cv2.COLOR_BGR2RGB))
        if not left_frames:
            left_frames = [np.full((360, 640, 3), 40, np.uint8)]
        for t in range(n_out):
            st = states[t]
            lf = left_frames[min(t, len(left_frames) - 1)]
            frame = compose(
                lf, f"Generating + ranking — {args.pause_speedup}x speed up",
                cand_frame(cand, K, t, fps, speed=args.cand_speed), geo, row_h,
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
                win_cand_last, geo, row_h,
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
