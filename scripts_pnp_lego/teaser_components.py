#!/usr/bin/env python
"""Render the individual *components* of the full-pipeline teaser animation as
separate files, so they can be assembled into slides by hand.

This does NOT change the existing explainer scripts. For one VLM run dir it
emits, per decision step, the pieces of the story:

    State  ->  Diffusion Policy  ->  YAM-Pro action cloud  ->  Videos  ->  pick

Concretely, the pipeline behind each step is:
  * the diffusion policy samples ``actions_full_pre`` = (N_sampled, T, 7)
    JOINT-space trajectories (6 arm joints + 1 gripper),
  * ``keep_indices`` selects K of them; those K condition the K video-gen clips
    (candidate k.mp4 <- kept trajectory k),
  * a VLM ranks the K candidate videos and picks a winner (``ranking.json``),
  * the chosen action is executed on the real robot (``rollout.mp4``).

The "action cloud" is rendered as the *actual* I2RT YAM Pro arm (its MuJoCo
model lives in scripts_pnp_lego/assets/yam_pro): the K kept joint trajectories
are played forward together as K color-coded arms on a white background with a
fixed camera (no orbit), winner bold and the others as faint ghosts. Because the
7-d actions are joint angles, this is pure forward kinematics -- qpos = a[:6].

Outputs (under ``<run_dir>/teaser_components/`` by default):

  states/
    step{i}_state.png          input observation for the step
    step{i}_state_task.png     same, with the task name captioned
    step{i}_exec.mp4           the rollout execution segment for the step (1x)
  robot/
    step{i}_robot.mp4          the K kept joint-trajectories played as K
                               color-coded YAM-Pro arms (winner bold), white bg
    step{i}_cloud.png          RGBA still of all K arms spread (transparent bg)
    step{i}_selected.png       RGBA still of just the winner arm
  videos/
    step{i}/cand{k}.mp4        candidate clip k (plain, no outline)
    step{i}/cand{k}_winner.mp4 the winning candidate (plain copy)
    step{i}/grid.mp4           all K candidates playing together (no outlines)
  manifest.json                per-step mapping: task, votes, winner, keep_indices,
                               candidate<->color key, and every emitted file path.

The K candidate colors (manifest ``colors_rgb``) are reused for the arms so the
cloud-to-video correspondence is legible.

Usage:
python scripts_pnp_lego/teaser_components.py \
    --run_dir /proj/.../21_ss_2026-05-11_09-24-50_VLM
"""
import argparse
import glob
import json
import os
import shutil
import sys

# MuJoCo must pick a headless GL backend before it is imported. OSMesa renders
# purely on CPU (no display / EGL device needed) which is the most portable on
# a cluster node; override with MUJOCO_GL=egl if a GPU context is available.
os.environ.setdefault("MUJOCO_GL", "osmesa")

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
    "epoch=0200-train_loss=0.0116/21_ss_2026-05-11_09-24-50_VLM"
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAM_XML = os.path.join(_HERE, "assets", "yam_pro", "yam_pro.xml")

# Candidate palette (RGB). Index k -> kept trajectory k -> candidate video k ->
# arm color k. Muted, paper-figure palette (seaborn "deep") -- harmonious and
# desaturated rather than neon, reads professionally over the white background.
COLORS = [
    (76, 114, 176),    # blue
    (221, 132, 82),    # orange
    (85, 168, 104),    # green
    (129, 114, 179),   # purple
    (147, 120, 96),    # taupe
    (218, 139, 195),   # mauve
    (140, 140, 140),   # gray
    (204, 185, 116),   # sand
]
WHITE = (255, 255, 255)
BG = (16, 18, 24)
CAND_SRC_FPS = 24.0   # HunyuanVideo writes the candidate clips at 24fps

_FONT_FILES = {
    "regular": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf"],
    "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"],
}
_FONT_CACHE = {}


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


def draw_text(img, xy, s, size, color=(235, 238, 244), bold=True, anchor="la"):
    if not _HAVE_PIL:
        cv2.putText(img, s, (int(xy[0]), int(xy[1]) + size),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color, 1, cv2.LINE_AA)
        return
    pim = Image.fromarray(img)
    d = ImageDraw.Draw(pim)
    d.text(xy, s, font=get_font(size, bold), fill=tuple(color), anchor=anchor)
    img[:] = np.asarray(pim)


# ---------------------------------------------------------------------------
# Output writer (imageio -> libx264, cv2 fallback)
# ---------------------------------------------------------------------------
class Writer:
    def __init__(self, path, fps, size):
        self.path, self.fps = str(path), fps
        try:
            import imageio
            self.w = imageio.get_writer(self.path, fps=fps, codec="libx264",
                                        quality=8, macro_block_size=None)
            self.kind = "imageio"
        except Exception as e:
            print(f"[writer] imageio unavailable ({e}); using cv2.VideoWriter")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.w = cv2.VideoWriter(self.path, fourcc, fps, size)
            self.kind = "cv2"

    def write(self, rgb):
        if self.kind == "imageio":
            self.w.append_data(rgb)
        else:
            self.w.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        self.w.close() if self.kind == "imageio" else self.w.release()


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


# ---------------------------------------------------------------------------
# Execution-burst detection (mirrors the explainer script so the per-step state
# clips line up with the per-step generations).
# ---------------------------------------------------------------------------
def robust_motion_signal(rollout_path, work_w=160, work_h=90, pix_thr=18):
    cap = cv2.VideoCapture(str(rollout_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, sig = None, []
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


def _smooth(x, k):
    k = max(1, int(k))
    return x.astype(float) if k <= 1 else np.convolve(x, np.ones(k) / k, "same")


def find_execution_bursts(sig, fps, n_expected, min_dur=0.25, merge_gap=0.8):
    sm = _smooth(sig, int(0.5 * fps))
    base = float(np.percentile(sm, 50))
    hi = float(np.percentile(sm, 98))
    thr = base + 0.15 * max(1e-9, hi - base)
    mask = sm > thr
    n = len(mask)
    runs, i = [], 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged, gap = [], int(merge_gap * fps)
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
# YAM-Pro robot renderer (MuJoCo, headless). One renderer instance, reused for
# every step. Each kept joint-trajectory is rendered as one color-tinted arm and
# alpha-composited onto a white canvas using MuJoCo's segmentation mask.
# ---------------------------------------------------------------------------
class YamRenderer:
    def __init__(self, xml_path, W, H, cam_lookat, cam_dist, cam_az, cam_el,
                 light_ambient=0.5, light_diffuse=0.7, light_specular=0.15):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        # the model has no <light>, so brighten the headlight to light the arm
        self.m.vis.headlight.ambient[:] = light_ambient
        self.m.vis.headlight.diffuse[:] = light_diffuse
        self.m.vis.headlight.specular[:] = light_specular
        self.W, self.H = W, H
        self.r = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = cam_lookat
        self.cam.distance = cam_dist
        self.cam.azimuth = cam_az
        self.cam.elevation = cam_el
        self._base_rgba = self.m.geom_rgba.copy()

    def render(self, action, color, finger_stroke=0.0475):
        """Render the arm at `action` (>=6: 6 joint angles, optional 7th =
        gripper open-fraction in [0,1]), every geom flat-tinted to `color`.
        Returns (rgb uint8 HxWx3, mask bool HxW)."""
        self.d.qpos[:6] = action[:6]
        # finger slide joints (joint7/joint8) from the gripper open-fraction
        if self.m.nq >= 8 and len(action) >= 7:
            f = float(np.clip(action[6], 0.0, 1.0)) * finger_stroke
            self.d.qpos[6] = f
            self.d.qpos[7] = f
        self.mj.mj_forward(self.m, self.d)
        c = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0)
        for g in range(self.m.ngeom):
            self.m.geom_rgba[g] = c
        self.r.update_scene(self.d, self.cam)
        rgb = self.r.render().copy()
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.d, self.cam)
        seg = self.r.render()[:, :, 0].copy()
        self.r.disable_segmentation_rendering()
        return rgb, (seg >= 0)

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass


def composite_arms(renderer, configs, colors, alphas, bg=WHITE):
    """Render each (qpos6, color, alpha) arm and alpha-composite over `bg`.
    Arms are drawn in the given order (last on top). Returns (rgb uint8,
    alpha float HxW in [0,1] = union coverage)."""
    H, W = renderer.H, renderer.W
    canvas = np.full((H, W, 3), bg, np.float32)
    cover = np.zeros((H, W), np.float32)
    for q, col, a in zip(configs, colors, alphas):
        if a <= 0:
            continue
        rgb, mask = renderer.render(q, col)
        am = (mask.astype(np.float32) * a)[..., None]
        canvas = canvas * (1 - am) + rgb.astype(np.float32) * am
        cover = np.maximum(cover, mask.astype(np.float32) * a)
    return np.clip(canvas, 0, 255).astype(np.uint8), cover


def save_rgba(rgb, cover, path):
    """Save with `cover` (0..1) as the alpha channel -> transparent background
    for slide overlay."""
    if not _HAVE_PIL:
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return
    out = np.dstack([rgb, np.clip(cover * 255, 0, 255).astype(np.uint8)])
    Image.fromarray(out, "RGBA").save(path)


def _finish(rgb, cover, out_w, out_h, flip_v, flip_h):
    """Downsample a supersampled (rgb, cover) pair to the output size -- this is
    what anti-aliases the silhouette: MuJoCo's segmentation mask is a hard 0/1
    edge, so we render large and INTER_AREA-shrink so edges become smooth -- then
    optionally flip vertically and/or horizontally."""
    if (rgb.shape[1], rgb.shape[0]) != (out_w, out_h):
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        cover = cv2.resize(cover, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if flip_v:
        rgb = rgb[::-1]; cover = cover[::-1]
    if flip_h:
        rgb = rgb[:, ::-1]; cover = cover[:, ::-1]
    return np.ascontiguousarray(rgb), np.ascontiguousarray(cover)


def render_robot(pre, keep, winner_kept, out_path, png_dir, step_tag,
                 renderer, fps, args):
    """Play the K kept joint-trajectories forward together as K color-coded
    YAM-Pro arms (winner bold, others faint), then fade the non-winners so only
    the chosen arm remains. Writes the mp4 plus RGBA stills (cloud / selected).
    """
    keep = list(map(int, keep))
    K = len(keep)
    T = pre.shape[1]
    colors = [COLORS[k % len(COLORS)] for k in range(K)]
    win = winner_kept if (winner_kept is not None and winner_kept < K) else None
    out_w, out_h = args.robot_w, args.robot_h
    flip_v, flip_h = args.flip_vertical, not args.no_flip_horizontal

    f_move = max(1, int(round(args.robot_move_sec * fps)))
    f_hold = max(0, int(round(args.robot_hold_sec * fps)))
    f_sel = max(0, int(round(args.robot_select_sec * fps))) if win is not None else 0
    # every arm is drawn at the SAME opacity -- the executed action is not
    # rendered any darker/bolder than the rest.
    uni = args.arm_alpha

    # draw order: ghosts first, winner last (on top). If no winner, natural order.
    order = [k for k in range(K) if k != win] + ([win] if win is not None else [])

    writer = Writer(out_path, fps, (out_w, out_h))
    cloud_still = sel_still = None

    def frame_at(ti, sel_u):
        """sel_u in [0,1]: 0 = all K shown (ghosts at `ghost`), 1 = only winner.
        Returns the finished (downsampled + flipped) (rgb, cover)."""
        configs, cols, alphas = [], [], []
        for k in order:
            q = pre[keep[k], ti]          # full 7-d (6 joints + gripper)
            # uniform opacity for every arm; during the (optional) select phase
            # the non-chosen arms fade OUT (they vanish) but the chosen one is
            # never drawn darker than the others were.
            a = uni if k == win else uni * (1.0 - sel_u)
            configs.append(q); cols.append(colors[k]); alphas.append(a)
        rgb, cover = composite_arms(renderer, configs, cols, alphas)
        return _finish(rgb, cover, out_w, out_h, flip_v, flip_h)

    # phase 1: all arms play their trajectories forward together
    for fi in range(f_move):
        ti = min(T - 1, int(round((fi / max(1, f_move - 1)) * (T - 1))))
        rgb, cover = frame_at(ti, 0.0)
        writer.write(rgb)
        if cloud_still is None and fi == f_move - 1:
            cloud_still = (rgb, cover)
    # phase 2: hold the full spread
    for _ in range(f_hold):
        writer.write(rgb)
    # phase 3: fade non-winners out, leaving the chosen arm
    for fi in range(f_sel):
        u = (fi + 1) / max(1, f_sel)
        rgb, cover = frame_at(T - 1, u)
        writer.write(rgb)
    if win is not None:
        sel_rgb, sel_cover = frame_at(T - 1, 1.0)
        # brief hold on the lone winner
        for _ in range(max(1, int(round(0.4 * fps)))):
            writer.write(sel_rgb)
        sel_still = (sel_rgb, sel_cover)
    writer.close()

    if cloud_still is None:
        cloud_still = frame_at(T - 1, 0.0)
    if sel_still is None:
        sel_still = frame_at(T - 1, 1.0 if win is not None else 0.0)
    save_rgba(cloud_still[0], cloud_still[1],
              os.path.join(png_dir, f"{step_tag}_cloud.png"))
    save_rgba(sel_still[0], sel_still[1],
              os.path.join(png_dir, f"{step_tag}_selected.png"))
    return (f_move + f_hold + f_sel) / fps


# ---------------------------------------------------------------------------
# Candidate-video components: each clip opens with a diffusion denoise (random
# noise -> the clip's first frame) so it visibly "diffuses" into the video, then
# plays. No outlines.
# ---------------------------------------------------------------------------
def denoise_intro(first_frame, n_frames, n_steps, seed):
    """`n_frames` frames crossfading from pure static toward `first_frame`. The
    noise is re-sampled every diffusion step (sigma = 1 - (s+1)/n_steps) so the
    static visibly jitters like iterative sampling; the last frame is clean and
    hands off seamlessly to clip playback."""
    out = []
    tgt = first_frame.astype(np.float32)
    n_steps = max(1, n_steps)
    for i in range(n_frames):
        s = min(n_steps - 1, int(i / max(1, n_frames) * n_steps))
        sigma = 1.0 - (s + 1) / n_steps
        rng = np.random.default_rng(seed + s)
        noise = rng.integers(0, 256, tgt.shape, dtype=np.uint8).astype(np.float32)
        mix = (1.0 - sigma) * tgt + sigma * noise
        out.append(np.clip(mix, 0, 255).astype(np.uint8))
    return out


WIN_GREEN = (46, 204, 113)


def draw_border(img, color, t):
    """Draw a `t`-px border of `color` inset so it sits fully inside the frame."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (t // 2, t // 2), (w - 1 - t // 2, h - 1 - t // 2),
                  tuple(int(c) for c in color), t)


def draw_corner_label(img, text, size=26, pad=9, margin=12):
    """Small label on a darkened plate in the BOTTOM-LEFT corner (readable over
    video)."""
    h, w = img.shape[:2]
    tw = int(len(text) * size * 0.6) + 2 * pad
    th = size + 2 * pad
    x0, y1 = margin, h - margin
    x1, y0 = min(w, x0 + tw), y1 - th
    roi = img[max(0, y0):y1, x0:x1].astype(np.float32) * 0.35
    img[max(0, y0):y1, x0:x1] = roi.astype(np.uint8)
    draw_text(img, (x0 + pad, y0 + pad), text, size, (255, 255, 255), bold=True)


def winner_blink_tail(last_frame, fps, args):
    """Frames appended to the WINNER clip: freeze on the last frame and flash a
    green border `winner_blink_count` times, then hold a solid green border to
    mark it as the chosen video."""
    out = []
    bt = max(14, last_frame.shape[1] // 28)
    period = max(2, int(round(fps / max(0.1, args.winner_blink_hz))))
    half = max(1, period // 2)
    for _ in range(max(0, args.winner_blink_count)):
        for _ in range(half):                       # border ON
            f = last_frame.copy(); draw_border(f, WIN_GREEN, bt); out.append(f)
        for _ in range(half):                       # border OFF
            out.append(last_frame.copy())
    for _ in range(max(0, int(round(args.winner_hold_sec * fps)))):
        f = last_frame.copy(); draw_border(f, WIN_GREEN, bt); out.append(f)
    return out


def write_candidate(src, dst, args, seed, index, winner=False):
    """Write a candidate clip: diffusion-denoise intro (noise -> first frame)
    followed by the real video, at the clip's native fps. A small "Action N"
    label (N = index, 1-based) sits in the bottom-left corner throughout. If
    `winner`, append a blinking green border at the end to mark it chosen."""
    frames = read_all_frames(src)
    if not frames:
        return False
    cap = cv2.VideoCapture(src)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or CAND_SRC_FPS
    cap.release()
    n_dn = max(0, int(round(args.cand_denoise_sec * fps_src)))
    intro = (denoise_intro(frames[0], n_dn, args.cand_denoise_steps, seed)
             if n_dn > 0 else [])
    seq = intro + frames
    if winner:
        seq = seq + winner_blink_tail(frames[-1], fps_src, args)
    label = f"Action {index}"
    lbl_size = max(16, frames[0].shape[1] // 32)
    h, w = frames[0].shape[:2]
    wr = Writer(dst, fps_src, (w, h))
    for f in seq:
        draw_corner_label(f, label, size=lbl_size)
        wr.write(f)
    wr.close()
    return True


def grid_clip(srcs, dst, winner_idx, fps_out, args, gap=12):
    """Tile the K candidate clips side by side on a neutral background with NO
    per-cell outlines. The winner is indicated only by a small text tag."""
    clips = [read_all_frames(s) for s in srcs]
    clips = [c for c in clips if c]
    if not clips:
        return False
    K = len(clips)
    n = max(len(c) for c in clips)
    h, w = clips[0][0].shape[:2]
    sc = min(1.0, (1900 / K) / w)
    tw, th = int(w * sc), int(h * sc)
    cell_w, cell_h = tw, th
    W2 = K * cell_w + (K + 1) * gap
    H2 = cell_h + 2 * gap
    W2 += W2 % 2; H2 += H2 % 2          # libx264 needs even dims
    # all candidates diffuse together first, then play
    n_dn = max(0, int(round(args.cand_denoise_sec * fps_out)))
    intros = [denoise_intro(c[0], n_dn, args.cand_denoise_steps,
                            args.denoise_seed + k) for k, c in enumerate(clips)]
    wr = Writer(dst, fps_out, (W2, H2))
    for fi in range(n_dn + n):
        canvas = np.empty((H2, W2, 3), np.uint8)
        canvas[:] = BG
        for k in range(K):
            if fi < n_dn:
                f = intros[k][fi]
            else:
                f = clips[k][min(fi - n_dn, len(clips[k]) - 1)]
            f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
            x0 = gap + k * (cell_w + gap)
            y0 = gap
            canvas[y0:y0 + th, x0:x0 + tw] = f
            lbl = f"Action {k + 1}" + ("  ✓ SELECTED" if k == winner_idx else "")
            ly = y0 + th - 34                       # bottom-left of the cell
            plate = canvas[ly:ly + 30, x0:x0 + tw].astype(np.float32) * 0.35
            canvas[ly:ly + 30, x0:x0 + tw] = plate.astype(np.uint8)
            draw_text(canvas, (x0 + 8, ly + 4), lbl, 22,
                      WHITE if k == winner_idx else (235, 238, 244), bold=True)
        wr.write(canvas)
    wr.close()
    return True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover(run_dir):
    pre = sorted(glob.glob(os.path.join(run_dir, "actions_full_pre_*.npy")))
    vg = sorted(d for d in glob.glob(os.path.join(run_dir, "videogen", "*"))
                if os.path.isdir(d) and
                os.path.exists(os.path.join(d, "ranking.json")))
    obs = sorted(glob.glob(os.path.join(run_dir, "observations", "frame_*.jpg")))
    return pre, vg, obs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out_dir", default=None,
                    help="output dir (default <run_dir>/teaser_components)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max_steps", type=int, default=None)
    # which components to emit
    ap.add_argument("--no_robot", action="store_true")
    ap.add_argument("--no_videos", action="store_true")
    ap.add_argument("--no_states", action="store_true")
    # YAM-Pro renderer
    ap.add_argument("--yam_xml", default=DEFAULT_YAM_XML,
                    help="path to the YAM-Pro MuJoCo xml")
    ap.add_argument("--robot_w", type=int, default=900,
                    help="output frame width (px)")
    ap.add_argument("--robot_h", type=int, default=900,
                    help="output frame height (px)")
    ap.add_argument("--supersample", type=int, default=2,
                    help="render at this multiple of the output size then shrink "
                         "(anti-aliases the arm edges; 1 disables)")
    ap.add_argument("--flip_vertical", action="store_true",
                    help="flip the robot rendering top-to-bottom (default off)")
    ap.add_argument("--no_flip_horizontal", action="store_true",
                    help="do NOT flip the robot rendering left-to-right (the "
                         "default flips it horizontally)")
    ap.add_argument("--cam_lookat", default="0.17,0.0,0.29",
                    help="camera look-at point 'x,y,z' (meters)")
    ap.add_argument("--cam_dist", type=float, default=1.5)
    ap.add_argument("--cam_az", type=float, default=96.0,
                    help="azimuth: ~96 turns the base a few degrees past 90 "
                         "counter-clockwise (back edge slightly right of the "
                         "front edge); 90 = edge-on/perpendicular")
    ap.add_argument("--cam_el", type=float, default=-13.0)
    ap.add_argument("--arm_alpha", type=float, default=0.6,
                    help="opacity used for EVERY kept arm (all actions drawn the "
                         "same; the executed one is not darker than the rest)")
    # robot timeline (seconds)
    ap.add_argument("--robot_move_sec", type=float, default=2.6,
                    help="seconds to play the kept trajectories forward")
    ap.add_argument("--robot_hold_sec", type=float, default=0.5,
                    help="seconds to hold on the full spread")
    ap.add_argument("--robot_select_sec", type=float, default=1.2,
                    help="seconds to fade the non-winners out")
    # candidate-video diffusion-denoise intro
    ap.add_argument("--cand_denoise_sec", type=float, default=1.5,
                    help="seconds of diffusion denoise (noise -> first frame) "
                         "prepended to each candidate clip. 0 disables it.")
    ap.add_argument("--cand_denoise_steps", type=int, default=40,
                    help="number of discrete diffusion steps in the denoise "
                         "intro (noise re-sampled each step). Higher = smoother.")
    ap.add_argument("--denoise_seed", type=int, default=0,
                    help="RNG seed for the candidate denoise static")
    # winner clip: blinking green border at the end
    ap.add_argument("--winner_blink_count", type=int, default=3,
                    help="times the green border flashes at the end of the "
                         "winner clip to mark it chosen")
    ap.add_argument("--winner_blink_hz", type=float, default=2.0,
                    help="blink rate (Hz) of the winner green border")
    ap.add_argument("--winner_hold_sec", type=float, default=0.7,
                    help="seconds to hold a solid green border after the blinks")
    # state clip padding
    ap.add_argument("--exec_pad_pre", type=float, default=0.25)
    ap.add_argument("--exec_pad_post", type=float, default=0.6)
    ap.add_argument("--camera", default="image2",
                    help="rollout_<camera>.mp4 for two-camera runs")
    ap.add_argument("--rollout", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "teaser_components")
    pre_files, vg_dirs, obs_files = discover(run_dir)
    assert pre_files, f"no actions_full_pre_*.npy in {run_dir}"
    n_steps = len(pre_files)
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    print(f"[info] run_dir   : {run_dir}")
    print(f"[info] out_dir   : {out_dir}")
    print(f"[info] {len(pre_files)} sampled-action steps, {len(vg_dirs)} "
          f"videogen steps, {len(obs_files)} observation frames"
          + (f"  (rendering {n_steps})" if n_steps != len(pre_files) else ""))

    os.makedirs(out_dir, exist_ok=True)
    robot_dir = os.path.join(out_dir, "robot")
    vids_dir = os.path.join(out_dir, "videos")
    st_dir = os.path.join(out_dir, "states")
    for d in (robot_dir, vids_dir, st_dir):
        os.makedirs(d, exist_ok=True)

    # ---- YAM-Pro renderer (built once, reused for every step) ----
    renderer = None
    if not args.no_robot:
        assert os.path.exists(args.yam_xml), (
            f"YAM-Pro model not found: {args.yam_xml}\n"
            "download i2rt/robot_models/arm/yam_pro from i2rt-robotics/i2rt")
        lookat = [float(x) for x in args.cam_lookat.split(",")]
        ss = max(1, args.supersample)
        rw, rh = args.robot_w * ss, args.robot_h * ss
        print(f"[info] MuJoCo backend: {os.environ.get('MUJOCO_GL')}  "
              f"out {args.robot_w}x{args.robot_h}, render {rw}x{rh} (ss={ss}), "
              f"flip v={args.flip_vertical} h={not args.no_flip_horizontal}, "
              f"az={args.cam_az} el={args.cam_el}")
        renderer = YamRenderer(args.yam_xml, rw, rh,
                               lookat, args.cam_dist, args.cam_az, args.cam_el)

    # ---- rollout + execution segmentation (for the per-step state clips) ----
    rollout_path = args.rollout
    if not rollout_path:
        cands = [os.path.join(run_dir, f"rollout_{args.camera}.mp4"),
                 os.path.join(run_dir, "rollout.mp4")]
        rollout_path = next((c for c in cands if os.path.exists(c)), cands[-1])
    have_rollout = os.path.exists(rollout_path)
    segs, fps_in, nframes = [], 30.0, 0
    if have_rollout and not args.no_states:
        print("[info] scanning rollout for execution bursts ...")
        sig, fps_in, nframes = robust_motion_signal(rollout_path)
        bursts = find_execution_bursts(sig, fps_in, n_expected=len(pre_files))
        pad_pre = int(args.exec_pad_pre * fps_in)
        pad_post = int(args.exec_pad_post * fps_in)
        prev_end = -1
        for i, (bs, be) in enumerate(bursts):
            es = max(prev_end + 1, bs - pad_pre)
            ee = min(be + pad_post, nframes - 1)
            if i + 1 < len(bursts):
                ee = min(ee, bursts[i + 1][0] - 1)
            segs.append((es, ee))
            prev_end = ee
        print(f"[info] detected {len(segs)} execution burst(s)")

    manifest = {
        "run_dir": run_dir, "rollout": rollout_path if have_rollout else None,
        "yam_xml": args.yam_xml, "colors_rgb": COLORS, "steps": [],
    }
    # merge with any prior manifest so a partial run (e.g. --no_robot) keeps the
    # file paths emitted by earlier runs instead of dropping them.
    prev_files = {}
    mpath = os.path.join(out_dir, "manifest.json")
    if os.path.exists(mpath):
        try:
            for s in json.load(open(mpath)).get("steps", []):
                prev_files[s["index"]] = s.get("files", {})
        except Exception:
            pass

    for si in range(n_steps):
        step_tag = f"step{si}"
        print(f"\n[step {si}] -----------------------------------------")
        pre = np.load(pre_files[si]).astype(np.float32)   # (N,T,7)
        keep_path = pre_files[si].replace("actions_full_pre_", "keep_indices_")
        keep = (np.load(keep_path) if os.path.exists(keep_path)
                else np.arange(min(5, len(pre)))).astype(int)
        N, K = len(pre), len(keep)

        ranking = winner_kept = task = votes = None
        if si < len(vg_dirs):
            ranking = json.load(open(os.path.join(vg_dirs[si], "ranking.json")))
            task = ranking.get("task_name")
            votes = ranking.get("votes")
            wi = ranking.get("winner_idx")
            if wi is None and votes is not None:
                wi = int(np.argmax(votes))
            if wi is not None and wi < K:
                winner_kept = int(wi)
        step_rec = {"index": si, "task": task, "n_sampled": N,
                    "keep_indices": keep.tolist(), "votes": votes,
                    "winner_idx": winner_kept,
                    "files": dict(prev_files.get(si, {}))}

        # ----- states -----
        if not args.no_states:
            if si < len(obs_files):
                sp = os.path.join(st_dir, f"{step_tag}_state.png")
                shutil.copy(obs_files[si], sp)
                step_rec["files"]["state"] = sp
                if task and _HAVE_PIL:
                    img = np.asarray(Image.open(obs_files[si]).convert("RGB")).copy()
                    bar = np.empty((48, img.shape[1], 3), np.uint8)
                    bar[:] = (20, 22, 30)
                    img = np.vstack([bar, img])
                    draw_text(img, (14, 10), f"Task: {task}", 28, (235, 238, 244),
                              bold=True)
                    tp = os.path.join(st_dir, f"{step_tag}_state_task.png")
                    Image.fromarray(img).save(tp)
                    step_rec["files"]["state_task"] = tp
            if have_rollout and si < len(segs):
                es, ee = segs[si]
                cap = cv2.VideoCapture(rollout_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, es)
                ep = os.path.join(st_dir, f"{step_tag}_exec.mp4")
                wr = None
                for _ in range(es, ee + 1):
                    ok, f = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    if wr is None:
                        wr = Writer(ep, fps_in, (rgb.shape[1], rgb.shape[0]))
                    wr.write(rgb)
                cap.release()
                if wr is not None:
                    wr.close()
                    step_rec["files"]["exec_clip"] = ep
                    print(f"  state exec clip f{es}-{ee} -> {os.path.basename(ep)}")

        # ----- robot (YAM-Pro) action cloud -----
        if renderer is not None:
            rp = os.path.join(robot_dir, f"{step_tag}_robot.mp4")
            dur = render_robot(pre, keep, winner_kept, rp, robot_dir, step_tag,
                               renderer, int(args.fps), args)
            step_rec["files"]["robot"] = rp
            step_rec["files"]["cloud_png"] = os.path.join(
                robot_dir, f"{step_tag}_cloud.png")
            step_rec["files"]["selected_png"] = os.path.join(
                robot_dir, f"{step_tag}_selected.png")
            print(f"  robot ({K} kept arms, winner={winner_kept}) ~{dur:.1f}s -> "
                  f"{os.path.basename(rp)}")

        # ----- candidate videos (plain, no outlines) -----
        if not args.no_videos and si < len(vg_dirs):
            sd = os.path.join(vids_dir, step_tag)
            os.makedirs(sd, exist_ok=True)
            mp4s = sorted(glob.glob(os.path.join(vg_dirs[si], "*.mp4")),
                          key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            cand_files = []
            for k, src in enumerate(mp4s):
                dst = os.path.join(sd, f"cand{k}.mp4")
                write_candidate(src, dst, args, args.denoise_seed + k, k + 1)
                cand_files.append(dst)
                if k == winner_kept:
                    wdst = os.path.join(sd, f"cand{k}_winner.mp4")
                    write_candidate(src, wdst, args, args.denoise_seed + k,
                                    k + 1, winner=True)
                    step_rec["files"]["winner_clip"] = wdst
            step_rec["files"]["candidates"] = cand_files
            gd = os.path.join(sd, "grid.mp4")
            if grid_clip(mp4s, gd, winner_kept, args.fps, args):
                step_rec["files"]["grid"] = gd
            print(f"  {len(mp4s)} candidate clips (+ grid, winner={winner_kept}) "
                  f"-> {os.path.relpath(sd, out_dir)}/")

        manifest["steps"].append(step_rec)

    if renderer is not None:
        renderer.close()
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] components in {out_dir}\n"
          f"       manifest.json maps every file + the candidate<->color key.")


if __name__ == "__main__":
    sys.exit(main())
