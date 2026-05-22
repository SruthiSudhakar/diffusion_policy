"""
Live RealSense overlay for scene alignment.

Blends a reference image (e.g. observations/frame_000000.jpg from a previous
run) on top of the live RealSense color stream so you can reposition objects
in the workspace to match the original setup.

Usage:
    python scripts_pnp_lego/realsense_overlay.py \
        /path/to/run/observations/frame_000000.jpg

Keys:
    +/=    increase blend alpha (more reference)
    -/_    decrease blend alpha (more live)
    space  toggle reference overlay on/off
    q/esc  quit

Note: realsense-viewer must be closed; only one process can hold the device.
"""
import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW = "realsense overlay (q to quit)"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="Path to reference image to overlay (BGR JPEG/PNG).")
    ap.add_argument("--rs-width", type=int, default=1280)
    ap.add_argument("--rs-height", type=int, default=720)
    ap.add_argument("--rs-fps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.5, help="Initial blend weight on reference (0..1).")
    ap.add_argument("--serial", default="317222070925", help="RealSense serial; default = camera 2 (317222070925).")
    return ap.parse_args()


def load_reference(path: str, target_wh: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"[overlay] could not read reference image: {path}")
    tw, th = target_wh
    if (img.shape[1], img.shape[0]) != (tw, th):
        print(f"[overlay] resizing reference {img.shape[1]}x{img.shape[0]} -> {tw}x{th}")
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return img


def get_screen_size(fallback: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return fallback


def start_pipeline(width: int, height: int, fps: int, serial: Optional[str]) -> rs.pipeline:
    pipe = rs.pipeline()
    cfg = rs.config()
    if serial is not None:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipe.start(cfg)
    return pipe


def draw_hud(frame: np.ndarray, alpha: float, overlay_on: bool, ref_name: str) -> None:
    lines = [
        f"alpha={alpha:.2f}  overlay={'ON' if overlay_on else 'OFF'}",
        f"ref: {ref_name}",
        "+/- alpha   space toggle   q quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main() -> None:
    args = parse_args()
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    overlay_on = True
    ref_name = os.path.basename(args.ref)

    print(f"[overlay] starting RealSense {args.rs_width}x{args.rs_height}@{args.rs_fps} BGR8")
    pipe = start_pipeline(args.rs_width, args.rs_height, args.rs_fps, args.serial)
    try:
        ref = load_reference(args.ref, (args.rs_width, args.rs_height))
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        sw, sh = get_screen_size()
        cv2.resizeWindow(WINDOW, sw, sh)
        cv2.moveWindow(WINDOW, 0, 0)

        while True:
            frames = pipe.wait_for_frames(timeout_ms=2000)
            c = frames.get_color_frame()
            if not c:
                continue
            live = np.asanyarray(c.get_data())

            if overlay_on:
                blended = cv2.addWeighted(live, 1.0 - alpha, ref, alpha, 0.0)
            else:
                blended = live.copy()

            draw_hud(blended, alpha, overlay_on, ref_name)
            cv2.imshow(WINDOW, blended)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if key in (ord('q'), 27):
                break
            if key in (ord('+'), ord('=')):
                alpha = float(np.clip(alpha + 0.05, 0.0, 1.0))
            elif key in (ord('-'), ord('_')):
                alpha = float(np.clip(alpha - 0.05, 0.0, 1.0))
            elif key == ord(' '):
                overlay_on = not overlay_on
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
