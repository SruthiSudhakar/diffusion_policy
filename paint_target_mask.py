"""Interactive painter: load a frame, draw the target region with the mouse,
save a binary mask (PNG, uint8, 0/255) at the same resolution as the input.

Usage
-----
python paint_target_mask.py \
  --image data/jgd/2026.05.18/15.12.41_train_diffusion_unet_hybrid_push_bowl_image_only_trajectory/checkpoints/epoch=0250-train_loss=0.0247/19jgd_6s_2026-05-19_14-43-32_VLM/observations/image2/frame_000000.jpg \
  --out  data/jgd/2026.05.18/15.12.41_train_diffusion_unet_hybrid_push_bowl_image_only_trajectory/checkpoints/epoch=0250-train_loss=0.0247/target_mask.png \
  --scale 2

python paint_target_mask.py \
--image /home/cvlabusers/Appaji/diffusion_policy/data/jgd/2026.05.18/15.12.41_train_diffusion_unet_hybrid_push_bowl_image_only_trajectory/checkpoints/epoch=0250-train_loss=0.0247/37jgd_f_2026-05-19_16-42-24_VLM/observations/image2/frame_000000.jpg \
--out bowl_mask.png \
--scale 2

python paint_target_mask.py \
  --image data/jgd/2026.05.18/15.12.41_train_diffusion_unet_hybrid_push_bowl_image_only_trajectory/checkpoints/epoch=0250-train_loss=0.0247/10jgd_4s_2026-05-19_12-26-10/observations/image2/frame_000000.jpg \
  --out bowl_mask.png \
  --scale 2

Controls
--------
    left-mouse drag : paint
    right-mouse drag: erase
    [ / ]           : decrease / increase brush radius
    e               : toggle eraser
    c               : clear mask
    u               : undo last stroke
    s               : save mask to --out and exit
    q / ESC         : quit without saving
"""

import argparse
import pathlib
import sys

import cv2
import numpy as np


class Painter:
    def __init__(self, image_bgr: np.ndarray, scale: float):
        self.img = image_bgr
        self.h, self.w = image_bgr.shape[:2]
        self.scale = scale
        self.disp_w = int(round(self.w * scale))
        self.disp_h = int(round(self.h * scale))
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.brush = 12
        self.erasing = False
        self.drawing = False
        self.last_pt = None
        self.cursor = None
        self.undo_stack: list[np.ndarray] = []
        self.stroke_snapshot: np.ndarray | None = None

    def _to_img_xy(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x / self.scale)), int(round(y / self.scale))

    def _paint_segment(self, p0, p1, value: int) -> None:
        cv2.line(self.mask, p0, p1, value, thickness=self.brush * 2,
                 lineType=cv2.LINE_8)
        cv2.circle(self.mask, p1, self.brush, value, thickness=-1)

    def on_mouse(self, event, x, y, flags, _param):
        ix, iy = self._to_img_xy(x, y)
        self.cursor = (ix, iy)

        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.erasing = event == cv2.EVENT_RBUTTONDOWN
            self.stroke_snapshot = self.mask.copy()
            self.last_pt = (ix, iy)
            value = 0 if self.erasing else 255
            cv2.circle(self.mask, (ix, iy), self.brush, value, thickness=-1)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            value = 0 if self.erasing else 255
            if self.last_pt is not None:
                self._paint_segment(self.last_pt, (ix, iy), value)
            self.last_pt = (ix, iy)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            if self.drawing and self.stroke_snapshot is not None:
                self.undo_stack.append(self.stroke_snapshot)
                if len(self.undo_stack) > 50:
                    self.undo_stack.pop(0)
            self.drawing = False
            self.last_pt = None
            self.stroke_snapshot = None

    def render(self) -> np.ndarray:
        overlay = self.img.copy()
        m = self.mask > 0
        if m.any():
            tint = np.zeros_like(overlay)
            tint[..., 1] = 255  # green
            overlay[m] = (0.5 * overlay[m] + 0.5 * tint[m]).astype(np.uint8)
            contours, _ = cv2.findContours(
                self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

        if self.scale != 1.0:
            overlay = cv2.resize(
                overlay, (self.disp_w, self.disp_h),
                interpolation=cv2.INTER_LINEAR)

        if self.cursor is not None:
            cx = int(round(self.cursor[0] * self.scale))
            cy = int(round(self.cursor[1] * self.scale))
            r = max(1, int(round(self.brush * self.scale)))
            color = (0, 0, 255) if self.erasing else (255, 255, 255)
            cv2.circle(overlay, (cx, cy), r, color, 1)

        area = int((self.mask > 0).sum())
        hud = (f'brush={self.brush}  mode={"erase" if self.erasing else "paint"}  '
               f'area={area}px  [/]=brush  e=erase  c=clear  u=undo  s=save  q=quit')
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(overlay, hud, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return overlay

    def undo(self) -> None:
        if self.undo_stack:
            self.mask = self.undo_stack.pop()

    def clear(self) -> None:
        if self.mask.any():
            self.undo_stack.append(self.mask.copy())
        self.mask[:] = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, type=pathlib.Path,
                    help='reference frame (jpg/png) or .mp4 (uses --frame index)')
    ap.add_argument('--frame', type=int, default=0,
                    help='frame index if --image is a video (default 0)')
    ap.add_argument('--out', required=True, type=pathlib.Path,
                    help='where to write the mask PNG (uint8, 0/255)')
    ap.add_argument('--scale', type=float, default=2.0,
                    help='display scale (mask is always saved at original res)')
    ap.add_argument('--init-mask', type=pathlib.Path, default=None,
                    help='optional existing mask to start from')
    args = ap.parse_args()

    if not args.image.exists():
        print(f'image not found: {args.image}', file=sys.stderr)
        return 2

    if args.image.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
        cap = cv2.VideoCapture(str(args.image))
        if not cap.isOpened():
            print(f'cannot open video {args.image}', file=sys.stderr)
            return 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            print(f'cannot read frame {args.frame} from {args.image}',
                  file=sys.stderr)
            return 2
    else:
        img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if img is None:
            print(f'cannot read {args.image}', file=sys.stderr)
            return 2

    painter = Painter(img, scale=args.scale)

    if args.init_mask is not None and args.init_mask.exists():
        init = cv2.imread(str(args.init_mask), cv2.IMREAD_GRAYSCALE)
        if init is None or init.shape != painter.mask.shape:
            print(f'[warn] ignoring --init-mask: shape mismatch or unreadable',
                  file=sys.stderr)
        else:
            painter.mask = (init > 0).astype(np.uint8) * 255

    win = 'paint target mask'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, painter.on_mouse)

    while True:
        cv2.imshow(win, painter.render())
        k = cv2.waitKey(16) & 0xFF
        if k == 255:
            continue
        if k in (ord('q'), 27):
            print('quit without saving')
            cv2.destroyAllWindows()
            return 1
        if k == ord('s'):
            args.out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out), painter.mask)
            area = int((painter.mask > 0).sum())
            print(f'wrote {args.out}  ({area} px set, '
                  f'{painter.w}x{painter.h})')
            cv2.destroyAllWindows()
            return 0
        if k == ord('e'):
            painter.erasing = not painter.erasing
        elif k == ord('c'):
            painter.clear()
        elif k == ord('u'):
            painter.undo()
        elif k == ord(']') or k == ord('='):
            painter.brush = min(200, painter.brush + 2)
        elif k == ord('[') or k == ord('-'):
            painter.brush = max(1, painter.brush - 2)


if __name__ == '__main__':
    sys.exit(main())
