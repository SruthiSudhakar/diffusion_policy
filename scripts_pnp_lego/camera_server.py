"""
RealSense color-stream server for remote diffusion-policy inference.

Run on the box where the camera is plugged in (the i2rt venv has both
pyrealsense2 and portal already):

    cd /home/cvlabusers/Appaji/i2rt && source .venv/bin/activate
    python /home/cvlabusers/Appaji/diffusion_policy/scripts_pnp_lego/camera_server.py

The remote inference process calls get_frame() and gets back a JPEG-encoded
bytes blob of the latest color frame (BGR -> JPEG). Keeps bandwidth manageable
(~50 KB/frame at 1280x720 vs. 2.6 MB raw), so a 10 Hz feed is well under
1 MB/s.
"""
import argparse
import threading
import time

import cv2
import numpy as np
import portal
import pyrealsense2 as rs


class FrameStore:
    def __init__(self, jpeg_quality: int):
        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._stamp: float = 0.0
        self._jpeg_quality = jpeg_quality

    def update(self, bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg", bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._stamp = time.time()

    def get_frame(self) -> dict:
        with self._lock:
            return {"jpeg": self._jpeg, "stamp": np.float64(self._stamp)}


def capture_loop(pipe: rs.pipeline, store: FrameStore, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            frames = pipe.wait_for_frames(timeout_ms=2000)
        except Exception as e:
            print(f"[camera_server] wait_for_frames failed: {e}")
            time.sleep(0.1)
            continue
        c = frames.get_color_frame()
        if not c:
            continue
        bgr = np.asanyarray(c.get_data())
        store.update(bgr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11335)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    args = ap.parse_args()

    print(f"[camera_server] starting RealSense {args.width}x{args.height}@{args.fps} BGR8")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipe.start(cfg)

    store = FrameStore(jpeg_quality=args.jpeg_quality)
    stop = threading.Event()
    t = threading.Thread(target=capture_loop, args=(pipe, store, stop), daemon=True)
    t.start()

    # wait until the first frame is in
    print("[camera_server] waiting for first frame...")
    while True:
        f = store.get_frame()
        if f["jpeg"]:
            break
        time.sleep(0.05)
    print(f"[camera_server] first frame ok ({len(f['jpeg'])} bytes JPEG); binding port {args.port}")

    server = portal.Server(args.port)
    server.bind("get_frame", store.get_frame)
    try:
        server.start()  # blocks
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        pipe.stop()


if __name__ == "__main__":
    main()
