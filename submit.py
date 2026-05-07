#!/usr/bin/env python3
"""Submit one (image, [T,7] actions) request to the cv16 video-generation server.

Usage:
    python submit.py <image.png|jpg|...> <actions.npy> [name]

`actions.npy` must be a [T,7] or [T,1,7] float array (one 7-DOF action per frame).
Length is truncated to the nearest 4n+1 and capped at 33 to match the server's
default --video_length.

This script is meant to run on your *local* machine. It scps the image and the
manifest into cv16:/proj/vondrick3/HunyuanVideo-1.5-train-sruthi/inbox/,
which the running serve.py picks up automatically.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REMOTE = "sruthi@cv16.cs.columbia.edu"
REMOTE_REPO = "/proj/vondrick3/HunyuanVideo-1.5-train-sruthi"
REMOTE_INBOX = f"{REMOTE_REPO}/inbox"
REMOTE_OUTPUTS = f"{REMOTE_REPO}/outputs/serve"
SERVER_VIDEO_LENGTH = 33  # must match --video_length in run_serve.sh


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    image_path = Path(sys.argv[1]).resolve()
    actions_path = Path(sys.argv[2]).resolve()
    name = sys.argv[3] if len(sys.argv) > 3 else f"req_{datetime.now():%Y%m%d_%H%M%S}"

    if not image_path.is_file():
        sys.exit(f"image not found: {image_path}")
    if not actions_path.is_file():
        sys.exit(f"actions .npy not found: {actions_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        sys.exit(f"unsupported image suffix: {image_path.suffix}")

    actions = np.load(actions_path)
    if actions.ndim == 3 and actions.shape[1] == 1:
        actions = actions[:, 0, :]
    if actions.ndim != 2 or actions.shape[1] != 7:
        sys.exit(f"actions must be [T,7] or [T,1,7], got {actions.shape}")

    T = actions.shape[0]
    T -= (T - 1) % 4
    T = min(T, SERVER_VIDEO_LENGTH)
    if T < 1:
        sys.exit(f"actions too short after 4n+1 truncation: {actions.shape}")
    actions = actions[:T].astype(np.float64)

    img_name = f"{name}{image_path.suffix.lower()}"
    manifest = {
        "image_paths": np.array([img_name] * T),
        "trajectory": actions,
    }
    local_npy = Path("/tmp") / f"{name}.npy"
    np.save(local_npy, manifest, allow_pickle=True)

    def run(cmd):
        subprocess.run(cmd, check=True)

    run(["scp", str(image_path), f"{REMOTE}:{REMOTE_INBOX}/{img_name}"])
    run(["scp", str(local_npy), f"{REMOTE}:{REMOTE_INBOX}/{name}.npy.partial"])
    run(["ssh", REMOTE, "mv",
         f"{REMOTE_INBOX}/{name}.npy.partial",
         f"{REMOTE_INBOX}/{name}.npy"])

    print(f"submitted: {name}  (T={T} frames)")
    print(f"output (when ready): {REMOTE}:{REMOTE_OUTPUTS}/*_{name}_generated.mp4")


if __name__ == "__main__":
    main()
