"""
Broadcast video ingestion: sample frames from real .mp4 broadcasts.

Extracts frames at a target rate (thesis annotates at 2 fps; the live pipeline
can run at any rate up to source fps) and writes a manifest mapping each saved
frame to its source video-frame index and timestamp. The video-frame index is
what sync.py later aligns to demo ticks.

No simulator dependency. Works on any video OpenCV can decode (install
opencv-python-headless; ffmpeg backend recommended for H.264 broadcasts).
"""
from __future__ import annotations

import argparse
import json
import os

import cv2


def extract_frames(video_path: str, out_dir: str, target_fps: float = 2.0,
                   start_sec: float = 0.0, end_sec: float | None = None,
                   resize_to: tuple[int, int] | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(src_fps / target_fps))
    manifest = dict(video=os.path.abspath(video_path), src_fps=src_fps,
                    target_fps=target_fps, step=step, frames=[])

    start_f = int(start_sec * src_fps)
    end_f = int(end_sec * src_fps) if end_sec else n_total
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    saved = 0
    fidx = start_f
    while True:
        if end_f and fidx >= end_f:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx - start_f) % step == 0:
            if resize_to:
                frame = cv2.resize(frame, resize_to)
            name = f"f{saved:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest["frames"].append(dict(
                file=name, src_frame=fidx, t_sec=fidx / src_fps))
            saved += 1
        fidx += 1
    cap.release()
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()
    man = extract_frames(args.video, args.out_dir, args.fps, args.start, args.end)
    print(f"saved {len(man['frames'])} frames (src {man['src_fps']:.1f} fps, "
          f"step {man['step']}) -> {args.out_dir}")
