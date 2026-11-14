"""Visual checks: demo positions drawn over the broadcast radar.

If calibration and sync are right, every coloured ring sits on a blip. Run this after
`cs2v align` and look at the sheet before ingesting anything.
"""
from __future__ import annotations

import cv2
import numpy as np

from .blobs import BlobDetector
from .sync import tick_at
from .video import VideoReader

RING = {0: (255, 160, 0), 1: (0, 200, 255)}


def check_sheet(match, n=12, zoom=2, seed=0):
    cfg = match.cfg
    tr = cfg["demo"]["tickrate"]
    calib, sync, gt = match.calib(), match.sync(), match.gt()
    blob = BlobDetector.from_cfg(cfg, calib.blip_px)
    ok = sync[sync["ok"]] if sync["ok"].any() else sync
    rng = np.random.default_rng(seed)
    picks = ok.iloc[rng.integers(0, len(ok), n)]
    reader = VideoReader(match.video)
    x0, y0, _, _ = calib.crop_box()
    tiles = []
    for _, r in picks.iterrows():
        dur = (r["end_tick"] - r["freeze_end_tick"]) / tr
        t = r["start_time"] + rng.uniform(2, max(3, dur - 1))
        frame, t = reader.read_at(t)
        if frame is None:
            continue
        crop = calib.crop(frame)
        big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        pts, cls, _ = gt.at(tick_at(r, t, tr))
        for (x, y), c in zip(calib.canon_to_frame(pts) - [x0, y0], cls):
            cv2.circle(big, (int(x * zoom), int(y * zoom)), int(calib.blip_px * zoom * 0.7),
                       RING.get(int(c), (0, 255, 0)), 1)
        for x, y in blob.detect(crop).xy:
            cv2.drawMarker(big, (int(x * zoom), int(y * zoom)), (255, 255, 255),
                           cv2.MARKER_CROSS, 6, 1)
        cv2.putText(big, f"r{int(r['round'])} t={t:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        tiles.append(big)
    reader.close()
    if not tiles:
        raise SystemExit("no frames could be read")
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                                cv2.BORDER_CONSTANT) for t in tiles]
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = match.dir / "check_sheet.png"
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  (rings = demo positions, crosses = detected blobs)")
    return out
