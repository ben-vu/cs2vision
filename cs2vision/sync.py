"""
Temporal synchronisation between broadcast video and demo ticks.

The broadcast and the demo run on independent clocks. To score a video frame
against demo ground truth you need tick(frame). We model it as affine:

    tick = a * src_frame + b

which absorbs both the broadcast/demo frame-rate ratio (a) and the start
offset (b). You provide >=2 anchor correspondences (video src_frame, demo
tick) for visually unambiguous shared events -- round-start freeze-time end,
bomb-plant, a specific kill. Three or more lets least-squares average out the
human picking error and report residual drift (thesis Sec 2.2.2).

Finding the anchors is the part that needs your eyes: open the frame where a
round visibly starts, read its src_frame from the manifest, and read the
matching round_start tick from groundtruth.event_ticks('round_start'). This
module does the fitting and reports how good the alignment is; it does not
fabricate anchors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyncModel:
    a: float
    b: float
    residual_ticks: float          # std of anchor residuals, in ticks
    n_anchors: int
    tickrate: float = 64.0

    def tick_for_frame(self, src_frame: int) -> int:
        return int(round(self.a * src_frame + self.b))

    def residual_seconds(self) -> float:
        return self.residual_ticks / self.tickrate


def fit_sync(anchors: list[tuple[int, int]], tickrate: float = 64.0) -> SyncModel:
    """anchors: list of (video_src_frame, demo_tick)."""
    if len(anchors) < 2:
        raise ValueError("need >= 2 (src_frame, tick) anchors")
    f = np.array([a[0] for a in anchors], float)
    t = np.array([a[1] for a in anchors], float)
    A = np.vstack([f, np.ones_like(f)]).T
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - (a * f + b)
    return SyncModel(a=float(a), b=float(b),
                     residual_ticks=float(resid.std(ddof=0)),
                     n_anchors=len(anchors), tickrate=tickrate)


def ticks_for_manifest(manifest: dict, sync: SyncModel) -> list[int]:
    """Tick for every extracted frame in an ingest manifest."""
    return [sync.tick_for_frame(fr["src_frame"]) for fr in manifest["frames"]]


class DemoCaptureSync(SyncModel):
    """Exact sync for DEMO-PLAYBACK capture: you control playback, so the
    tick of each captured frame is deterministic. If you start capture at
    `start_tick` and render at `capture_fps`, then
        tick = start_tick + frame * (tickrate / capture_fps).
    No anchors, no audio, no drift.
    """
    def __init__(self, start_tick: int, capture_fps: float, tickrate: float = 64.0):
        a = tickrate / capture_fps
        super().__init__(a=a, b=float(start_tick), residual_ticks=0.0,
                         n_anchors=0, tickrate=tickrate)
