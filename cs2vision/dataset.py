"""
Assemble a scored-ready clip from the real artefacts:
  ingest manifest (frame <-> src_frame)  +  sync model (src_frame -> tick)
  +  demo ground truth (tick -> players/smokes)
=> per-frame records: (frame_path, gt_players, gt_smokes)

Also adapts CVAT YOLO-format exports into the {cls,bbox,occluded} ground-truth
entity format that evaluation.py expects for Experiment 1.
"""
from __future__ import annotations

import json
import os
from bisect import bisect_left

from .classes import CLASSES


def _nearest_tick(sorted_ticks, t):
    i = bisect_left(sorted_ticks, t)
    if i == 0:
        return sorted_ticks[0]
    if i == len(sorted_ticks):
        return sorted_ticks[-1]
    lo, hi = sorted_ticks[i - 1], sorted_ticks[i]
    return lo if (t - lo) <= (hi - t) else hi


def assemble_clip(frames_dir: str, gt_json: str, sync):
    """Returns list of dict(frame_path, tick, players, smokes)."""
    manifest = json.load(open(os.path.join(frames_dir, "manifest.json")))
    gts = json.load(open(gt_json))
    by_tick = {g["tick"]: g for g in gts}
    sorted_ticks = sorted(by_tick)
    records = []
    for fr in manifest["frames"]:
        tick = sync.tick_for_frame(fr["src_frame"])
        nt = _nearest_tick(sorted_ticks, tick)
        g = by_tick[nt]
        records.append(dict(
            frame_path=os.path.join(frames_dir, fr["file"]),
            tick=tick,
            players=g["players"],
            smokes=g.get("smokes", [])))
    return records


def read_yolo_labels(label_path: str, img_w: int, img_h: int,
                     class_names=CLASSES) -> list[dict]:
    """CVAT YOLO-format .txt -> [{cls,bbox(px),occluded=False}] for evaluation."""
    out = []
    if not os.path.exists(label_path):
        return out
    for line in open(label_path):
        p = line.split()
        if len(p) < 5:
            continue
        ci, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        out.append(dict(cls=class_names[ci], bbox=[x1, y1, x2, y2], occluded=False))
    return out
