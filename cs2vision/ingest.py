"""Sample live-round frames, crop the radar and write a manifest (thesis Stage 1).

Only rounds that synced cleanly (sync.csv ok == True) are used. Each sampled frame is
checked against the calibration reference crop; frames whose radar is hidden or
covered (replays, scoreboards, transitions) are recorded but marked visible = False
and excluded from labelling and evaluation (thesis Section 2.2.3, third check).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .radar import radar_ncc
from .sync import tick_at
from .video import VideoReader


def ingest_match(match, verbose=True) -> pd.DataFrame:
    cfg = match.cfg
    tr = cfg["demo"]["tickrate"]
    fps = cfg["video"]["sample_fps"]
    calib, sync, ref = match.calib(), match.sync(), match.ref()
    x0, y0, _, _ = calib.crop_box()
    match.frames_dir.mkdir(exist_ok=True)
    reader = VideoReader(match.video)
    rows = []
    for _, r in sync[sync["ok"]].iterrows():
        dur = (r["end_tick"] - r["freeze_end_tick"]) / tr
        for t in np.arange(r["start_time"], r["start_time"] + dur, 1.0 / fps):
            frame, t_exact = reader.read_at(t)
            if frame is None:
                continue
            crop = calib.crop(frame)
            ncc = radar_ncc(crop, ref) if ref is not None else 1.0
            idx = reader.index_of(t_exact)
            name = f"r{int(r['round']):02d}_{idx:07d}.png"
            visible = ncc >= cfg["video"]["radar_visible_min_ncc"]
            if visible:
                cv2.imwrite(str(match.frames_dir / name), crop)
            rows.append({"match": match.id, "round": int(r["round"]), "frame_idx": idx,
                         "time": t_exact, "tick": tick_at(r, t_exact, tr),
                         "secs_into_round": t_exact - r["start_time"],
                         "post_plant": bool(r["plant_tick"] >= 0 and
                                            tick_at(r, t_exact, tr) >= r["plant_tick"]),
                         "ncc": ncc, "visible": bool(visible),
                         "file": f"frames/{name}" if visible else "",
                         "crop_x0": x0, "crop_y0": y0})
    reader.close()
    man = pd.DataFrame(rows)
    man.to_csv(match.manifest_path, index=False)
    if verbose and len(man):
        print(f"[{match.id}] {len(man)} frames sampled, {int(man['visible'].sum())} with "
              f"visible radar ({100 * man['visible'].mean():.1f}%)")
    return man
