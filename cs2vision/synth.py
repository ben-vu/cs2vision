"""Synthetic matches for tests and the `cs2v selftest` dry run.

A synthetic match is a folder with the same files a real one has after `parse-demo`
(gt_ticks.parquet, rounds.csv) plus a rendered video.mp4 and anchors.csv. It exists
only to prove the software runs end to end; numbers produced from it say nothing about
real broadcasts. Unlike the Stage-0 pilot simulator, blips are NOT hidden by smoke:
every alive player is drawn, as on the production observer radar.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .units import MapMeta

CT_BGR, T_BGR = (255, 120, 20), (20, 200, 255)


def _texture(size, rng):
    base = rng.integers(40, 90, (size // 8, size // 8)).astype(np.uint8)
    base = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    tex = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for _ in range(25):  # map "rooms": light, low-saturation rectangles
        x, y = rng.integers(0, size, 2)
        w, h = rng.integers(size // 20, size // 6, 2)
        g = int(rng.integers(90, 150))
        cv2.rectangle(tex, (int(x), int(y)), (int(x + w), int(y + h)), (g, g + 8, g), -1)
    return tex


def make_synthetic_match(out_dir, meta: MapMeta, seed=0, n_rounds=4, round_s=20.0,
                         gap_s=10.0, fps=30, frame_size=(640, 360), rect=(24, 20, 200, 200),
                         blip_px=7, offset_s=4.0, cut_after_round=2, cut_s=1.5,
                         tickrate=64, anchor_error_s=0.3):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    W, H = frame_size
    s = meta.radar_size

    # ---- ground truth --------------------------------------------------------------------
    rows, rounds, starts = [], [], []
    tick0 = 2000
    for k in range(n_rounds):
        fz = tick0 + int(k * (round_s + gap_s) * tickrate)
        end = fz + int(round_s * tickrate)
        plant = fz + int(0.6 * round_s * tickrate) if k % 2 == 1 else -1
        rounds.append({"round": k + 1, "freeze_end_tick": fz, "end_tick": end,
                       "plant_tick": plant, "winner": "CT"})
        pos = np.r_[rng.normal([700, 300], 30, (5, 2)), rng.normal([300, 780], 30, (5, 2))]
        vel = rng.normal(0, 20, (10, 2))
        death = np.where(rng.random(10) < 0.5, rng.integers(fz + tickrate * 5, end, 10), 10**9)
        for tk in range(fz - tickrate * 2, end + 1):
            if tk >= fz:
                vel += rng.normal(0, 3, (10, 2))
                vel = np.clip(vel, -45, 45)        # <= 45 canonical px/s ~ 225 u/s
                pos = np.clip(pos + vel / tickrate, 80, s - 80)
            wx, wy = meta.canon_to_world(pos[:, 0], pos[:, 1])
            for i in range(10):
                rows.append((tk, str(76561190000000000 + i), f"p{i}", 3 if i < 5 else 2,
                             tk < death[i], wx[i], wy[i], 0.0, pos[i, 0], pos[i, 1]))
        starts.append(fz)
    gt = pd.DataFrame(rows, columns=["tick", "steamid", "name", "team_num", "alive", "x", "y",
                                     "z", "cx", "cy"])
    gt.to_parquet(out / "gt_ticks.parquet", index=False)
    rounds = pd.DataFrame(rounds)
    rounds.to_csv(out / "rounds.csv", index=False)

    # ---- video ------------------------------------------------------------------------------
    def video_time_of_tick(tk):
        t = offset_s + (tk - tick0) / tickrate
        if cut_after_round is not None and tk > rounds["end_tick"].iloc[cut_after_round - 1]:
            t -= cut_s  # an edited VOD: a piece of the break was cut out
        return t

    last = rounds["end_tick"].iloc[-1]
    duration = video_time_of_tick(last) + 2.0
    x, y, w, h = rect
    tex = cv2.resize(_texture(512, rng), (w, h), interpolation=cv2.INTER_AREA)
    by_tick = {t: g for t, g in gt.groupby("tick")}
    ticks_sorted = np.array(sorted(by_tick))
    writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W, H))
    replay = (video_time_of_tick(rounds["freeze_end_tick"].iloc[-1]) + 8,
              video_time_of_tick(rounds["freeze_end_tick"].iloc[-1]) + 9.5)
    for f in range(int(duration * fps)):
        t = f / fps
        frame = np.full((H, W, 3), 35, np.uint8)
        cv2.putText(frame, "SYNTHETIC", (W - 180, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (80, 80, 80), 2)
        frame[y:y + h, x:x + w] = tex
        # invert the piecewise mapping to find the tick on screen
        tk_est = tick0 + (t - offset_s) * tickrate
        if cut_after_round is not None and \
                video_time_of_tick(rounds["end_tick"].iloc[cut_after_round - 1]) < t:
            tk_est = tick0 + (t - offset_s + cut_s) * tickrate
        i = np.searchsorted(ticks_sorted, tk_est)
        if 0 <= i < len(ticks_sorted) and abs(ticks_sorted[i] - tk_est) <= 1:
            g = by_tick[ticks_sorted[i]]
            g = g[g["alive"]]
            for cx, cy, team in zip(g["cx"], g["cy"], g["team_num"]):
                px = x + cx * w / s
                py = y + cy * h / s
                cv2.circle(frame, (int(round(px)), int(round(py))), blip_px // 2,
                           CT_BGR if team == 3 else T_BGR, -1, lineType=cv2.LINE_AA)
        if replay[0] <= t < replay[1]:
            frame[:] = rng.integers(0, 255, frame.shape, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    true_starts = [video_time_of_tick(t) for t in rounds["freeze_end_tick"]]
    pd.DataFrame({"round": [1], "video_time": [round(true_starts[0] + anchor_error_s, 2)]}) \
        .to_csv(out / "anchors.csv", index=False)
    (out / "demo.dem").write_text("synthetic placeholder: ground truth already parsed\n")
    pd.DataFrame({"round": rounds["round"], "true_start_time": true_starts}) \
        .to_csv(out / "truth_sync.csv", index=False)
    return {"rect": rect, "blip_px": blip_px, "true_starts": true_starts}
