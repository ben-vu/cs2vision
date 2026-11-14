"""Integrated pipeline: detect -> project -> track -> score (Experiment C).

run_match() processes every visible frame in a match manifest with one detector and
returns one row per frame (counts for completeness, fidelity, ceiling), the matched
positional errors, and failure-attribution labels.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from .channel import phase_of, unresolvable
from .evaluate import attribute, match_distances
from .tracker import Tracker
from .units import MapMeta


def run_match(match, detector, mode="tracked", verbose=False):
    cfg = match.cfg
    meta = MapMeta.from_cfg(cfg)
    calib, gt, man = match.calib(), match.gt(), match.manifest()
    man = man[man["visible"]].sort_values(["round", "frame_idx"])
    thresholds = cfg["eval"]["fidelity_m"]
    thr_px = [meta.m_to_canon(t) for t in thresholds]
    gate_px = meta.m_to_canon(cfg["eval"]["presence_gate_m"])
    # Match at the largest reported distance, then evaluate each metric with
    # its own threshold. In particular, fidelity at 10 m may exceed the
    # separate completeness measure (presence within 5 m).
    max_match_px = max(gate_px, *thr_px)
    single = cfg["classes"] == "single"
    tracker = Tracker.from_cfg(cfg, meta)
    rows, errors, attrib, lat = [], [], [], []
    cur_round = None
    for _, row in man.iterrows():
        if row["round"] != cur_round:
            tracker.reset()
            cur_round = row["round"]
        img = cv2.imread(str(match.dir / row["file"]))
        if img is None:
            continue
        t0 = time.perf_counter()
        det = detector.detect(img)
        lat.append(time.perf_counter() - t0)
        off = np.array([row["crop_x0"], row["crop_y0"]], float)
        raw_frame = det.xy + off if len(det) else np.zeros((0, 2))
        raw_canon = calib.frame_to_canon(raw_frame)
        raw_cls = np.zeros(len(det), int) if single else det.cls
        if mode == "tracked":
            tracker.update(raw_canon, raw_cls, det.score)
            pred_xy, pred_cls, _ = tracker.state()
        else:
            pred_xy, pred_cls = raw_canon, raw_cls

        gt_canon, gt_cls, _ = gt.at(row["tick"])
        if len(gt_canon) == 0:
            continue
        gt_frame = calib.canon_to_frame(gt_canon)
        unres = unresolvable(gt_frame, calib.blip_px * cfg["eval"].get("merge_frac", 1.0))
        dist = match_distances(pred_xy, pred_cls, gt_canon, gt_cls, max_match_px)
        present = dist <= gate_px
        errors.extend(meta.canon_to_m(dist[present]).tolist())
        labels = attribute(dist, thr_px[0], unres, gt_frame, gt_cls, gt_canon,
                           raw_frame, raw_cls, raw_canon, calib.blip_px)
        attrib.extend(labels)
        rec = {"match": match.id, "round": int(row["round"]), "frame_idx": int(row["frame_idx"]),
               "phase": phase_of(row["secs_into_round"], bool(row["post_plant"]),
                                 cfg["eval"]["phases"]),
               "n_alive": len(gt_canon), "n_pred": len(pred_xy),
               "n_present": int(present.sum()), "n_recoverable": int((~unres).sum())}
        for t, tp in zip(thresholds, thr_px):
            rec[f"n_fid_{t:g}m"] = int((dist <= tp).sum())
        for lab in ("channel", "detection", "localisation", "association"):
            rec[f"fail_{lab}"] = labels.count(lab)
        rows.append(rec)
    frames = pd.DataFrame(rows)
    if verbose and len(frames):
        print(f"  {match.id}: {len(frames)} frames, median latency "
              f"{1000 * np.median(lat):.1f} ms")
    return frames, np.array(errors), np.array(lat)
