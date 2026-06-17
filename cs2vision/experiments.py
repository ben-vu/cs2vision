"""
Experiments 1-4 on REAL data (thesis Sec 2.3). Structurally identical to the
pilot harness; the data source is real frames + real demo ground truth instead
of the simulator.

Inputs you provide (see README_REAL.md):
  * a list of CLIPS, each = (frames_dir, gt_json, sync, label_dir)
      - frames_dir : ingest.extract_frames output (with manifest.json)
      - gt_json    : groundtruth output for that clip
      - sync       : fitted sync.SyncModel
      - label_dir  : CVAT YOLO labels for the annotated frames (Exp 1 only)
  * a trained YoloDetector and a calibrated radar (Calibrated/RegisteredRadar)

This module does not run in the analysis sandbox (it needs your trained model
and real footage). It is written to run unchanged on your machine. For a
plumbing check that the code path executes, run tests/smoke_test.py.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
from scipy import stats

from .classes import CLASSES
from .dataset import assemble_clip, read_yolo_labels
from .evaluation import evaluate_detections, map_summary
from .pipeline import CS2VisionPipeline

IOU_THRS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
CONDITIONS = ["detection_only", "smoke_flagging", "smoke_imputation"]


def experiment_channel(clips, rect, radar_meta, out_dir="outputs/metrics",
                       method_fidelity: float | None = None):
    """Reframed headline analysis: the broadcast spatial-channel ceiling.

    Needs only ground truth + radar geometry (no detector). Reports the
    precision floor and the fraction of authoritative state that is recoverable
    from the broadcast at all -- the upper bound any method can reach. If you
    pass your measured pipeline fidelity, also reports it as a fraction of the
    ceiling (the honest broadcast-only-reconstruction number).
    """
    import json
    import os
    from .channel import channel_ceiling, fraction_of_ceiling
    from .dataset import assemble_clip

    os.makedirs(out_dir, exist_ok=True)
    records = []
    for frames_dir, gt_json, sync, _lbl in clips:
        records += assemble_clip(frames_dir, gt_json, sync)
    out = channel_ceiling(records, rect, radar_meta)
    if method_fidelity is not None:
        out["method_fidelity"] = float(method_fidelity)
        out["fraction_of_ceiling"] = fraction_of_ceiling(method_fidelity, out)
    out.pop("per_frame_ceiling", None)            # keep the file small
    json.dump(out, open(f"{out_dir}/exp_channel_ceiling.json", "w"))
    return out


# ---------------- Experiment 1: detection ---------------------------------- #
def experiment_1(detector, clips, out_dir="outputs/metrics"):
    os.makedirs(out_dir, exist_ok=True)
    per_frame_dets, per_frame_gts = [], []
    for frames_dir, _gt, _sync, label_dir in clips:
        manifest = json.load(open(os.path.join(frames_dir, "manifest.json")))
        for fr in manifest["frames"]:
            img = cv2.imread(os.path.join(frames_dir, fr["file"]))
            if img is None:
                continue
            h, w = img.shape[:2]
            dets = detector.detect(img)
            per_frame_dets.append([(d.cls, d.bbox, d.conf) for d in dets])
            lbl = os.path.join(label_dir, os.path.splitext(fr["file"])[0] + ".txt")
            per_frame_gts.append(read_yolo_labels(lbl, w, h))
    res = evaluate_detections(per_frame_dets, per_frame_gts, CLASSES, IOU_THRS)
    maps = map_summary(res, CLASSES)
    from .classes import CHANNELS, PRIMARY_CHANNELS
    ap50 = {c: v["ap"] for c, v in res[0.5].items()}

    def chan_map(names):
        vals = [ap50[c] for c in names if c in ap50]
        return float(np.mean(vals)) if vals else None

    out = dict(
        per_class_ap50=ap50,
        per_channel_ap50={ch: chan_map(cls) for ch, cls in CHANNELS.items()},
        # headline: reconstruction-relevant channels only; players excluded
        primary_map50=chan_map([c for ch in PRIMARY_CHANNELS for c in CHANNELS[ch]]),
        auxiliary_map50=chan_map(CHANNELS["auxiliary"]),
        map50=maps[0.5],
        map5095=float(np.mean(list(maps.values()))),
    )
    json.dump(out, open(f"{out_dir}/exp1_detection.json", "w"))
    return out


# ---------------- Experiments 2-4: geometry / smoke / full ----------------- #
def run_pipeline_experiments(detector, radar_factory, clips, out_dir="outputs/metrics",
                             round_len=22):
    """radar_factory() -> a fresh radar estimator per clip (state resets)."""
    os.makedirs(out_dir, exist_ok=True)
    per_round = {c: [] for c in CONDITIONS}
    per_round_fid = {c: [] for c in CONDITIONS}
    err_matched, times, fid_rows = [], [], []
    from scipy.optimize import linear_sum_assignment

    for frames_dir, gt_json, sync, _lbl in clips:
        records = assemble_clip(frames_dir, gt_json, sync)
        pipe = CS2VisionPipeline(detector, radar_factory())
        comp_acc = {c: [] for c in CONDITIONS}
        fidc_acc = {c: [] for c in CONDITIONS}
        for t, rec in enumerate(records):
            img = cv2.imread(rec["frame_path"])
            if img is None:
                continue
            t0 = time.time()
            o = pipe.process(img)
            times.append(time.time() - t0)
            for c in CONDITIONS:
                comp, fid, n = CS2VisionPipeline.score_frame(o, rec["players"], c)
                comp_acc[c].append(comp); fidc_acc[c].append(fid)
            for team in ("CT", "T"):
                gt_pos = np.array([p["map_pos"] for p in rec["players"]
                                   if p["team"] == team and p["alive"]])
                est = o["positions"][team]
                if len(gt_pos) and len(est):
                    cost = np.linalg.norm(gt_pos[:, None, :] - est[None, :, :], axis=2)
                    gi, ei = linear_sum_assignment(cost)
                    for a, b in zip(gi, ei):
                        if cost[a, b] <= 25.0:
                            err_matched.append(float(cost[a, b]))
            if (t + 1) % round_len == 0:
                for c in CONDITIONS:
                    per_round[c].append(float(np.mean(comp_acc[c][-round_len:])))
                    per_round_fid[c].append(float(np.mean(fidc_acc[c][-round_len:])))
                fid_rows.append(dict(fidelity=per_round_fid["smoke_imputation"][-1]))

    def paired(x, y):
        x, y = np.array(x), np.array(y)
        if len(x) < 2:
            return dict(t=float("nan"), p=float("nan"), mean_diff=float("nan"))
        tval, p = stats.ttest_rel(y, x)
        return dict(t=float(tval), p=float(p), mean_diff=float((y - x).mean()))

    exp3 = dict(
        rounds=len(per_round["detection_only"]),
        completeness={c: dict(mean=float(np.mean(per_round[c]) if per_round[c] else 0),
                              sd=float(np.std(per_round[c], ddof=1)) if len(per_round[c]) > 1 else 0,
                              per_round=per_round[c]) for c in CONDITIONS},
        fidelity={c: dict(mean=float(np.mean(per_round_fid[c]) if per_round_fid[c] else 0),
                          per_round=per_round_fid[c]) for c in CONDITIONS},
        paired_t=dict(
            A_vs_B=paired(per_round["detection_only"], per_round["smoke_flagging"]),
            A_vs_C=paired(per_round["detection_only"], per_round["smoke_imputation"])),
    )
    exp4 = dict(
        fidelity_mean=float(np.mean([r["fidelity"] for r in fid_rows]) if fid_rows else 0),
        matched_positional_error=dict(
            mean=float(np.mean(err_matched)) if err_matched else None,
            median=float(np.median(err_matched)) if err_matched else None,
            p95=float(np.percentile(err_matched, 95)) if err_matched else None,
            n=len(err_matched)),
        fps=float(1.0 / np.mean(times)) if times else None,
    )
    json.dump(exp3, open(f"{out_dir}/exp3_completeness.json", "w"))
    json.dump(exp4, open(f"{out_dir}/exp4_full_pipeline.json", "w"))
    return exp3, exp4
