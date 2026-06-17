"""
CS2Vision evaluation module (thesis §2.3.1 / §2.3.4 metrics).

Implements standard object-detection evaluation:
  * greedy confidence-ranked matching of detections to ground truth at a
    given IoU threshold
  * occluded ground-truth entities act as "ignore" regions (detections
    matching them are neither TP nor FP; unmatched occluded GT is not FN),
    following the COCO crowd-region convention
  * per-class AP via 101-point interpolation (COCO style)
  * mAP@0.50 and mAP@[0.50:0.95] (10 thresholds)
"""

from __future__ import annotations
import numpy as np

from .detector import iou


def match_frame(dets, gts, iou_thr):
    """dets: list[(cls,bbox,conf)] sorted desc by conf; gts: list[dict].
    Returns (tp_flags, fp_flags, n_valid_gt) aligned with dets order."""
    used = [False] * len(gts)
    tp, fp = [], []
    for cls, bbox, conf in dets:
        best, best_iou, best_ignored = -1, iou_thr, False
        for gi, g in enumerate(gts):
            if used[gi] or g["cls"] != cls:
                continue
            i = iou(bbox, g["bbox"])
            if i >= best_iou:
                best, best_iou, best_ignored = gi, i, g.get("occluded", False)
        if best >= 0:
            used[best] = True
            if best_ignored:
                tp.append(0); fp.append(0)        # ignore
            else:
                tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    n_valid = sum(1 for g in gts if not g.get("occluded", False))
    return tp, fp, n_valid


def average_precision(recalls, precisions):
    """COCO 101-point interpolated AP."""
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precisions[recalls >= r].max() if np.any(recalls >= r) else 0.0
        ap += p / 101
    return float(ap)


def evaluate_detections(per_frame_dets, per_frame_gts, classes,
                        iou_thresholds=(0.5,)):
    """
    per_frame_dets: list over frames of list[(cls, bbox, conf)]
    per_frame_gts:  list over frames of list[dict(cls,bbox,occluded)]
    Returns {iou_thr: {cls: dict(ap, n_gt, precision, recall, pr_curve)}}
    """
    results = {}
    for thr in iou_thresholds:
        per_cls = {}
        for cls in classes:
            recs = []          # (conf, tp, fp)
            n_gt = 0
            for dets, gts in zip(per_frame_dets, per_frame_gts):
                d = sorted([x for x in dets if x[0] == cls], key=lambda x: -x[2])
                g = [x for x in gts if x["cls"] == cls]
                tp, fp, nv = match_frame(d, g, thr)
                n_gt += nv
                for (c, b, cf), t, f in zip(d, tp, fp):
                    recs.append((cf, t, f))
            if n_gt == 0:
                continue
            recs.sort(key=lambda r: -r[0])
            tps = np.cumsum([r[1] for r in recs]) if recs else np.array([0])
            fps = np.cumsum([r[2] for r in recs]) if recs else np.array([0])
            rec = tps / n_gt
            prec = tps / np.maximum(tps + fps, 1e-9)
            ap = average_precision(rec, prec) if recs else 0.0
            per_cls[cls] = dict(
                ap=ap, n_gt=int(n_gt),
                precision=float(prec[-1]) if recs else 0.0,
                recall=float(rec[-1]) if recs else 0.0,
                pr_curve=(rec.tolist(), prec.tolist(),
                          [r[0] for r in recs]),
            )
        results[thr] = per_cls
    return results


def map_summary(results, classes):
    out = {}
    for thr, per_cls in results.items():
        aps = [per_cls[c]["ap"] for c in classes if c in per_cls]
        out[thr] = float(np.mean(aps)) if aps else 0.0
    return out
