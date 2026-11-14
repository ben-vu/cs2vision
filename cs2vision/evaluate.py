"""Frame- and round-level scoring against demo ground truth, plus statistics.

Per frame, predicted positions are matched to alive ground-truth players (per class
in "team" mode) by Hungarian assignment on Euclidean distance.

completeness   share of alive players matched within eval.presence_gate_m
fidelity@T     share of alive players matched within T metres
error          distance of matched pairs within the 5 m presence gate, metres
              (assignment itself searches through the largest fidelity threshold)
fraction of ceiling = round fidelity / round recoverable fraction (Experiment A rule)

Failure attribution for each player not placed within the headline threshold:
channel       player is unresolvable on the radar (merged with another blip)
detection     no raw detection of the right class within one blip diameter
localisation  a detection exists but its projected position is off by more than T
association   a good detection existed but the tracker output was not within T
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def match_distances(pred_xy, pred_cls, gt_xy, gt_cls, gate):
    """Distance from each GT player to its assigned prediction (inf if none within gate)."""
    out = np.full(len(gt_xy), np.inf)
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return out
    for c in np.unique(gt_cls):
        gi = np.where(gt_cls == c)[0]
        pi = np.where(pred_cls == c)[0]
        if len(pi) == 0:
            continue
        C = cdist(gt_xy[gi], pred_xy[pi])
        ri, ci = linear_sum_assignment(C)
        for i, j in zip(ri, ci):
            if C[i, j] <= gate:
                out[gi[i]] = C[i, j]
    return out


def attribute(dist_final, thr, unres, gt_frame, gt_cls, gt_canon, raw_frame, raw_cls,
              raw_canon, blip_px):
    labels = []
    for k in range(len(gt_frame)):
        if dist_final[k] <= thr:
            labels.append("ok")
            continue
        if unres[k]:
            labels.append("channel")
            continue
        sel = raw_cls == gt_cls[k] if len(raw_cls) else np.zeros(0, bool)
        if not sel.any():
            labels.append("detection")
            continue
        dfr = np.linalg.norm(raw_frame[sel] - gt_frame[k], axis=1)
        near = dfr <= blip_px
        if not near.any():
            labels.append("detection")
            continue
        dcan = np.linalg.norm(raw_canon[sel][near] - gt_canon[k], axis=1)
        labels.append("localisation" if dcan.min() > thr else "association")
    return labels


# statistics ------------------------------------------------------------------------------
def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    m, sd = x.mean(), x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return {"n": n, "mean": m, "sd": sd, "ci_lo": m - h, "ci_hi": m + h}


def paired(a, b, conf=0.95):
    """Paired comparison b - a over matched units (rounds). Cohen's d_z = mean/sd of diffs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = b - a
    n = len(d)
    if n < 2 or np.allclose(d, d[0]):
        return {"n": n, "mean_a": a.mean() if n else np.nan, "mean_b": b.mean() if n else np.nan,
                "diff": d.mean() if n else np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "t": np.nan, "p": np.nan, "dz": np.nan}
    t, pval = stats.ttest_rel(b, a)
    ci = mean_ci(d, conf)
    return {"n": n, "mean_a": a.mean(), "mean_b": b.mean(), "diff": d.mean(),
            "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "t": float(t), "p": float(pval),
            "dz": d.mean() / d.std(ddof=1)}


def fmt_ci(d, digits=3):
    return f"{d['mean']:.{digits}f} [{d['ci_lo']:.{digits}f}, {d['ci_hi']:.{digits}f}]"


def round_table(frames: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Aggregate frame rows to rounds (the unit of analysis)."""
    agg = {"n_alive": "sum", "n_present": "sum", "n_recoverable": "sum"}
    for t in thresholds:
        agg[f"n_fid_{t:g}m"] = "sum"
    g = frames.groupby(["match", "round"]).agg(agg).reset_index()
    g["completeness"] = g["n_present"] / g["n_alive"]
    g["recoverable"] = g["n_recoverable"] / g["n_alive"]
    for t in thresholds:
        g[f"fidelity_{t:g}m"] = g[f"n_fid_{t:g}m"] / g["n_alive"]
    h = thresholds[0]
    g["fraction_of_ceiling"] = g[f"fidelity_{h:g}m"] / g["recoverable"]
    return g
