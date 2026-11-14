"""Frame-to-tick synchronisation (thesis Section 2.2.2, RO1).

1. Prediction: each round's start (freeze end) in the video is predicted from the
   nearest hand-marked anchor at or before it in anchors.csv, assuming the broadcast
   runs in real time (1 s of video = tickrate ticks). One anchor is enough for an
   uncut VOD; add an anchor after every cut, pause-skip or ad break in an edited VOD.
2. Per-round refinement: the start time of each round is searched within
   +/- sync.search_s of the global prediction by minimising the distance between
   blips found on the radar and demo positions projected through the calibration.
3. Residuals: refined minus predicted start (s) is the drift of the global model,
   and the matching error late in each round checks that the within-round mapping
   stays linear. Both are written to sync.csv and summarised for the thesis.

The calibration refinement (refine_calibration) lives here too because the two are
solved alternately by `cs2v align`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .blobs import BlobDetector
from .radar import RadarCalib, fit_transform, radar_ncc
from .video import VideoReader, parse_time


def tick_at(row, t: float, tickrate: int) -> float:
    """Demo tick shown at video time t, within the round described by a sync row."""
    return row["freeze_end_tick"] + tickrate * (t - row["start_time"])


def predict_starts(anchors: pd.DataFrame, rounds: pd.DataFrame, tickrate: int) -> np.ndarray:
    """Predicted video time of every round's freeze end (piecewise, real-time slope)."""
    a = anchors.merge(rounds[["round", "freeze_end_tick"]], on="round", how="inner")
    if a.empty:
        raise ValueError("anchors.csv rounds do not match any round in rounds.csv")
    a = a.sort_values("round")
    a_round = a["round"].to_numpy()
    a_tick = a["freeze_end_tick"].to_numpy(float)
    a_time = a["video_time"].map(parse_time).to_numpy(float)
    preds = []
    for rnd, tick in zip(rounds["round"], rounds["freeze_end_tick"]):
        i = np.searchsorted(a_round, rnd, side="right") - 1
        i = max(i, 0)
        preds.append(a_time[i] + (tick - a_tick[i]) / tickrate)
    return np.array(preds)


def chamfer(a: np.ndarray, b: np.ndarray, cap: float) -> float:
    """Symmetric truncated chamfer distance between point sets (px)."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return cap
    d = np.minimum(cdist(a, b), cap)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def _observe(reader, calib, blob, ref, times, min_ncc):
    """Blob centres (frame px) at each time; frames with a hidden radar are skipped."""
    x0, y0, _, _ = calib.crop_box()
    obs = []
    for t in times:
        frame, t_exact = reader.read_at(t)
        if frame is None:
            continue
        crop = calib.crop(frame)
        if ref is not None and radar_ncc(crop, ref) < min_ncc:
            continue
        det = blob.detect(crop)
        obs.append((t_exact, det.xy + [x0, y0]))
    return obs


def _cost(obs, gt, calib, freeze_tick, start, tickrate, cap):
    costs = []
    for t, xy in obs:
        pts, _, _ = gt.at(freeze_tick + tickrate * (t - start))
        if len(pts) == 0:
            continue
        costs.append(chamfer(calib.canon_to_frame(pts), xy, cap))
    return float(np.mean(costs)) if costs else np.inf


def sync_match(match, calib: RadarCalib | None = None, verbose=True) -> pd.DataFrame:
    cfg = match.cfg
    sc, tr = cfg["sync"], cfg["demo"]["tickrate"]
    calib = calib or match.calib()
    rounds, gt, ref = match.rounds(), match.gt(), match.ref()
    anchors = pd.read_csv(match.anchors_path)
    preds = predict_starts(anchors, rounds, tr)
    blob = BlobDetector.from_cfg(cfg, calib.blip_px)
    reader = VideoReader(match.video)
    cap = 2.5 * calib.blip_px
    step = sc["step_ticks"] / tr
    rows = []
    for (_, r), pred in zip(rounds.iterrows(), preds):
        dur = (r["end_tick"] - r["freeze_end_tick"]) / tr
        if pred < 0 or pred + dur > reader.duration:
            continue
        w0, w1 = sc["window_s"]
        w1 = min(w1, dur)
        times = np.linspace(pred + w0, pred + w1, sc["frames_per_round"])
        obs = _observe(reader, calib, blob, ref, times, cfg["video"]["radar_visible_min_ncc"])
        if len(obs) < 3:
            rows.append({**r.to_dict(), "pred_time": pred, "start_time": pred,
                         "residual_s": np.nan, "match_err_px": np.nan, "late_err_px": np.nan,
                         "n_frames": len(obs), "ok": False})
            continue
        cands = np.arange(pred - sc["search_s"], pred + sc["search_s"] + 1e-9, step)
        costs = np.array([_cost(obs, gt, calib, r["freeze_end_tick"], s, tr, cap)
                          for s in cands])
        best = cands[int(np.argmin(costs))]
        err = float(costs.min())
        # drift check: matching error late in the round, using the refined start
        late = np.nan
        l0 = min(max(w1, 0.6 * dur), dur - 2.0)
        if l0 > w0:
            late_t = np.linspace(best + l0, best + dur - 1.0, 6)
            late_obs = _observe(reader, calib, blob, ref, late_t,
                                cfg["video"]["radar_visible_min_ncc"])
            late = _cost(late_obs, gt, calib, r["freeze_end_tick"], best, tr, cap)
            late = late if np.isfinite(late) else np.nan
        rows.append({**r.to_dict(), "pred_time": pred, "start_time": float(best),
                     "residual_s": float(best - pred), "match_err_px": err,
                     "late_err_px": late, "n_frames": len(obs),
                     "ok": bool(err <= sc["max_residual_px"])})
        if verbose:
            print(f"  round {int(r['round']):>2}: start {best:9.2f}s  residual "
                  f"{best - pred:+.3f}s  match err {err:.2f}px  late {late:.2f}px")
    reader.close()
    out = pd.DataFrame(rows)
    out.to_csv(match.sync_path, index=False)
    return out


def refine_calibration(match, calib: RadarCalib, sync: pd.DataFrame, frames_per_round=8,
                       model="affine", verbose=True) -> RadarCalib:
    """Fit canonical->frame transform from demo positions vs. detected blips."""
    cfg = match.cfg
    tr = cfg["demo"]["tickrate"]
    gt, ref = match.gt(), match.ref()
    blob = BlobDetector.from_cfg(cfg, calib.blip_px)
    reader = VideoReader(match.video)
    A, B = [], []
    for _, r in sync[sync["n_frames"] > 0].iterrows():
        dur = (r["end_tick"] - r["freeze_end_tick"]) / tr
        times = np.linspace(r["start_time"] + 3, r["start_time"] + dur - 1, frames_per_round)
        for t, xy in _observe(reader, calib, blob, ref, times,
                              cfg["video"]["radar_visible_min_ncc"]):
            pts, _, _ = gt.at(tick_at(r, t, tr))
            if len(pts) == 0 or len(xy) == 0:
                continue
            pf = calib.canon_to_frame(pts)
            # only use players that are well separated from everyone else
            dd = cdist(pf, pf) + np.eye(len(pf)) * 1e9
            sep = dd.min(1) > 2 * calib.blip_px if len(pf) > 1 else np.ones(1, bool)
            C = cdist(pf, xy)
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if sep[i] and C[i, j] <= 1.5 * calib.blip_px:
                    A.append(pts[i])
                    B.append(xy[j])
    reader.close()
    if len(A) < 20:
        if verbose:
            print(f"  calibration refinement skipped: only {len(A)} clean pairs")
        return calib
    H, mask, rms = fit_transform(np.array(A), np.array(B), model=model)
    new = RadarCalib(H, calib.blip_px, calib.radar_size, calib.frame_size, calib.pad,
                     rms, int(mask.sum()), dict(calib.notes, refined=model))
    if verbose:
        print(f"  calibration refined from {mask.sum()}/{len(A)} inlier pairs, "
              f"RMS {rms:.2f}px, scale {new.frame_px_per_canon:.4f} frame px/canon px")
    return new


def align(match, iterations: int = 2, model="affine", verbose=True):
    """Alternate sync and calibration refinement; writes sync.csv and calib.json."""
    calib = match.calib()
    for it in range(iterations):
        if verbose:
            print(f"[{match.id}] iteration {it + 1}: sync")
        sync = sync_match(match, calib, verbose=verbose)
        if verbose:
            print(f"[{match.id}] iteration {it + 1}: calibration")
        calib = refine_calibration(match, calib, sync, model=model, verbose=verbose)
        calib.save(match.calib_path)
    if verbose:
        print(f"[{match.id}] final sync")
    sync = sync_match(match, calib, verbose=verbose)
    return calib, sync


def sync_summary(sync: pd.DataFrame) -> dict:
    ok = sync[sync["ok"]]
    return {
        "rounds": int(len(sync)), "rounds_ok": int(len(ok)),
        "residual_s_mean_abs": float(ok["residual_s"].abs().mean()) if len(ok) else np.nan,
        "residual_s_max_abs": float(ok["residual_s"].abs().max()) if len(ok) else np.nan,
        "match_err_px_median": float(ok["match_err_px"].median()) if len(ok) else np.nan,
        "late_err_px_median": float(ok["late_err_px"].median()) if len(ok) else np.nan,
    }
