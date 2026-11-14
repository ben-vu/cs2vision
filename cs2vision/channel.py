"""The broadcast radar channel ceiling (Experiment A, RQ1).

Needs only demo ground truth plus two numbers from one calibrated broadcast:
the radar's size on screen (frame px per canonical px) and the blip diameter.

precision floor   world units (and metres) covered by one broadcast radar pixel
unresolvable      an alive player whose blip centre is within one blip diameter of
                  another alive player's: the two render as one mark
recoverable frac  1 - unresolvable / alive, per sampled tick
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .groundtruth import team_to_class
from .units import units_to_m

SENSITIVITY = (0.5, 0.75, 1.0, 1.25)   # merge distance as a multiple of blip diameter


def unresolvable(frame_xy: np.ndarray, blip_px: float, cls=None, same_class_only=False):
    """Boolean mask of players that collapse into another player's blip."""
    n = len(frame_xy)
    if n < 2:
        return np.zeros(n, bool)
    d = cdist(frame_xy, frame_xy)
    np.fill_diagonal(d, np.inf)
    close = d < blip_px
    if same_class_only and cls is not None:
        close &= (np.asarray(cls)[:, None] == np.asarray(cls)[None, :])
    return close.any(1)


def phase_of(secs: float, post_plant: bool, phases: dict) -> str:
    if post_plant:
        return "post_plant"
    for name, (a, b) in phases.items():
        if a <= secs < b:
            return name
    return list(phases)[-1]


def precision_floor(map_scale: float, frame_px_per_canon: float) -> dict:
    wu = map_scale / frame_px_per_canon
    return {"world_units_per_px": wu, "metres_per_px": float(units_to_m(wu))}


def ceiling_from_gt(gt: pd.DataFrame, rounds: pd.DataFrame, frame_px_per_canon: float,
                    blip_px: float, cfg: dict, step_ticks: int = 16, match_id="") -> pd.DataFrame:
    """Per-sample recoverable fraction over the live part of every round."""
    tr = cfg["demo"]["tickrate"]
    gt = gt[gt["alive"]]
    by_tick = {t: g for t, g in gt.groupby("tick")}
    rows = []
    for _, r in rounds.iterrows():
        for tk in range(int(r["freeze_end_tick"]), int(r["end_tick"]) + 1, step_ticks):
            g = by_tick.get(tk)
            if g is None or len(g) == 0:
                continue
            xy = g[["cx", "cy"]].to_numpy(float) * frame_px_per_canon
            cls = team_to_class(g["team_num"].to_numpy(), cfg["classes"])
            merge = blip_px * cfg["eval"].get("merge_frac", 1.0)
            u_any = unresolvable(xy, merge)
            u_same = unresolvable(xy, merge, cls, same_class_only=True)
            post = bool(r["plant_tick"] >= 0 and tk >= r["plant_tick"])
            secs = (tk - r["freeze_end_tick"]) / tr
            sens = {f"recoverable_merge_{f:g}": 1 - unresolvable(xy, blip_px * f).mean()
                    for f in SENSITIVITY}
            rows.append({"match": match_id, "round": int(r["round"]), "tick": tk,
                         "secs_into_round": secs,
                         "phase": phase_of(secs, post, cfg["eval"]["phases"]),
                         "n_alive": len(g), "n_unresolvable": int(u_any.sum()),
                         "recoverable": 1 - u_any.mean(),
                         "recoverable_same_team_rule": 1 - u_same.mean(), **sens})
    return pd.DataFrame(rows)
