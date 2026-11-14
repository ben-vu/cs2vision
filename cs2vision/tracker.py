"""Per-class Hungarian tracker with confidence decay (thesis Section 2.3.3).

Positions are canonical radar pixels. An unobserved track keeps its last position
(last-known-position imputation) while its confidence decays by `decay` per frame;
it is dropped once confidence falls below `min_conf`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(eq=False)
class Track:
    xy: np.ndarray
    cls: int
    conf: float
    age: int = 0
    missed: int = 0


class Tracker:
    def __init__(self, gate_px: float, decay=0.85, min_conf=0.30, max_per_class=5):
        self.gate, self.decay, self.min_conf, self.max_per_class = gate_px, decay, min_conf, \
            max_per_class
        self.tracks: list[Track] = []

    @classmethod
    def from_cfg(cls, cfg, meta):
        t = cfg["tracker"]
        return cls(meta.m_to_canon(t["gate_m"]), t["decay"], t["min_conf"],
                   t["max_tracks_per_class"])

    def reset(self):
        self.tracks = []

    def update(self, xy: np.ndarray, cls: np.ndarray, score: np.ndarray):
        xy = np.asarray(xy, float).reshape(-1, 2)
        for c in np.unique(np.r_[cls, [t.cls for t in self.tracks]]).astype(int):
            tr = [t for t in self.tracks if t.cls == c]
            di = np.where(cls == c)[0]
            di = di[np.argsort(-score[di])]
            used_t, used_d = set(), set()
            if tr and len(di):
                C = cdist(np.stack([t.xy for t in tr]), xy[di])
                ri, ci = linear_sum_assignment(C)
                for i, j in zip(ri, ci):
                    if C[i, j] <= self.gate:
                        t = tr[i]
                        t.xy, t.conf, t.missed = xy[di[j]].copy(), 1.0, 0
                        used_t.add(i)
                        used_d.add(j)
            for i, t in enumerate(tr):
                if i not in used_t:
                    t.conf *= self.decay
                    t.missed += 1
            n_alive = sum(1 for t in tr if t.conf >= self.min_conf)
            for j, d in enumerate(di):
                if j in used_d:
                    continue
                if n_alive >= self.max_per_class:
                    # replace the weakest coasting track rather than exceed the cap
                    weakest = min((t for t in tr if t.missed > 0), key=lambda t: t.conf,
                                  default=None)
                    if weakest is None:
                        continue
                    self.tracks.remove(weakest)
                    tr.remove(weakest)
                    n_alive -= 1
                nt = Track(xy[d].copy(), int(c), 1.0)
                self.tracks.append(nt)
                tr.append(nt)
                n_alive += 1
        self.tracks = [t for t in self.tracks if t.conf >= self.min_conf]
        for t in self.tracks:
            t.age += 1

    def state(self):
        if not self.tracks:
            return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
        return (np.stack([t.xy for t in self.tracks]), np.array([t.cls for t in self.tracks]),
                np.array([t.conf for t in self.tracks]))
