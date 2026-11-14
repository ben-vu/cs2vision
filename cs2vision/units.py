"""Coordinate systems (thesis Appendix C).

world   : demo coordinates, Source units (1 u = 0.75 in = 0.01905 m)
canon   : pixels of the 1024x1024 canonical radar image, origin top-left, y down
frame   : pixels of the broadcast video frame
metres  : used for every reported threshold
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METRES_PER_UNIT = 0.01905
UNITS_PER_METRE = 1.0 / METRES_PER_UNIT  # ~52.49


def units_to_m(u):
    return np.asarray(u, dtype=float) * METRES_PER_UNIT


def m_to_units(m):
    return np.asarray(m, dtype=float) * UNITS_PER_METRE


@dataclass(frozen=True)
class MapMeta:
    name: str
    pos_x: float
    pos_y: float
    scale: float          # world units per canonical pixel
    radar_size: int = 1024

    @classmethod
    def from_cfg(cls, cfg: dict) -> "MapMeta":
        m = cfg["map"]
        return cls(m["name"], float(m["pos_x"]), float(m["pos_y"]), float(m["scale"]),
                   int(m.get("radar_size", 1024)))

    def world_to_canon(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return (x - self.pos_x) / self.scale, (self.pos_y - y) / self.scale

    def canon_to_world(self, px, py):
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        return px * self.scale + self.pos_x, self.pos_y - py * self.scale

    def m_to_canon(self, metres) -> float:
        """Metres -> canonical pixels on this map."""
        return float(m_to_units(metres)) / self.scale

    def canon_to_m(self, px):
        return units_to_m(np.asarray(px, dtype=float) * self.scale)
