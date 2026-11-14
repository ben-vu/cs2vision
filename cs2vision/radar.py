"""Radar geometry: canonical radar pixels <-> broadcast frame pixels (thesis Stage 2).

The observer radar on a production with a static, full-map radar occupies a fixed
region of the frame, so one transform per overlay style is enough. The transform is
stored as a 3x3 matrix H mapping canonical pixels to frame pixels. It starts as a
rectangle (scale + offset) drawn by hand and is then refined by fitting an affine map
to demo-projected positions vs. detected blips with RANSAC (3 px inlier threshold).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


@dataclass
class RadarCalib:
    H: np.ndarray                     # canonical px -> frame px
    blip_px: float                    # blip diameter in frame px
    radar_size: int = 1024
    frame_size: tuple = (1920, 1080)  # (w, h)
    pad: int = 6
    rms_px: float | None = None       # residual of the last refinement
    n_pairs: int = 0
    notes: dict = field(default_factory=dict)

    # construction -----------------------------------------------------------------
    @classmethod
    def from_rect(cls, x, y, w, h, blip_px, radar_size=1024, frame_size=(1920, 1080), pad=6):
        s = radar_size
        H = np.array([[w / s, 0, x], [0, h / s, y], [0, 0, 1]], dtype=float)
        return cls(H, float(blip_px), radar_size, tuple(frame_size), pad)

    # transforms ---------------------------------------------------------------------
    def canon_to_frame(self, pts):
        return apply_h(self.H, pts)

    def frame_to_canon(self, pts):
        return apply_h(np.linalg.inv(self.H), pts)

    @property
    def frame_px_per_canon(self) -> float:
        """Linear scale of the radar inset (frame px per canonical px)."""
        return float(np.sqrt(abs(np.linalg.det(self.H[:2, :2]))))

    def world_units_per_frame_px(self, map_scale: float) -> float:
        """Precision floor: world units spanned by one broadcast radar pixel."""
        return map_scale / self.frame_px_per_canon

    def crop_box(self) -> tuple[int, int, int, int]:
        """Integer (x0, y0, x1, y1) of the radar inset in the frame, padded and clipped."""
        s = self.radar_size
        corners = self.canon_to_frame(np.array([[0, 0], [s, 0], [s, s], [0, s]]))
        x0, y0 = np.floor(corners.min(0)).astype(int) - self.pad
        x1, y1 = np.ceil(corners.max(0)).astype(int) + self.pad
        W, Hh = self.frame_size
        return max(0, x0), max(0, y0), min(W, x1), min(Hh, y1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.crop_box()
        return frame[y0:y1, x0:x1]

    # persistence --------------------------------------------------------------------
    def save(self, path: str | Path):
        d = {"H": self.H.tolist(), "blip_px": self.blip_px, "radar_size": self.radar_size,
             "frame_size": list(self.frame_size), "pad": self.pad, "rms_px": self.rms_px,
             "n_pairs": self.n_pairs, "notes": self.notes}
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RadarCalib":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["H"], float), float(d["blip_px"]), int(d["radar_size"]),
                   tuple(d["frame_size"]), int(d.get("pad", 6)), d.get("rms_px"),
                   int(d.get("n_pairs", 0)), d.get("notes", {}))


def fit_transform(canon_pts: np.ndarray, frame_pts: np.ndarray, model: str = "affine",
                  thresh_px: float = 3.0, seed: int = 0):
    """Robustly fit canonical->frame transform. Returns (H 3x3, inlier mask, rms_px)."""
    canon_pts = np.asarray(canon_pts, np.float32).reshape(-1, 2)
    frame_pts = np.asarray(frame_pts, np.float32).reshape(-1, 2)
    cv2.setRNGSeed(seed)
    if model == "homography":
        H, mask = cv2.findHomography(canon_pts, frame_pts, cv2.RANSAC, thresh_px,
                                     maxIters=5000)
    else:
        A, mask = cv2.estimateAffine2D(canon_pts, frame_pts, method=cv2.RANSAC,
                                       ransacReprojThreshold=thresh_px, maxIters=5000)
        H = None if A is None else np.vstack([A, [0, 0, 1]])
    if H is None:
        raise RuntimeError("transform fit failed (too few consistent pairs)")
    mask = mask.ravel().astype(bool)
    res = np.linalg.norm(apply_h(H, canon_pts[mask]) - frame_pts[mask], axis=1)
    return H, mask, float(np.sqrt(np.mean(res ** 2)))


def _prep(img: np.ndarray, size=(128, 128)) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)


def radar_ncc(crop: np.ndarray, ref: np.ndarray) -> float:
    """Normalised cross-correlation between a radar crop and the reference crop.

    Low values mean the radar is hidden or covered (replay, scoreboard, transition).
    """
    if crop.size == 0 or ref is None:
        return 0.0
    a, b = _prep(crop), _prep(ref)
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])
