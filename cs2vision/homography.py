"""
Radar-anchored geometric reconstruction for REAL CS2 broadcasts.

The simulator drew bright cyan cross-markers at known map fractions and the
old homography.py template-matched them. Real CS2 radars have no such markers,
so that approach does not transfer. This module replaces it with two real
options, both producing a transform  radar-crop-pixels -> canonical-map-pixels
(the same RadarMeta image space the demo ground truth lives in):

  (A) CalibratedRadar  -- the spectator HUD radar is the *static full-map*
      radar at a fixed screen rectangle. You calibrate that rectangle ONCE per
      overlay template (helpers below). Because the on-screen radar is just a
      scaled copy of the canonical radar image, the transform is a fixed
      similarity (translation + uniform scale) you get for free from the
      calibrated rectangle. Fast, exact, no per-frame estimation. This is the
      recommended path and matches how pro spectator HUDs render the radar.

  (B) RegisteredRadar  -- when the radar pans/zooms (some HUDs follow the
      spectated player), the transform changes per frame. This estimates it by
      image-registering the live radar crop to the canonical radar image with
      ECC (cv2.findTransformECC, affine). Robust to brightness but needs the
      canonical de_<map>_radar.png. Falls back to the previous frame on failure.

Both reuse the temporal-consistency constraint from the original design
(Pham et al. 2026): implausible inter-frame jumps are blended with the
previous estimate. The constraint logic is unchanged; only the anchor source
is real now.

You supply: the radar screen rectangle (calibrate once) and, for (B), the
canonical radar image. Nothing here is invented from the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_TRANS_PX = 6.0       # canonical-map pixels
MAX_ROT_DEG = 2.0
BLEND = 0.35


@dataclass
class RadarRect:
    """Calibrated on-screen radar panel (one per overlay template)."""
    x: int
    y: int
    w: int
    h: int
    map_image_size: float = 1024.0    # canonical radar image is square

    def similarity_H(self) -> np.ndarray:
        """radar-crop-pixel (relative to crop top-left) -> canonical map px."""
        s = self.map_image_size / self.w        # assume square radar widget
        return np.array([[s, 0, 0],
                         [0, s, 0],
                         [0, 0, 1]], float)


class CalibratedRadar:
    """Option A: fixed similarity from a calibrated radar rectangle."""

    def __init__(self, rect: RadarRect):
        self.rect = rect
        self.H = rect.similarity_H()

    def estimate(self, frame: np.ndarray):
        # static: same transform every frame
        return self.H, dict(method="calibrated", ok=True)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        r = self.rect
        return frame[r.y:r.y + r.h, r.x:r.x + r.w]


class RegisteredRadar:
    """Option B: per-frame ECC registration of the radar crop to the canonical
    radar image, with temporal-consistency smoothing."""

    def __init__(self, rect: RadarRect, canonical_radar_png: str,
                 temporal: bool = True):
        self.rect = rect
        canon = cv2.imread(canonical_radar_png, cv2.IMREAD_GRAYSCALE)
        if canon is None:
            raise RuntimeError(f"cannot read canonical radar {canonical_radar_png}")
        self.canon = cv2.resize(canon, (int(rect.map_image_size),) * 2)
        self.temporal = temporal
        self.H_prev: np.ndarray | None = None
        self.n_frames = 0
        self.n_rejected = 0

    def crop(self, frame: np.ndarray) -> np.ndarray:
        r = self.rect
        return frame[r.y:r.y + r.h, r.x:r.x + r.w]

    def estimate(self, frame: np.ndarray):
        self.n_frames += 1
        diag = dict(method="ecc", ok=False, fallback=False, rejected=False)
        crop = self.crop(frame)
        if crop.size == 0:
            return self.H_prev, dict(diag, fallback=True)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        # bring crop to canonical size so warp is ~identity at start
        gray = cv2.resize(gray, (self.canon.shape[1], self.canon.shape[0]))
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp = cv2.findTransformECC(self.canon, gray, warp,
                                           cv2.MOTION_AFFINE, crit, None, 5)
            diag["ok"] = True
        except cv2.error:
            return self._fallback(diag)
        # warp maps canonical->crop-resized; invert to crop-resized->canonical
        Wsq = np.vstack([warp, [0, 0, 1]])
        try:
            Winv = np.linalg.inv(Wsq)
        except np.linalg.LinAlgError:
            return self._fallback(diag)
        # account for the resize crop->canonical-size done above
        sx = self.canon.shape[1] / max(self.rect.w, 1)
        sy = self.canon.shape[0] / max(self.rect.h, 1)
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)
        H = Winv @ S
        if self.temporal:
            H = self._temporal(H)
        self.H_prev = H
        return H, diag

    def _fallback(self, diag):
        diag["fallback"] = True
        return self.H_prev, diag

    def _temporal(self, H):
        if self.H_prev is None:
            return H
        dt = float(np.linalg.norm(H[:2, 2] - self.H_prev[:2, 2]))
        dr = abs(np.degrees(np.arctan2(H[1, 0], H[0, 0]))
                 - np.degrees(np.arctan2(self.H_prev[1, 0], self.H_prev[0, 0])))
        if dt > MAX_TRANS_PX or dr > MAX_ROT_DEG:
            self.n_rejected += 1
            return BLEND * H + (1 - BLEND) * self.H_prev
        return H


def project(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Project crop-relative pixel points through H to canonical map pixels."""
    if len(pts_px) == 0:
        return np.empty((0, 2))
    pts = cv2.perspectiveTransform(
        np.asarray(pts_px, np.float32).reshape(-1, 1, 2), H)
    return pts.reshape(-1, 2)
