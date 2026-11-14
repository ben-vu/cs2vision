"""Classical radar-blip detector (saturation/value threshold + connected components).

Used for three things:
  * automatic sync refinement and calibration refinement (team-agnostic),
  * the automatic label check in autolabel.py,
  * the non-learned baseline in Experiment C (with per-class colour prototypes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detections:
    xy: np.ndarray            # [N, 2] centre in the image the detector saw
    cls: np.ndarray           # [N] class id (-1 = unknown)
    score: np.ndarray         # [N]

    @staticmethod
    def empty():
        return Detections(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0))

    def __len__(self):
        return len(self.xy)


def _features(hsv: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    m = labels == k
    h = hsv[..., 0][m].astype(float) * (2 * np.pi / 180.0)
    s = hsv[..., 1][m].astype(float) / 255.0
    v = hsv[..., 2][m].astype(float) / 255.0
    return np.array([np.cos(h).mean() * s.mean(), np.sin(h).mean() * s.mean(), v.mean()])


@dataclass
class BlobDetector:
    blip_px: float
    min_saturation: int = 90
    min_value: int = 110
    area_min_frac: float = 0.25
    area_max_frac: float = 3.0
    prototypes: dict = field(default_factory=dict)   # class id -> feature vector

    @classmethod
    def from_cfg(cls, cfg: dict, blip_px: float) -> "BlobDetector":
        b = cfg["blob"]
        return cls(blip_px, b["min_saturation"], b["min_value"], b["area_min_frac"],
                   b["area_max_frac"])

    def _components(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 1] >= self.min_saturation) & (hsv[..., 2] >= self.min_value))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        nominal = np.pi * (self.blip_px / 2) ** 2
        keep = [k for k in range(1, n)
                if self.area_min_frac * nominal <= stats[k, cv2.CC_STAT_AREA]
                <= self.area_max_frac * nominal]
        return hsv, labels, stats, cents, keep

    def detect(self, img: np.ndarray) -> Detections:
        hsv, labels, stats, cents, keep = self._components(img)
        if not keep:
            return Detections.empty()
        nominal = np.pi * (self.blip_px / 2) ** 2
        xy = cents[keep]
        area = stats[keep, cv2.CC_STAT_AREA].astype(float)
        score = np.clip(1.0 - np.abs(np.log(area / nominal)) / 2.0, 0.05, 1.0)
        cls = np.full(len(keep), -1, int)
        if self.prototypes:
            ids = sorted(self.prototypes)
            P = np.stack([self.prototypes[i] for i in ids])
            for j, k in enumerate(keep):
                f = _features(hsv, labels, k)
                cls[j] = ids[int(np.argmin(np.linalg.norm(P - f, axis=1)))]
        return Detections(xy, cls, score)

    def fit_colours(self, samples):
        """samples: iterable of (crop_bgr, gt_xy_in_crop [N,2], gt_cls [N]).

        Learns one colour prototype per class from the blob nearest each GT player.
        """
        feats: dict[int, list] = {}
        for img, gxy, gcls in samples:
            hsv, labels, stats, cents, keep = self._components(img)
            if not keep:
                continue
            c = cents[keep]
            for (x, y), k in zip(gxy, gcls):
                d = np.linalg.norm(c - [x, y], axis=1)
                j = int(np.argmin(d))
                if d[j] <= self.blip_px * 0.6:
                    feats.setdefault(int(k), []).append(_features(hsv, labels, keep[j]))
        self.prototypes = {k: np.mean(v, 0) for k, v in feats.items() if len(v) >= 5}
        return {k: len(v) for k, v in feats.items()}

    def save(self, path):
        d = {k: getattr(self, k) for k in ("blip_px", "min_saturation", "min_value",
                                           "area_min_frac", "area_max_frac")}
        d["prototypes"] = {str(k): v.tolist() for k, v in self.prototypes.items()}
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path) -> "BlobDetector":
        d = json.loads(Path(path).read_text())
        protos = {int(k): np.array(v) for k, v in d.pop("prototypes", {}).items()}
        return cls(**d, prototypes=protos)
