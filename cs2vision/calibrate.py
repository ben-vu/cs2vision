"""
Assisted radar-rectangle calibration (thesis Sec 2.3.2 setup step).

The radar->map transform needs the on-screen radar widget's pixel rectangle.
It is fixed per overlay template, so you calibrate it ONCE per tournament
production style. This module provides:

  * interactive_calibrate(image)  -- click the radar's top-left then
    bottom-right corner in a window; returns a RadarRect. (Needs a desktop /
    GUI OpenCV build; run on your machine, not a headless server.)

  * rect_from_corners(tl, br)     -- the pure core (no GUI), unit-tested.

  * auto_suggest_rect(frame)      -- a rough heuristic starting point: the
    brightest near-square panel in a frame corner. Treat as a suggestion to
    nudge, not a final answer.

  * verification_overlay(...)     -- draw the calibrated rectangle and a map
    grid back-projected into it, so you can SEE whether the calibration is
    right before committing.

  * overlay_gt_on_radar(...)      -- the strongest check: inverse-project demo
    ground-truth player positions onto the frame; the dots must land on the
    real radar blips. If they do, your RadarRect + RadarMeta are consistent.
"""
from __future__ import annotations

import cv2
import numpy as np

from .homography import RadarRect


def rect_from_corners(tl, br, map_image_size: float = 1024.0) -> RadarRect:
    x1, y1 = int(round(tl[0])), int(round(tl[1]))
    x2, y2 = int(round(br[0])), int(round(br[1]))
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    if w < 8 or h < 8:
        raise ValueError("radar rectangle too small; click two opposite corners")
    return RadarRect(x=x, y=y, w=w, h=h, map_image_size=map_image_size)


def auto_suggest_rect(frame: np.ndarray, map_image_size: float = 1024.0):
    """Rough heuristic: brightest near-square blob in a frame corner.
    Returns a RadarRect suggestion or None. Always eyeball with
    verification_overlay before trusting it."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    best = None
    for (qx, qy) in [(0, 0), (W // 2, 0), (0, H // 2), (W // 2, H // 2)]:
        roi = gray[qy:qy + H // 2, qx:qx + W // 2]
        _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 80 or h < 80:
                continue
            ar = w / h
            if not (0.8 <= ar <= 1.25):
                continue
            score = w * h
            if best is None or score > best[0]:
                best = (score, RadarRect(qx + x, qy + y, w, h, map_image_size))
    return best[1] if best else None


def interactive_calibrate(image, map_image_size: float = 1024.0) -> RadarRect:
    """Click top-left then bottom-right corner of the radar. GUI required."""
    img = cv2.imread(image) if isinstance(image, str) else image.copy()
    pts = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("calibrate radar (click TL then BR, then any key)", img)

    cv2.namedWindow("calibrate radar (click TL then BR, then any key)")
    cv2.setMouseCallback("calibrate radar (click TL then BR, then any key)", on_mouse)
    cv2.imshow("calibrate radar (click TL then BR, then any key)", img)
    while len(pts) < 2:
        if cv2.waitKey(20) == 27:        # ESC aborts
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(pts) < 2:
        raise RuntimeError("calibration aborted")
    return rect_from_corners(pts[0], pts[1], map_image_size)


def verification_overlay(frame: np.ndarray, rect: RadarRect, n_grid: int = 8):
    """Draw the rectangle + a back-projected map grid to eyeball the scale."""
    img = frame.copy()
    cv2.rectangle(img, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h),
                  (0, 255, 0), 2)
    s = rect.w / rect.map_image_size       # map px -> crop px
    for i in range(n_grid + 1):
        m = i * rect.map_image_size / n_grid
        px = int(rect.x + m * s)
        py = int(rect.y + m * s)
        cv2.line(img, (px, rect.y), (px, rect.y + rect.h), (0, 180, 0), 1)
        cv2.line(img, (rect.x, py), (rect.x + rect.w, py), (0, 180, 0), 1)
    return img


def overlay_gt_on_radar(frame: np.ndarray, rect: RadarRect, gt_players: list[dict]):
    """Inverse-project demo GT map positions onto the frame radar. Dots should
    land on the real blips if RadarRect + RadarMeta are consistent."""
    img = frame.copy()
    s = rect.w / rect.map_image_size
    for p in gt_players:
        if not p.get("alive", True) or "map_pos" not in p:
            continue
        mx, my = p["map_pos"]
        fx = int(rect.x + mx * s)
        fy = int(rect.y + my * s)
        col = (235, 180, 80) if p.get("team") == "CT" else (80, 160, 245)
        cv2.circle(img, (fx, fy), 4, col, 1)
        cv2.drawMarker(img, (fx, fy), col, cv2.MARKER_CROSS, 8, 1)
    cv2.rectangle(img, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h),
                  (0, 255, 0), 1)
    return img
