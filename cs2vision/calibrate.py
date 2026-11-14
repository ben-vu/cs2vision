"""Per-overlay radar calibration (replaces the pilot's synthetic fiducials).

Step 1 (by hand, once per production): draw a box around the full radar image and
around one blip. This gives a rough rectangle transform and the blip diameter.
Step 2 (automatic, `cs2v align`): the rectangle is refined against demo positions.
"""
from __future__ import annotations

import cv2
import numpy as np

from .radar import RadarCalib
from .video import VideoReader, parse_time


def _select(frame, title):
    print(f"-> {title}. Drag a box, then press ENTER/SPACE (c to cancel).")
    r = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if r[2] == 0 or r[3] == 0:
        raise SystemExit("selection cancelled")
    return r


def calibrate(match, time, rect=None, blip_px=None, copy_from=None):
    cfg = match.cfg
    reader = VideoReader(match.video)
    frame, _ = reader.read_at(parse_time(time))
    reader.close()
    if frame is None:
        raise SystemExit(f"could not read a frame at {time}")
    fh, fw = frame.shape[:2]

    if copy_from is not None:
        calib = copy_from.calib()
        if tuple(calib.frame_size) != (fw, fh):
            raise SystemExit("source calibration was made at a different video resolution")
    else:
        if rect is None:
            x, y, w, h = _select(frame, "Box the WHOLE radar map image (its full extent)")
        else:
            x, y, w, h = rect
        if blip_px is None:
            x2, y2, w2, h2 = _select(frame, "Box ONE player blip tightly")
            blip_px = (w2 + h2) / 2.0
        calib = RadarCalib.from_rect(x, y, w, h, blip_px, cfg["map"]["radar_size"],
                                     (fw, fh), cfg["video"]["crop_pad"])
        calib.notes["initial_rect"] = [int(x), int(y), int(w), int(h)]
    calib.save(match.calib_path)
    crop = calib.crop(frame)
    cv2.imwrite(str(match.ref_path), crop)

    preview = frame.copy()
    s = calib.radar_size
    for k in np.linspace(0, s, 9):
        a = calib.canon_to_frame([[k, 0], [k, s]]).astype(int)
        b = calib.canon_to_frame([[0, k], [s, k]]).astype(int)
        cv2.line(preview, tuple(a[0]), tuple(a[1]), (0, 255, 0), 1)
        cv2.line(preview, tuple(b[0]), tuple(b[1]), (0, 255, 0), 1)
    cv2.imwrite(str(match.dir / "calib_preview.png"), preview)
    print(f"saved {match.calib_path.name}, {match.ref_path.name}, calib_preview.png "
          f"(blip {calib.blip_px:.1f}px, radar scale {calib.frame_px_per_canon:.4f} "
          f"frame px per canonical px)")
    return calib
