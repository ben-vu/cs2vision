"""
Automatic annotation for DEMO-PLAYBACK mode (no CVAT, no manual labelling).

Because you possess the demo, every entity's position is known, so bounding
boxes can be generated programmatically by projecting demo positions into the
rendered frame. Two projectors:

  * radar blips  -- EXACT and robust. Project the player's canonical map
    position (already computed by groundtruth via RadarMeta) to the on-screen
    radar via the calibrated RadarRect. This is the inverse of
    calibrate.overlay_gt_on_radar and needs no camera model. Use this as your
    primary free-label source -- it directly trains the geometry pipeline.

  * world-view players -- project world coords through the spectated player's
    camera (eye position + yaw/pitch from the demo) with a Source-style pinhole
    model. EXPERIMENTAL: the spectator camera has an offset/FOV that varies by
    playback mode, so calibrate `fov_deg` and the eye offset against a couple of
    rendered frames before trusting the boxes. Radar labels do not need this.

Output is YOLO 1.1 format (one .txt per frame, normalised cx,cy,w,h), so the
existing train_yolo.py / data.yaml path is unchanged -- the labels are just
machine-made instead of hand-made.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .classes import CLS_ID
from .homography import RadarRect


# --------------------------------------------------------------------------- #
def map_to_screen(map_pos, rect: RadarRect) -> tuple[float, float]:
    """Canonical map px -> on-screen radar px (inverse of the calibrated H)."""
    s = rect.w / rect.map_image_size
    return rect.x + map_pos[0] * s, rect.y + map_pos[1] * s


def radar_blip_labels(gt_players, rect: RadarRect, frame_wh, box_px: int = 7):
    """Auto YOLO labels for radar blips. Returns list of (cls_id, cx,cy,w,h)
    normalised to frame_wh. Skips dead players and blips off the radar."""
    W, H = frame_wh
    out = []
    half = box_px / 2
    for p in gt_players:
        if not p.get("alive", True) or "map_pos" not in p:
            continue
        sx, sy = map_to_screen(p["map_pos"], rect)
        if not (rect.x <= sx <= rect.x + rect.w and rect.y <= sy <= rect.y + rect.h):
            continue
        cls = "blip_ct" if p.get("team") == "CT" else "blip_t"
        out.append((CLS_ID[cls], sx / W, sy / H, box_px / W, box_px / H))
    return out


# --------------------------------------------------------------------------- #
def _source_basis(yaw_deg: float, pitch_deg: float):
    """Source-engine forward/right/up unit vectors from view angles."""
    y, pch = math.radians(yaw_deg), math.radians(pitch_deg)
    cp, sp = math.cos(pch), math.sin(pch)
    cy, sy = math.cos(y), math.sin(y)
    fwd = np.array([cp * cy, cp * sy, -sp])
    right = np.array([sy, -cy, 0.0])               # +right = -left
    up = np.array([sp * cy, sp * sy, cp])
    return fwd, right, up


def project_world_to_screen(cam_pos, yaw_deg, pitch_deg, target,
                            fov_deg: float, frame_wh) -> tuple[float, float] | None:
    """Pinhole projection of a world point to screen px. Returns None if the
    point is behind the camera. EXPERIMENTAL -- calibrate fov_deg + eye offset."""
    W, H = frame_wh
    fwd, right, up = _source_basis(yaw_deg, pitch_deg)
    d = np.asarray(target, float) - np.asarray(cam_pos, float)
    z = float(d @ fwd)
    if z <= 1e-3:
        return None
    x = float(d @ right)
    y = float(d @ up)
    f = (W / 2) / math.tan(math.radians(fov_deg) / 2)
    sx = W / 2 + f * x / z
    sy = H / 2 - f * y / z                          # screen y grows downward
    return sx, sy


def world_player_labels(gt_players, spectated_pid, fov_deg, frame_wh,
                        eye_offset_z: float = 64.0, box_wh=(60, 150)):
    """EXPERIMENTAL world-view player boxes. spectated player's eye is the
    camera. Calibrate fov_deg/eye_offset_z/box scaling on real renders first."""
    W, H = frame_wh
    spec = next((p for p in gt_players if p.get("pid") == spectated_pid
                 or p.get("steamid") == spectated_pid), None)
    if spec is None or "world" not in spec:
        return []
    cam = np.array(spec["world"], float) + np.array([0, 0, eye_offset_z])
    yaw, pitch = spec.get("yaw", 0.0), spec.get("pitch", 0.0)
    out = []
    for p in gt_players:
        if p is spec or not p.get("alive", True) or "world" not in p:
            continue
        head = np.array(p["world"], float) + np.array([0, 0, 72])
        proj = project_world_to_screen(cam, yaw, pitch, head, fov_deg, frame_wh)
        if proj is None:
            continue
        sx, sy = proj
        if not (0 <= sx <= W and 0 <= sy <= H):
            continue
        dist = float(np.linalg.norm(np.array(p["world"]) - cam))
        scale = max(0.3, 800.0 / max(dist, 1))
        bw, bh = box_wh[0] * scale, box_wh[1] * scale
        cls = "player_ct" if p.get("team") == "CT" else "player_t"
        out.append((CLS_ID[cls], sx / W, (sy + bh / 2) / H, bw / W, bh / H))
    return out


# --------------------------------------------------------------------------- #
def write_yolo_labels(records, rect, frames_dir, frame_wh,
                      world_view=False, fov_deg=90.0):
    """records: assemble_clip output (frame_path, players, ...). Writes one
    YOLO .txt per frame next to it. Radar labels always; world-view optional."""
    n = 0
    for rec in records:
        labels = radar_blip_labels(rec["players"], rect, frame_wh)
        if world_view and rec.get("spectated") is not None:
            labels += world_player_labels(rec["players"], rec["spectated"],
                                          fov_deg, frame_wh)
        stem = os.path.splitext(os.path.basename(rec["frame_path"]))[0]
        with open(os.path.join(frames_dir, stem + ".txt"), "w") as f:
            for cls_id, cx, cy, w, h in labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        n += 1
    return n
