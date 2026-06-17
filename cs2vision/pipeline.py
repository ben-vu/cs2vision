"""
Integrated CS2Vision pipeline for REAL footage (thesis Sec 1.4, 2.3.3-2.3.4).

Per frame:
  1. detector.detect(frame)                          (YoloDetector)
  2. radar.estimate(frame) -> H (crop px -> map px)  (Calibrated/RegisteredRadar)
  3. project blip detections to canonical map pixels
  4. project detected radar smokes to map-pixel exclusion zones
  5. per-team nearest-neighbour tracking; tracks whose last position lies in an
     active exclusion zone become OCCLUDED (structured missingness) with
     last-known-position imputation under confidence decay.

Scoring (score_frame) is unchanged from the validated pilot: Hungarian matching
of estimates to demo-derived ground truth, with completeness and fidelity@10px.
The ONLY differences from the pilot are real inputs (real detector, real radar
transform, real demo GT) -- the geometry/tracking/scoring maths are identical,
which is the whole point of keeping the pilot's interfaces.

Note: thresholds below are in canonical-map PIXELS (the radar-image space),
not the simulator's abstract "map units". Calibrate them on your validation
split; the defaults assume a 1024px radar image (~one CS2 map across ~1024 px).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .homography import project

ASSOC_THR_PX = 25.0
FIDELITY_THR_PX = 10.0
TRACK_DECAY = 0.85
TRACK_MIN_CONF = 0.30


@dataclass
class Track:
    team: str
    pos: np.ndarray
    conf: float = 1.0
    occluded: bool = False
    age: int = 0


class TeamTracker:
    def __init__(self, team: str):
        self.team = team
        self.tracks: list[Track] = []

    def update(self, obs: np.ndarray, smoke_zones):
        matched_t, matched_o = set(), set()
        if len(self.tracks) and len(obs):
            cost = np.linalg.norm(
                np.array([t.pos for t in self.tracks])[:, None, :] - obs[None, :, :],
                axis=2)
            ti, oi = linear_sum_assignment(cost)
            for a, b in zip(ti, oi):
                if cost[a, b] <= ASSOC_THR_PX * 2:
                    tr = self.tracks[a]
                    tr.pos, tr.conf, tr.occluded, tr.age = obs[b].copy(), 1.0, False, 0
                    matched_t.add(a); matched_o.add(b)
        fresh = set()
        for b in range(len(obs)):
            if b not in matched_o and len(self.tracks) < 5:
                self.tracks.append(Track(self.team, obs[b].copy()))
                fresh.add(len(self.tracks) - 1)
        new_tracks = []
        for a, tr in enumerate(self.tracks):
            if a in matched_t or a in fresh:
                new_tracks.append(tr); continue
            tr.conf *= TRACK_DECAY
            tr.age += 1
            tr.occluded = any(np.linalg.norm(tr.pos - c) <= r + 5 for c, r in smoke_zones)
            if tr.conf >= TRACK_MIN_CONF:
                new_tracks.append(tr)
        self.tracks = new_tracks


class CS2VisionPipeline:
    def __init__(self, detector, radar):
        self.detector = detector
        self.radar = radar
        self.trackers = {"CT": TeamTracker("CT"), "T": TeamTracker("T")}

    def process(self, frame) -> dict:
        dets = self.detector.detect(frame)
        H, diag = self.radar.estimate(frame)
        rx = getattr(self.radar.rect, "x", 0)
        ry = getattr(self.radar.rect, "y", 0)

        def to_map(bbox):
            # blip/smoke centre in full-frame px -> crop-relative -> map px
            cx = (bbox[0] + bbox[2]) / 2 - rx
            cy = (bbox[1] + bbox[3]) / 2 - ry
            if H is None:
                return None
            return project(H, np.array([[cx, cy]]))[0]

        blips = {"CT": [], "T": []}
        for d in dets:
            if d.cls in ("blip_ct", "blip_t"):
                m = to_map(d.bbox)
                if m is not None:
                    blips["CT" if d.cls == "blip_ct" else "T"].append(m)
        pos = {t: (np.array(blips[t]) if blips[t] else np.empty((0, 2)))
               for t in ("CT", "T")}

        zones = []
        for d in dets:
            if d.cls == "smoke_radar":
                m = to_map(d.bbox)
                if m is not None:
                    # radius: use detected box half-width in map px as estimate
                    half = (d.bbox[2] - d.bbox[0]) / 2
                    rpx = project(H, np.array([[half, 0], [0, 0]]))
                    rad = float(np.linalg.norm(rpx[0] - rpx[1])) if H is not None else 36.0
                    zones.append((m, rad))

        imputed = {"CT": [], "T": []}
        for team in ("CT", "T"):
            self.trackers[team].update(pos[team], zones)
            for tr in self.trackers[team].tracks:
                if tr.age > 0 and tr.occluded:
                    imputed[team].append((tr.pos.copy(), tr.conf))

        return dict(dets=dets, H=H, hom_diag=diag, positions=pos,
                    zones=zones, imputed=imputed)

    @staticmethod
    def score_frame(out: dict, gt_players: list[dict], condition: str):
        res = {"accounted": 0, "recovered10": 0, "alive": 0}
        for team in ("CT", "T"):
            gt_pos = np.array([p["map_pos"] for p in gt_players
                               if p["team"] == team and p["alive"]])
            res["alive"] += len(gt_pos)
            if len(gt_pos) == 0:
                continue
            est = list(out["positions"][team])
            if condition == "smoke_imputation":
                est += [p for p, c in out["imputed"][team]]
            est = np.array(est) if est else np.empty((0, 2))
            matched = np.zeros(len(gt_pos), bool)
            within10 = np.zeros(len(gt_pos), bool)
            if len(est):
                cost = np.linalg.norm(gt_pos[:, None, :] - est[None, :, :], axis=2)
                gi, ei = linear_sum_assignment(cost)
                for a, b in zip(gi, ei):
                    if cost[a, b] <= ASSOC_THR_PX:
                        matched[a] = True
                        within10[a] = cost[a, b] <= FIDELITY_THR_PX
            if condition in ("smoke_flagging", "smoke_imputation"):
                for a, g in enumerate(gt_pos):
                    if matched[a]:
                        continue
                    if any(np.linalg.norm(g - c) <= r for c, r in out["zones"]):
                        matched[a] = True
            res["accounted"] += int(matched.sum())
            res["recovered10"] += int(within10.sum())
        comp = res["accounted"] / res["alive"] if res["alive"] else 1.0
        fid = res["recovered10"] / res["alive"] if res["alive"] else 1.0
        return comp, fid, res["alive"]
