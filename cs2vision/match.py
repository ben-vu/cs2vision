"""One match = one folder under data/matches/<match_id>/.

Files you provide:      demo.dem, video.mp4 (any extension cv2 opens), anchors.csv
Files the tools write:  gt_ticks.parquet, rounds.csv, calib.json, radar_ref.png,
                        sync.csv, frames/, manifest.csv
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from .config import p
from .groundtruth import GTIndex, load_rounds
from .radar import RadarCalib

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".avi")


class Match:
    def __init__(self, cfg: dict, match_id: str):
        self.cfg = cfg
        self.id = match_id
        self.dir = p(cfg, "matches") / match_id
        if not self.dir.exists():
            raise FileNotFoundError(f"no match folder {self.dir}")

    # inputs ---------------------------------------------------------------------------
    @property
    def demo(self) -> Path:
        dems = sorted(self.dir.glob("*.dem"))
        if not dems:
            raise FileNotFoundError(f"no .dem file in {self.dir}")
        return dems[0]

    @property
    def video(self) -> Path:
        vids = [f for f in sorted(self.dir.iterdir()) if f.suffix.lower() in VIDEO_EXT]
        if not vids:
            raise FileNotFoundError(f"no video file in {self.dir}")
        return vids[0]

    @property
    def anchors_path(self) -> Path:
        return self.dir / "anchors.csv"

    # derived --------------------------------------------------------------------------
    calib_path = property(lambda s: s.dir / "calib.json")
    ref_path = property(lambda s: s.dir / "radar_ref.png")
    sync_path = property(lambda s: s.dir / "sync.csv")
    manifest_path = property(lambda s: s.dir / "manifest.csv")
    frames_dir = property(lambda s: s.dir / "frames")

    def gt(self) -> GTIndex:
        return GTIndex.load(self.dir, self.cfg["classes"])

    def rounds(self) -> pd.DataFrame:
        return load_rounds(self.dir)

    def calib(self) -> RadarCalib:
        if not self.calib_path.exists():
            raise FileNotFoundError(f"{self.calib_path} missing: run `cs2v calibrate`")
        return RadarCalib.load(self.calib_path)

    def ref(self):
        return cv2.imread(str(self.ref_path)) if self.ref_path.exists() else None

    def sync(self) -> pd.DataFrame:
        if not self.sync_path.exists():
            raise FileNotFoundError(f"{self.sync_path} missing: run `cs2v align`")
        return pd.read_csv(self.sync_path)

    def manifest(self) -> pd.DataFrame:
        return pd.read_csv(self.manifest_path)


def list_matches(cfg: dict) -> list[str]:
    root = p(cfg, "matches")
    return sorted(d.name for d in root.iterdir() if d.is_dir()) if root.exists() else []
