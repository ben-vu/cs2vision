"""Server-authoritative ground truth from a CS2 demo (thesis Section 2.2.2, RO1).

parse_demo() writes two files into the match folder:
  gt_ticks.parquet  one row per (tick, player): team_num, alive, world x/y/z, canonical cx/cy
  rounds.csv        one row per round: freeze_end_tick, end_tick, plant_tick, winner

GTIndex gives fast "who is alive where at tick t" lookups for everything downstream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .units import MapMeta

TEAM_T, TEAM_CT = 2, 3


def team_to_class(team_num: np.ndarray, mode: str) -> np.ndarray:
    team_num = np.asarray(team_num)
    if mode == "single":
        return np.zeros(len(team_num), dtype=int)
    return np.where(team_num == TEAM_CT, 0, 1).astype(int)


def _event(parser, name: str) -> pd.DataFrame:
    try:
        df = parser.parse_event(name)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def build_rounds(freeze_ticks, end_df: pd.DataFrame, plant_ticks, round_no_at) -> pd.DataFrame:
    """Pair every round_end with the latest freeze_end that precedes it.

    Restarts (e.g. after a technical issue) produce duplicate round numbers; the last
    occurrence is kept, which is the round that actually counted.
    """
    freeze_ticks = np.sort(np.asarray(freeze_ticks, dtype=int))
    plant_ticks = np.sort(np.asarray(plant_ticks, dtype=int))
    rows, prev_end = [], -1
    for _, r in end_df.sort_values("tick").iterrows():
        end = int(r["tick"])
        cands = freeze_ticks[(freeze_ticks > prev_end) & (freeze_ticks < end)]
        prev_end = end
        if len(cands) == 0:
            continue
        fz = int(cands[-1])
        plants = plant_ticks[(plant_ticks >= fz) & (plant_ticks <= end)]
        rows.append({
            "round": int(round_no_at(fz)),
            "freeze_end_tick": fz,
            "end_tick": end,
            "plant_tick": int(plants[0]) if len(plants) else -1,
            "winner": r.get("winner", ""),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("round", keep="last").sort_values("round").reset_index(drop=True)
    return df


def parse_demo(demo_path: str | Path, out_dir: str | Path, meta: MapMeta) -> tuple[Path, Path]:
    from demoparser2 import DemoParser  # imported lazily so tests run without it

    demo_path, out_dir = Path(demo_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parser = DemoParser(str(demo_path))

    header = parser.parse_header()
    map_name = header.get("map_name", "")
    if map_name and map_name != meta.name:
        raise ValueError(f"{demo_path.name} is {map_name}, config expects {meta.name}")

    freeze = _event(parser, "round_freeze_end")
    ends = _event(parser, "round_end")
    plants = _event(parser, "bomb_planted")
    if freeze.empty or ends.empty:
        raise RuntimeError("demo has no round_freeze_end / round_end events")

    # Game-rules props at the event ticks: warmup flag and round counter.
    key_ticks = sorted(set(freeze["tick"].astype(int)) | set(ends["tick"].astype(int)))
    rules = parser.parse_ticks(["is_warmup_period", "total_rounds_played"], ticks=key_ticks)
    rules = rules.groupby("tick").first()
    warm = rules["is_warmup_period"].astype(bool)
    ends = ends[~ends["tick"].map(warm).fillna(False).astype(bool)]
    freeze = freeze[~freeze["tick"].map(warm).fillna(False).astype(bool)]

    def round_no_at(t):
        return int(rules.loc[t, "total_rounds_played"]) + 1 if t in rules.index else -1

    rounds = build_rounds(freeze["tick"], ends, plants["tick"] if not plants.empty else [],
                          round_no_at)
    if rounds.empty:
        raise RuntimeError("could not reconstruct any rounds from events")

    live = np.concatenate([np.arange(a, b + 1) for a, b in
                           zip(rounds["freeze_end_tick"], rounds["end_tick"])])
    ticks = parser.parse_ticks(["X", "Y", "Z", "team_num", "is_alive"], ticks=live.tolist())
    ticks = ticks[ticks["team_num"].isin([TEAM_T, TEAM_CT])].dropna(subset=["X", "Y"])
    cx, cy = meta.world_to_canon(ticks["X"].to_numpy(), ticks["Y"].to_numpy())
    gt = pd.DataFrame({
        "tick": ticks["tick"].astype(np.int32).to_numpy(),
        "steamid": ticks["steamid"].astype(str).to_numpy(),
        "name": ticks["name"].astype(str).to_numpy(),
        "team_num": ticks["team_num"].astype(np.int8).to_numpy(),
        "alive": ticks["is_alive"].astype(bool).to_numpy(),
        "x": ticks["X"].to_numpy(np.float32), "y": ticks["Y"].to_numpy(np.float32),
        "z": ticks["Z"].to_numpy(np.float32),
        "cx": cx.astype(np.float32), "cy": cy.astype(np.float32),
    }).sort_values(["tick", "steamid"])

    gt_path, rounds_path = out_dir / "gt_ticks.parquet", out_dir / "rounds.csv"
    gt.to_parquet(gt_path, index=False)
    rounds.to_csv(rounds_path, index=False)
    return gt_path, rounds_path


class GTIndex:
    """Alive-player lookup by tick."""

    def __init__(self, gt: pd.DataFrame, class_mode: str = "team"):
        gt = gt[gt["alive"]].sort_values("tick")
        self.tick = gt["tick"].to_numpy(np.int64)
        self.cx = gt["cx"].to_numpy(float)
        self.cy = gt["cy"].to_numpy(float)
        self.team = gt["team_num"].to_numpy(int)
        self.cls = team_to_class(self.team, class_mode)
        self.steamid = gt["steamid"].to_numpy()
        self.uticks = np.unique(self.tick)

    @classmethod
    def load(cls, match_dir: str | Path, class_mode: str = "team") -> "GTIndex":
        return cls(pd.read_parquet(Path(match_dir) / "gt_ticks.parquet"), class_mode)

    def nearest_tick(self, t: float, max_gap: int = 4) -> int | None:
        if len(self.uticks) == 0:
            return None
        i = int(np.clip(np.searchsorted(self.uticks, t), 1, len(self.uticks) - 1))
        best = self.uticks[i] if abs(self.uticks[i] - t) < abs(self.uticks[i - 1] - t) \
            else self.uticks[i - 1]
        return int(best) if abs(best - t) <= max_gap else None

    def at(self, t: float, max_gap: int = 4):
        """(canon_xy [N,2], cls [N], steamid [N]) of alive players at the nearest tick."""
        tk = self.nearest_tick(t, max_gap)
        if tk is None:
            return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, object)
        a, b = np.searchsorted(self.tick, tk, "left"), np.searchsorted(self.tick, tk, "right")
        return (np.stack([self.cx[a:b], self.cy[a:b]], 1), self.cls[a:b], self.steamid[a:b])


def load_rounds(match_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(match_dir) / "rounds.csv")
