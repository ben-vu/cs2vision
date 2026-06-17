"""
Ground-truth construction from real CS2 demo files (.dem) via demoparser2.

For each demo this produces, per server tick:
  * every player's world position (X,Y,Z), team, alive flag, health, armour
  * those positions projected into canonical map-image pixels (RadarMeta)
  * active smoke grenades as (map_pos, radius) exclusion zones
  * round-start / bomb-plant event ticks (used by sync.py for video alignment)

This is the *authoritative* reference the vision pipeline is scored against.
It invents nothing: every coordinate comes from the Valve-authored demo.

You supply: the .dem file (HLTV.org) and the matching Valve radar .txt
(radarmeta.py). Demo tick rate in pro play is 64/s.

Prop names differ slightly across demoparser2 versions and game updates.
Run `python -m cs2vision.groundtruth --discover your.dem` to print the prop
names actually present, then adjust WANTED_PROPS / TEAM_FIELD if needed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np

from .radarmeta import RadarMeta

# Smoke radius: CS2 smoke covers ~144 world units radius. Converted to map
# pixels per-map at runtime (depends on that map's `scale`).
SMOKE_WORLD_RADIUS = 144.0

WANTED_PROPS = ["X", "Y", "Z", "health", "armor_value", "team_num", "is_alive"]
TEAM_FIELD = "team_num"          # 2 = T, 3 = CT in CS2 demos
TEAM_MAP = {2: "T", 3: "CT"}


@dataclass
class TickGroundTruth:
    tick: int
    players: list[dict]                 # team, alive, hp, armour, world, map_pos
    smokes: list[dict] = field(default_factory=list)   # map_pos, radius (px)

    def to_dict(self) -> dict:
        return dict(tick=self.tick, players=self.players, smokes=self.smokes)


class DemoGroundTruth:
    def __init__(self, demo_path: str, radar: RadarMeta):
        from demoparser2 import DemoParser     # imported lazily
        self.parser = DemoParser(demo_path)
        self.radar = radar
        self.demo_path = demo_path

    # ------------------------------------------------------------------ #
    def discover_fields(self) -> list[str]:
        """Print prop names present in this demo so you can fix WANTED_PROPS."""
        try:
            fields = self.parser.list_updated_fields()
        except Exception:
            fields = []
        return list(fields)

    def header(self) -> dict:
        return self.parser.parse_header()

    # ------------------------------------------------------------------ #
    def event_ticks(self, event_name: str) -> list[int]:
        """Tick of every occurrence of an event (e.g. 'round_start')."""
        df = self.parser.parse_events(event_name)
        if df is None or len(df) == 0 or "tick" not in df:
            return []
        return sorted(int(t) for t in df["tick"].tolist())

    # ------------------------------------------------------------------ #
    def smoke_zones_by_tick(self, ticks: list[int] | None = None) -> dict[int, list[dict]]:
        """tick -> list of active smoke zones in map pixels.

        parse_grenades returns one row per (grenade, tick) over the grenade's
        lifetime, so the set of ticks a smoke appears at *is* its active span.
        Real demos use lowercase x/y/z and emit NaN coords before detonation;
        we keep only the deployed cloud (CSmokeGrenade, not the in-flight
        Projectile) with valid coordinates. Pass `ticks` to project only the
        frames you actually have (full-match iteration is hundreds of k rows).
        """
        gren = self.parser.parse_grenades()
        zones: dict[int, list[dict]] = {}
        if gren is None or len(gren) == 0:
            return zones
        cols = {c.lower(): c for c in gren.columns}
        cx, cy, cz = cols.get("x", "x"), cols.get("y", "y"), cols.get("z", "z")
        gtype = gren["grenade_type"].astype(str)
        smokes = gren[(gtype == "CSmokeGrenade") | (gtype == "SmokeGrenade")]
        smokes = smokes.dropna(subset=[cx, cy])
        if ticks is not None:
            smokes = smokes[smokes["tick"].isin(set(int(t) for t in ticks))]
        for _, r in smokes.iterrows():
            z = float(r[cz]) if cz in r and r[cz] == r[cz] else 0.0
            world = np.array([[float(r[cx]), float(r[cy]), z]])
            mp = self.radar.world_to_map(world)[0]
            _, _, sc = self.radar._params_for_z(z)
            rad_px = SMOKE_WORLD_RADIUS / sc
            zones.setdefault(int(r["tick"]), []).append(
                dict(map_pos=[float(mp[0]), float(mp[1])], radius=float(rad_px)))
        return zones

    # ------------------------------------------------------------------ #
    def build(self, ticks: list[int] | None = None) -> list[TickGroundTruth]:
        """Build per-tick ground truth. `ticks` restricts to the ticks you
        actually have video frames for (recommended; full-match parsing of
        every tick is large)."""
        df = self.parser.parse_ticks(WANTED_PROPS, ticks=ticks)
        smokes = self.smoke_zones_by_tick(ticks=ticks)
        out: list[TickGroundTruth] = []
        for tick, grp in df.groupby("tick"):
            players = []
            worlds = []
            for _, row in grp.iterrows():
                team = TEAM_MAP.get(int(row[TEAM_FIELD]), "?") \
                    if TEAM_FIELD in row and row[TEAM_FIELD] == row[TEAM_FIELD] else "?"
                alive = bool(row.get("is_alive", True))
                w = [float(row["X"]), float(row["Y"]), float(row.get("Z", 0.0))]
                worlds.append(w)
                players.append(dict(
                    steamid=str(row.get("steamid", "")),
                    name=str(row.get("name", "")),
                    team=team, alive=alive,
                    hp=int(row.get("health", 0) or 0),
                    armour=int(row.get("armor_value", 0) or 0),
                    world=w))
            if worlds:
                mp = self.radar.world_to_map(np.array(worlds))
                for p, m in zip(players, mp):
                    p["map_pos"] = [float(m[0]), float(m[1])]
            out.append(TickGroundTruth(int(tick), players, smokes.get(int(tick), [])))
        out.sort(key=lambda g: g.tick)
        return out


def save_ground_truth(gts: list[TickGroundTruth], path: str):
    with open(path, "w") as f:
        json.dump([g.to_dict() for g in gts], f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--radar", help="Valve overview .txt for this map")
    ap.add_argument("--out", default="ground_truth.json")
    ap.add_argument("--discover", action="store_true",
                    help="just print available demo prop names and exit")
    args = ap.parse_args()

    if args.discover:
        from demoparser2 import DemoParser
        p = DemoParser(args.demo)
        print("header:", p.parse_header())
        try:
            print("fields:", DemoParser(args.demo).list_updated_fields())
        except Exception as e:
            print("list_updated_fields unavailable:", e)
        raise SystemExit

    radar = RadarMeta.from_txt(args.radar)
    gt = DemoGroundTruth(args.demo, radar)
    print("round_start ticks:", gt.event_ticks("round_start")[:10], "...")
    records = gt.build()
    save_ground_truth(records, args.out)
    print(f"wrote {len(records)} tick records -> {args.out}")
