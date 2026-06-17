"""
Valve radar-metadata parsing and world -> canonical-map-pixel projection.

The canonical coordinate system used throughout CS2Vision is the map's
*radar image* (the de_<map>_radar.png shipped with the game / extractable
from the .vpk). It is a square image (1024x1024 in the standard CS2 export).
Valve ships, alongside each radar image, an overview .txt file in KeyValues
format giving the affine map from world coordinates to that image:

    de_mirage.txt
    "de_mirage"
    {
        "material" "overviews/de_mirage"
        "pos_x"    "-3230"     # world X that maps to image column 0
        "pos_y"    "1713"      # world Y that maps to image row 0
        "scale"    "5.00"      # world units per radar pixel
        ...
    }

Projection (standard awpy/leetify convention):

    col = (world_x - pos_x) / scale
    row = (pos_y - world_y) / scale          # note the sign flip on Y

Multi-level maps (Nuke, Vertigo) add a "verticalsections" block with a
separate pos_x/pos_y/scale per altitude band; pick the band whose
[AltitudeMin, AltitudeMax) contains world_z.

This module has NO simulator dependency and is unit-tested in tests/.
You must supply the real .txt files (extract from the game files or grab the
community copies); this code does not invent any coordinates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

RADAR_IMAGE_SIZE = 1024.0  # px; override per-map if your radar export differs


# --------------------------------------------------------------------------- #
# Minimal KeyValues parser (sufficient for overview .txt files).
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}])')


def parse_keyvalues(text: str) -> dict:
    """Parse a Valve KeyValues string into nested dicts. Values are strings."""
    tokens = []
    for m in _TOKEN.finditer(text):
        tokens.append(m.group(1) if m.group(1) is not None else m.group(2))
    pos = 0

    def parse_block() -> dict:
        nonlocal pos
        out: dict = {}
        while pos < len(tokens):
            tok = tokens[pos]
            if tok == "}":
                pos += 1
                return out
            key = tok
            pos += 1
            if pos < len(tokens) and tokens[pos] == "{":
                pos += 1
                out[key] = parse_block()
            else:
                out[key] = tokens[pos]
                pos += 1
        return out

    # top level usually has a single root key ("de_mirage" { ... })
    root = parse_block()
    return root


# --------------------------------------------------------------------------- #
@dataclass
class VerticalSection:
    name: str
    pos_x: float
    pos_y: float
    scale: float
    alt_min: float
    alt_max: float


@dataclass
class RadarMeta:
    """World -> radar-image-pixel affine for one map."""
    map_name: str
    pos_x: float
    pos_y: float
    scale: float
    image_size: float = RADAR_IMAGE_SIZE
    sections: list[VerticalSection] | None = None

    @classmethod
    def from_txt(cls, path: str, image_size: float = RADAR_IMAGE_SIZE) -> "RadarMeta":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            kv = parse_keyvalues(f.read())
        # root key is the map name
        map_name = next(iter(kv))
        body = kv[map_name]
        sections = None
        if "verticalsections" in body:
            sections = []
            for sname, sb in body["verticalsections"].items():
                sections.append(VerticalSection(
                    name=sname,
                    pos_x=float(sb.get("pos_x", body["pos_x"])),
                    pos_y=float(sb.get("pos_y", body["pos_y"])),
                    scale=float(sb.get("scale", body["scale"])),
                    alt_min=float(sb.get("AltitudeMin", -1e9)),
                    alt_max=float(sb.get("AltitudeMax", 1e9)),
                ))
        return cls(
            map_name=map_name,
            pos_x=float(body["pos_x"]),
            pos_y=float(body["pos_y"]),
            scale=float(body["scale"]),
            image_size=image_size,
            sections=sections,
        )

    def _params_for_z(self, z: float):
        if self.sections:
            for s in self.sections:
                if s.alt_min <= z < s.alt_max:
                    return s.pos_x, s.pos_y, s.scale
        return self.pos_x, self.pos_y, self.scale

    def world_to_map(self, world_xyz: np.ndarray) -> np.ndarray:
        """(N,3) or (N,2) world coords -> (N,2) radar-image pixel coords.

        For 2-D input, z is assumed 0 (single-level maps).
        """
        world_xyz = np.atleast_2d(np.asarray(world_xyz, float))
        out = np.empty((len(world_xyz), 2), float)
        for i, w in enumerate(world_xyz):
            z = w[2] if len(w) > 2 else 0.0
            px, py, sc = self._params_for_z(z)
            out[i, 0] = (w[0] - px) / sc
            out[i, 1] = (py - w[1]) / sc
        return out
