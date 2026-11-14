"""Configuration loading. config.yaml at the project root overrides DEFAULTS."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "map": {"name": "de_mirage", "pos_x": -3230, "pos_y": 1713, "scale": 5.0, "radar_size": 1024},
    "demo": {"tickrate": 64},
    "classes": "team",
    "video": {"sample_fps": 1.0, "radar_visible_min_ncc": 0.45, "crop_pad": 6},
    "sync": {"search_s": 4.0, "step_ticks": 2, "window_s": [3.0, 35.0],
             "frames_per_round": 12, "max_residual_px": 4.0},
    "blob": {"min_saturation": 90, "min_value": 110, "area_min_frac": 0.25, "area_max_frac": 3.0},
    "tracker": {"gate_m": 3.0, "decay": 0.85, "min_conf": 0.30, "max_tracks_per_class": 5},
    "eval": {"fidelity_m": [1.0, 3.0, 10.0], "presence_gate_m": 5.0, "merge_frac": 1.0,
             "phases": {"early": [0, 20], "mid": [20, 60], "late": [60, 1000]}},
    "train": {"model_size": "m", "seeds": [0, 1, 2], "epochs_phase1": 20, "epochs_phase2": 40,
              "lr_phase1": 1e-3, "lr_phase2": 1e-4, "batch": 16, "workers": 4, "device": None,
              "configs": {"full": {"p2": True, "imgsz": 640},
                          "no_p2": {"p2": False, "imgsz": 640},
                          "lowres": {"p2": True, "imgsz": 320}},
              "primary": "full"},
    "paths": {"matches": "data/matches", "splits": "data/splits.csv", "dataset": "work/dataset",
              "runs": "work/runs", "results": "results"},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "configs":
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = "config.yaml") -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    root = Path.cwd()
    if path is not None and Path(path).exists():
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
        root = Path(path).resolve().parent
    cfg["_root"] = str(root)
    return cfg


def p(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the project root."""
    path = Path(cfg["paths"][key])
    return path if path.is_absolute() else Path(cfg["_root"]) / path


def class_names(cfg: dict) -> list[str]:
    return ["CT", "T"] if cfg["classes"] == "team" else ["player"]
