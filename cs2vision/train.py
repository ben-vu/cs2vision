"""Detector training (thesis Section 2.3.5).

Every configuration in config.yaml (train.configs) is trained under every seed in
train.seeds with a two-phase schedule:
  phase 1: backbone frozen (first 10 layers), heads adapt, lr_phase1, cosine decay
  phase 2: whole network, lower lr_phase2, cosine decay, starting from phase-1 best
Final weights are copied to work/weights/<config>_s<seed>.pt.

Augmentation: horizontal/vertical flips off (the map is asymmetric) and hue jitter off
(blip colour carries team identity). Mosaic stays on.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .config import p


def _model(size: str, p2: bool):
    from ultralytics import YOLO
    if p2:
        return YOLO(f"yolov8{size}-p2.yaml").load(f"yolov8{size}.pt")
    return YOLO(f"yolov8{size}.pt")


def data_yaml(cfg) -> str:
    """Path to data.yaml, rewritten to point at the dataset where it is now.

    Lets you zip work/dataset, move it to another machine (e.g. Colab) and train there.
    """
    import yaml
    root = p(cfg, "dataset").resolve()
    f = root / "data.yaml"
    d = yaml.safe_load(f.read_text())
    if d.get("path") != str(root):
        d["path"] = str(root)
        f.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(f)


def _common(cfg, seed, imgsz):
    t = cfg["train"]
    return dict(data=data_yaml(cfg), imgsz=imgsz, batch=t["batch"],
                workers=t["workers"], device=t["device"], seed=seed, deterministic=True,
                optimizer="AdamW", cos_lr=True, fliplr=0.0, flipud=0.0, hsv_h=0.0,
                mosaic=1.0, project=str(p(cfg, "runs").resolve()), exist_ok=True,
                plots=True, verbose=False)


def train_one(cfg, name: str, seed: int) -> Path:
    from ultralytics import YOLO
    t, c = cfg["train"], cfg["train"]["configs"][name]
    runs = p(cfg, "runs")
    out = runs.parent / "weights" / f"{name}_s{seed}.pt"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return out
    m = _model(t["model_size"], c["p2"])
    m.train(epochs=t["epochs_phase1"], freeze=10, lr0=t["lr_phase1"], warmup_epochs=1,
            name=f"{name}_s{seed}_p1", **_common(cfg, seed, c["imgsz"]))
    best1 = runs / f"{name}_s{seed}_p1" / "weights" / "best.pt"
    m2 = YOLO(str(best1))
    m2.train(epochs=t["epochs_phase2"], lr0=t["lr_phase2"], warmup_epochs=0,
             name=f"{name}_s{seed}_p2", **_common(cfg, seed, c["imgsz"]))
    best2 = runs / f"{name}_s{seed}_p2" / "weights" / "best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best2, out)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"config": name, "seed": seed, **c, "model_size": t["model_size"],
         "epochs": [t["epochs_phase1"], t["epochs_phase2"]],
         "lr": [t["lr_phase1"], t["lr_phase2"]]}, indent=2))
    return out


def train_all(cfg, configs=None, seeds=None):
    configs = configs or list(cfg["train"]["configs"])
    seeds = seeds if seeds is not None else cfg["train"]["seeds"]
    return [train_one(cfg, n, s) for n in configs for s in seeds]


def box_scale_study(cfg) -> pd.DataFrame:
    """Size of a radar blip at each training resolution relative to YOLO strides.

    Labels are generated at one fixed size (the blip diameter), so the k-means scale
    clustering of the original plan reduces to a single box size; what matters is
    how many pixels that box occupies after the crop is resized to imgsz.
    """
    idx = pd.read_csv(p(cfg, "dataset") / "index.csv")
    import cv2
    first = cv2.imread(str(p(cfg, "dataset") / idx["image"].iloc[0]))
    crop_side = max(first.shape[:2])
    blip = float(idx["blip_px"].median())
    rows = []
    for name, c in cfg["train"]["configs"].items():
        px = blip * c["imgsz"] / crop_side
        rows.append({"config": name, "imgsz": c["imgsz"], "p2_head": c["p2"],
                     "crop_side_px": crop_side, "blip_native_px": blip,
                     "blip_at_imgsz_px": px, "cells_P2_stride4": px / 4,
                     "cells_P3_stride8": px / 8, "below_2xP3": bool(px < 16)})
    return pd.DataFrame(rows)

