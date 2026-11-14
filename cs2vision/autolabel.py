"""Automatic radar-blip labels from demo ground truth (thesis Section 2.2.3, RO3).

For every visible frame in every match manifest, each alive player's canonical
position is projected through the calibrated radar transform into the radar crop and
a blip-sized square box is written in YOLO format. Matches are assigned to
train/val/test by data/splits.csv (match-level split, thesis Section 2.2.4).

Verification replacing inter-rater reliability:
  * automatic agreement: share of generated boxes with a detected blob at their centre,
  * manual check: overlay images for a seeded random sample (default 200) that you
    inspect and score in label_check.csv,
  * sync quality and radar-visibility exclusions are carried over from earlier steps.
"""
from __future__ import annotations

import random
import shutil

import cv2
import numpy as np
import pandas as pd
import yaml

from .blobs import BlobDetector
from .config import class_names, p
from .match import Match, list_matches

COLOURS = [(255, 160, 0), (0, 200, 255), (0, 255, 0)]  # BGR: CT blue, T yellow


def make_splits(cfg, ratios=(0.7, 0.15, 0.15), seed=0, overwrite=False):
    path = p(cfg, "splits")
    if path.exists() and not overwrite:
        return pd.read_csv(path)
    ms = list_matches(cfg)
    if len(ms) < 3:
        raise SystemExit("need at least 3 matches for a train/val/test split")
    rng = random.Random(seed)
    rng.shuffle(ms)
    # With ten matches and ratios 70/15/15, allocate 7/1/2; Python's
    # round(1.5) == 2 would incorrectly allocate 6/2/2. Put the rounding
    # remainder in training while reserving at least one validation and one test match.
    import math
    n_test = max(1, math.ceil(ratios[2] * len(ms)))
    n_val = max(1, math.floor(ratios[1] * len(ms)))
    if n_test + n_val >= len(ms):
        raise ValueError('not enough matches for separate train/val/test sets')
    split = ["test"] * n_test + ["val"] * n_val + ["train"] * (len(ms) - n_val - n_test)
    df = pd.DataFrame({"match": ms, "split": split}).sort_values("match").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def gt_boxes_for_frame(gt, calib, row):
    """GT positions in crop pixel coordinates for one manifest row."""
    pts, cls, sid = gt.at(row["tick"])
    xy = calib.canon_to_frame(pts) - [row["crop_x0"], row["crop_y0"]] if len(pts) else pts
    return xy, cls, sid


def build_dataset(cfg, n_check=200, seed=0, verbose=True):
    splits = make_splits(cfg)
    root = p(cfg, "dataset")
    if root.exists():
        shutil.rmtree(root)
    names = class_names(cfg)
    index = []
    for _, s in splits.iterrows():
        m = Match(cfg, s["match"])
        calib, gt, man = m.calib(), m.gt(), m.manifest()
        blob = BlobDetector.from_cfg(cfg, calib.blip_px)
        (root / "images" / s["split"]).mkdir(parents=True, exist_ok=True)
        (root / "labels" / s["split"]).mkdir(parents=True, exist_ok=True)
        for _, row in man[man["visible"]].iterrows():
            src = m.dir / row["file"]
            img = cv2.imread(str(src))
            if img is None:
                continue
            h, w = img.shape[:2]
            xy, cls, _ = gt_boxes_for_frame(gt, calib, row)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h) \
                if len(xy) else np.zeros(0, bool)
            xy, cls = xy[inside], cls[inside]
            stem = f"{m.id}__{src.stem}"
            shutil.copy2(src, root / "images" / s["split"] / f"{stem}.png")
            b = calib.blip_px
            lines = [f"{c} {x / w:.6f} {y / h:.6f} {b / w:.6f} {b / h:.6f}"
                     for (x, y), c in zip(xy, cls)]
            (root / "labels" / s["split"] / f"{stem}.txt").write_text("\n".join(lines))
            det = blob.detect(img)
            if len(xy) and len(det):
                d = np.linalg.norm(xy[:, None] - det.xy[None], axis=2).min(1)
                hit = int((d <= 0.75 * b).sum())
            else:
                hit = 0
            index.append({"split": s["split"], "match": m.id, "round": row["round"],
                          "image": f"images/{s['split']}/{stem}.png", "n_boxes": len(xy),
                          "blob_agree": hit, "blip_px": b})
    idx = pd.DataFrame(index)
    idx.to_csv(root / "index.csv", index=False)
    data = {"path": str(root.resolve()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {i: n for i, n in enumerate(names)}}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    # manual verification sample --------------------------------------------------------
    chk = p(cfg, "results") / "label_check"
    if chk.exists():
        shutil.rmtree(chk)
    chk.mkdir(parents=True)
    sample = idx[idx["n_boxes"] > 0].sample(min(n_check, int((idx["n_boxes"] > 0).sum())),
                                            random_state=seed)
    for _, r in sample.iterrows():
        img = cv2.imread(str(root / r["image"]))
        lab = (root / r["image"].replace("images", "labels", 1)).with_suffix(".txt")
        h, w = img.shape[:2]
        big = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        for line in lab.read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            x0, y0 = int((x - bw / 2) * w * 3), int((y - bh / 2) * h * 3)
            x1, y1 = int((x + bw / 2) * w * 3), int((y + bh / 2) * h * 3)
            cv2.rectangle(big, (x0, y0), (x1, y1), COLOURS[int(c) % 3], 1)
        cv2.imwrite(str(chk / (r["image"].split("/")[-1])), big)
    sheet = sample[["image", "n_boxes"]].copy()
    sheet["image"] = sheet["image"].str.split("/").str[-1]
    sheet["bad_boxes"] = ""
    sheet.to_csv(chk / "label_check.csv", index=False)

    summary = idx.groupby("split").agg(images=("image", "count"), boxes=("n_boxes", "sum"),
                                       agree=("blob_agree", "sum"))
    summary["blob_agreement"] = summary["agree"] / summary["boxes"].clip(lower=1)
    summary.to_csv(root / "summary.csv")
    if verbose:
        print(summary.to_string())
        print(f"dataset -> {root}\nmanual check sample ({len(sample)} images) -> {chk}")
    return idx
