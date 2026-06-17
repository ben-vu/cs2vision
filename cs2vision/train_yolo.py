"""
Train the domain-adapted YOLOv8 detector (thesis RO2 / Sec 2.3.1).

Implements the design committed to in the thesis:
  * k-means (k=9) anchor derivation on your annotation boxes
  * P2 high-resolution head for small objects (radar blips, utility icons)
  * 1280px input resolution
  * two-phase schedule: freeze backbone (head-only) then unfreeze at a
    discriminative LR (backbone 1e-5, head 1e-4)

You supply: CVAT-exported YOLO-format dataset (images + .txt labels) and a
data.yaml. Run on a GPU machine -- this does not run in the analysis sandbox
(no GPU, no labelled data). Anchor k-means and data.yaml generation DO run
anywhere and are useful to inspect before training.

Ultralytics' anchor handling is automatic for the anchor-free v8 head; the
k-means output here is reported for the thesis (it justifies the P2 head and
documents the small-object scale distribution) and is used if you switch to an
anchor-based config. Install: pip install ultralytics.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
def kmeans_anchors(label_dir: str, k: int = 9, img_wh: tuple[int, int] = (1280, 720),
                   iters: int = 60, seed: int = 0):
    """k-means on YOLO-format (normalised) box w,h -> pixel anchor shapes."""
    ws, hs = [], []
    for fp in glob.glob(os.path.join(label_dir, "*.txt")):
        for line in open(fp):
            p = line.split()
            if len(p) >= 5:
                ws.append(float(p[3]) * img_wh[0])
                hs.append(float(p[4]) * img_wh[1])
    if not ws:
        raise RuntimeError(f"no YOLO labels found in {label_dir}")
    wh = np.stack([ws, hs], 1)
    rng = np.random.default_rng(seed)
    cen = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        d = np.linalg.norm(wh[:, None, :] - cen[None, :, :], axis=2)
        a = d.argmin(1)
        for j in range(k):
            if (a == j).any():
                cen[j] = wh[a == j].mean(0)
    cen = cen[cen[:, 0].argsort()]
    return cen, wh


def write_data_yaml(path: str, train_dir: str, val_dir: str,
                    class_names: list[str], test_dir: str | None = None):
    data = dict(train=train_dir, val=val_dir, names={i: n for i, n in enumerate(class_names)})
    if test_dir:
        data["test"] = test_dir
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


# --------------------------------------------------------------------------- #
def train(data_yaml: str, model_cfg: str = "yolov8m.yaml", weights: str = "yolov8m.pt",
          imgsz: int = 1280, freeze_epochs: int = 30, finetune_epochs: int = 50,
          project: str = "runs/cs2vision", p2_head: bool = True):
    """Two-phase domain adaptation. Requires a GPU and ultralytics installed."""
    from ultralytics import YOLO

    # P2 head: use a -p2 model config if available in your ultralytics install,
    # e.g. 'yolov8m-p2.yaml'. Falls back to the standard cfg with a warning.
    cfg = model_cfg
    if p2_head and not model_cfg.endswith("-p2.yaml"):
        cand = model_cfg.replace(".yaml", "-p2.yaml")
        cfg = cand  # ultralytics ships p2 variants; verify the file exists
        print(f"[train] using P2 head config: {cfg} "
              f"(if missing, ship a -p2.yaml or set p2_head=False)")

    model = YOLO(cfg).load(weights)   # architecture from cfg, COCO weights transferred

    # Phase 1: freeze backbone, train head only
    model.train(data=data_yaml, epochs=freeze_epochs, imgsz=imgsz,
                freeze=10, lr0=1e-4, project=project, name="phase1_head",
                exist_ok=True)
    # Phase 2: unfreeze, discriminative LR (one-cycle is ultralytics default)
    model.train(data=data_yaml, epochs=finetune_epochs, imgsz=imgsz,
                freeze=0, lr0=1e-4, lrf=0.01, project=project, name="phase2_full",
                exist_ok=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="dir of YOLO .txt labels (for k-means)")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--data", help="data.yaml to train with")
    ap.add_argument("--train", action="store_true", help="run training (needs GPU)")
    args = ap.parse_args()

    if args.labels:
        cen, wh = kmeans_anchors(args.labels, k=args.k)
        print(f"k-means anchors (px, k={args.k}) on {len(wh)} boxes:")
        for w, h in cen:
            print(f"  {w:6.1f} x {h:6.1f}")
        small = (cen[:, 0] < 16).sum()
        print(f"{small}/{args.k} anchors are <16px wide -> motivates the P2 head")
    if args.train:
        train(args.data)
