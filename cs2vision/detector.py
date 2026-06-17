"""
Detection stage for REAL footage: domain-adapted YOLOv8 (thesis Sec 2.3.1).

`YoloDetector` wraps a trained Ultralytics model behind the exact interface the
rest of the pipeline expects -- `detect(frame) -> list[Detection]` -- so it is
a drop-in replacement for the simulator-era PilotDetector. The Detection
dataclass, IoU and NMS helpers are unchanged so evaluation.py keeps working.

You supply the trained weights (.pt). Train them with train_yolo.py on your
CVAT-annotated frames; this wrapper only runs inference. With no weights it
raises a clear error rather than silently faking detections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    cls: str
    bbox: tuple          # x1, y1, x2, y2  (pixels in the full frame)
    conf: float


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets: list[Detection], iou_thr=0.45) -> list[Detection]:
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(not (k.cls == d.cls and iou(k.bbox, d.bbox) > iou_thr) for k in out):
            out.append(d)
    return out


class YoloDetector:
    """Inference wrapper around a trained Ultralytics YOLO model."""

    def __init__(self, weights: str, class_names: list[str],
                 conf: float = 0.25, iou: float = 0.45, imgsz: int = 1280,
                 device: str | None = None):
        from ultralytics import YOLO          # lazy: only needed for real runs
        self.model = YOLO(weights)
        self.names = class_names
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, device=self.device,
                                 verbose=False)[0]
        out: list[Detection] = []
        # map model class indices -> canonical names (model.names order must
        # match training data.yaml, which you build to equal CLASSES order)
        model_names = res.names
        for b in res.boxes:
            cls_id = int(b.cls.item())
            name = model_names.get(cls_id, str(cls_id)) \
                if isinstance(model_names, dict) else model_names[cls_id]
            xyxy = b.xyxy[0].tolist()
            out.append(Detection(name, tuple(xyxy), float(b.conf.item())))
        return out
