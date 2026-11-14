"""Detectors share one interface: detect(radar_crop_bgr) -> Detections (crop pixels)."""
from __future__ import annotations

from pathlib import Path

from .blobs import BlobDetector, Detections


class YoloDetector:
    def __init__(self, weights, imgsz=640, conf=0.25, iou=0.5, device=None):
        from ultralytics import YOLO  # lazy: only needed when a trained model is used
        self.model = YOLO(str(weights))
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device

    def detect(self, img) -> Detections:
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return Detections.empty()
        xywh = r.boxes.xywh.cpu().numpy()
        return Detections(xywh[:, :2].astype(float), r.boxes.cls.cpu().numpy().astype(int),
                          r.boxes.conf.cpu().numpy().astype(float))


def make_detector(spec: str, cfg: dict, blip_px: float | None = None):
    """spec: 'blob:<path to blob.json>' or a path to YOLO weights ('<name>.pt')."""
    if spec.startswith("blob:"):
        det = BlobDetector.load(spec[5:])
        if blip_px is not None:
            det.blip_px = blip_px
        return det
    name = Path(spec).stem.rsplit("_s", 1)[0]          # work/weights/<config>_s<seed>.pt
    imgsz = cfg["train"]["configs"].get(name, {}).get("imgsz", 640)
    return YoloDetector(spec, imgsz=imgsz, device=cfg["train"].get("device"))

