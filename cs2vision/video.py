"""Frame access by timestamp with cheap forward seeking."""
from __future__ import annotations

import cv2
import numpy as np


def parse_time(s) -> float:
    """'1:02:03.5', '62:03', '3723.5' -> seconds."""
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    parts = [float(x) for x in str(s).strip().split(":")]
    t = 0.0
    for x in parts:
        t = t * 60 + x
    return t


class VideoReader:
    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0  # index of the frame the next read() returns

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def index_of(self, t: float) -> int:
        return int(round(t * self.fps))

    def read_index(self, idx: int):
        if idx < 0 or idx >= self.n_frames:
            return None
        if idx < self._next or idx - self._next > int(3 * self.fps):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next = idx
        while self._next < idx:
            self.cap.grab()
            self._next += 1
        ok, frame = self.cap.read()
        self._next += 1
        return frame if ok else None

    def read_at(self, t: float):
        """Returns (frame, exact_time_of_that_frame)."""
        idx = self.index_of(t)
        return self.read_index(idx), idx / self.fps

    def close(self):
        self.cap.release()
