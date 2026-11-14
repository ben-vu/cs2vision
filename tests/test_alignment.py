"""Regression tests for source/thesis alignment (no external media needed)."""
from pathlib import Path

import numpy as np
import pandas as pd

from cs2vision.autolabel import make_splits
from cs2vision.config import load_config
from cs2vision.evaluate import match_distances
from cs2vision.units import MapMeta


def test_ten_match_split_seven_one_two(tmp_path):
    cfg = load_config(None)
    cfg['_root'] = str(tmp_path)
    ms = tmp_path / 'data' / 'matches'
    for i in range(10):
        (ms / f'match{i:02d}').mkdir(parents=True)
    split = make_splits(cfg, seed=0)
    assert split['split'].value_counts().to_dict() == {'train': 7, 'val': 1, 'test': 2}
    assert len(split['match'].unique()) == 10
    assert make_splits(cfg, seed=123).equals(split)  # persisted, never resplit


def test_presence_and_fidelity_use_distinct_thresholds():
    meta = MapMeta('de_mirage', -3230, 1713, 5.0)
    # One detection 7 m from a player: not present within 5 m, but recovered at 10 m.
    gt = np.array([[100.0, 100.0]])
    pred = np.array([[100.0 + meta.m_to_canon(7.0), 100.0]])
    ds = match_distances(pred, np.array([0]), gt, np.array([0]), meta.m_to_canon(10))
    assert (ds <= meta.m_to_canon(5)).sum() == 0
    assert (ds <= meta.m_to_canon(10)).sum() == 1


def test_pipeline_distinct_gates_on_single_frame(tmp_path):
    import cv2
    from cs2vision.blobs import Detections
    from cs2vision.pipeline import run_match
    from cs2vision.radar import RadarCalib

    cfg = load_config(None)
    cfg['_root'] = str(tmp_path)
    meta = MapMeta.from_cfg(cfg)
    calib = RadarCalib.from_rect(5, 5, 200, 200, 8.0,
                                 frame_size=(240, 240), pad=0)
    gtpt = np.array([[300., 300.]])
    prpt = gtpt + np.array([[meta.m_to_canon(7.0), 0.]])
    fr = calib.canon_to_frame(prpt)
    assert cv2.imwrite(str(tmp_path / 'frame.png'), np.zeros((240, 240, 3), dtype=np.uint8))

    class FakeGT:
        def at(self, tick):
            return gtpt, np.array([0]), np.array(['p1'])

    class FakeMatch:
        id = 'fake'
        dir = tmp_path
        def __init__(self): self.cfg = cfg
        def calib(self): return calib
        def gt(self): return FakeGT()
        def manifest(self):
            return pd.DataFrame([dict(visible=True, round=1, frame_idx=0, file='frame.png',
                                      crop_x0=0., crop_y0=0., tick=10, secs_into_round=5,
                                      post_plant=False)])

    class FakeDetector:
        def detect(self, img):
            return Detections(fr.copy(), np.array([0]), np.array([0.9]))

    frames, errs, lat = run_match(FakeMatch(), FakeDetector(), mode='raw')
    assert frames.iloc[0]['n_present'] == 0  # outside 5 m
    assert frames.iloc[0]['n_fid_1m'] == 0
    assert frames.iloc[0]['n_fid_3m'] == 0
    assert frames.iloc[0]['n_fid_10m'] == 1  # inside 10 m
    assert len(errs) == 0  # positional error list is within presence gate
