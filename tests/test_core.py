import numpy as np
import pandas as pd
import pytest

from cs2vision.channel import precision_floor, unresolvable
from cs2vision.evaluate import match_distances, mean_ci, paired, round_table
from cs2vision.groundtruth import GTIndex, build_rounds, team_to_class
from cs2vision.radar import RadarCalib, fit_transform
from cs2vision.sync import chamfer, predict_starts
from cs2vision.tracker import Tracker
from cs2vision.units import METRES_PER_UNIT, MapMeta

META = MapMeta("de_mirage", -3230, 1713, 5.0)


def test_units_roundtrip_and_conversions():
    x, y = np.array([-3230.0, 0.0, 1500.0]), np.array([1713.0, 0.0, -2000.0])
    cx, cy = META.world_to_canon(x, y)
    assert cx[0] == 0 and cy[0] == 0
    bx, by = META.canon_to_world(cx, cy)
    assert np.allclose(bx, x) and np.allclose(by, y)
    # Appendix C: 10 canonical px at scale 5 ~ 0.95 m, smoke 144 u ~ 2.74 m ~ 28.8 px
    assert META.canon_to_m(10) == pytest.approx(0.9525, rel=1e-3)
    assert 144 * METRES_PER_UNIT == pytest.approx(2.743, rel=1e-3)
    assert META.m_to_canon(144 * METRES_PER_UNIT) == pytest.approx(28.8, rel=1e-3)


def test_radar_rect_roundtrip_and_floor():
    c = RadarCalib.from_rect(30, 40, 300, 300, blip_px=10)
    pts = np.array([[0, 0], [1024, 1024], [512, 200]])
    f = c.canon_to_frame(pts)
    assert np.allclose(f[0], [30, 40]) and np.allclose(f[1], [330, 340])
    assert np.allclose(c.frame_to_canon(f), pts)
    fl = precision_floor(5.0, c.frame_px_per_canon)
    assert fl["world_units_per_px"] == pytest.approx(5.0 * 1024 / 300)


def test_fit_transform_recovers_affine_with_outliers():
    rng = np.random.default_rng(0)
    A = np.array([[0.3, 0.01, 50], [-0.01, 0.29, 20], [0, 0, 1]])
    src = rng.uniform(0, 1024, (200, 2))
    dst = (np.c_[src, np.ones(200)] @ A.T)[:, :2] + rng.normal(0, 0.3, (200, 2))
    dst[:20] += rng.uniform(20, 60, (20, 2))  # outliers
    H, mask, rms = fit_transform(src, dst)
    assert np.allclose(H[:, :2], A[:, :2], atol=0.005) and np.allclose(H[:, 2], A[:, 2], atol=0.3) and rms < 1.0 and mask[:20].sum() < 3


def test_unresolvable_and_chamfer():
    xy = np.array([[0, 0], [5, 0], [50, 50]], float)
    assert unresolvable(xy, 10).tolist() == [True, True, False]
    assert unresolvable(xy, 10, cls=[0, 1, 0], same_class_only=True).tolist() == [False] * 3
    assert chamfer(xy, xy, 10) == 0
    assert chamfer(xy, np.zeros((0, 2)), 10) == 10


def test_build_rounds_pairs_and_dedupes():
    freeze = [100, 1100, 2100, 2200]            # 2200 = restart of round 3
    ends = pd.DataFrame({"tick": [1000, 2000, 3000], "winner": ["CT", "T", "CT"]})
    no = {100: 1, 1100: 2, 2100: 3, 2200: 3}
    r = build_rounds(freeze, ends, [1500], lambda t: no[t])
    assert r["round"].tolist() == [1, 2, 3]
    assert r["freeze_end_tick"].tolist() == [100, 1100, 2200]
    assert r["plant_tick"].tolist() == [-1, 1500, -1]


def test_predict_starts_piecewise():
    rounds = pd.DataFrame({"round": [1, 2, 3], "freeze_end_tick": [640, 6400, 12800]})
    anchors = pd.DataFrame({"round": [1, 3], "video_time": ["0:10", "3:00"]})
    p = predict_starts(anchors, rounds, 64)
    assert p.tolist() == pytest.approx([10.0, 100.0, 180.0])


def test_gt_index_lookup_and_classes():
    gt = pd.DataFrame({"tick": [10, 10, 10, 20], "steamid": ["a", "b", "c", "a"],
                       "team_num": [3, 2, 2, 3], "alive": [True, True, False, True],
                       "cx": [1.0, 2.0, 3.0, 4.0], "cy": [0.0, 0.0, 0.0, 0.0]})
    idx = GTIndex(gt, "team")
    xy, cls, _ = idx.at(11)
    assert len(xy) == 2 and sorted(cls.tolist()) == [0, 1]
    assert idx.at(100)[0].shape == (0, 2)
    assert team_to_class([3, 2], "single").tolist() == [0, 0]


def test_tracker_coasts_then_drops():
    tr = Tracker(gate_px=20, decay=0.85, min_conf=0.3)
    tr.update(np.array([[100, 100]]), np.array([0]), np.array([0.9]))
    tr.update(np.array([[105, 100]]), np.array([0]), np.array([0.9]))
    assert len(tr.tracks) == 1 and np.allclose(tr.tracks[0].xy, [105, 100])
    n = 0
    while tr.tracks:
        tr.update(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0))
        n += 1
    assert n == 8  # 0.85^8 < 0.3 <= 0.85^7


def test_match_distances_per_class():
    gt = np.array([[0, 0], [100, 0]], float)
    pred = np.array([[0, 1], [100, 3]], float)
    d = match_distances(pred, np.array([0, 1]), gt, np.array([0, 1]), gate=10)
    assert d.tolist() == [1.0, 3.0]
    d = match_distances(pred, np.array([1, 0]), gt, np.array([0, 1]), gate=10)
    assert np.isinf(d).all()


def test_stats():
    ci = mean_ci([1, 2, 3, 4, 5])
    assert ci["mean"] == 3 and ci["ci_lo"] < 3 < ci["ci_hi"]
    r = paired([1, 2, 3, 4], [2, 3.5, 4, 5.5])
    assert r["diff"] == pytest.approx(1.25) and r["p"] < 0.05 and r["dz"] > 0


def test_round_table():
    f = pd.DataFrame({"match": ["m"] * 2, "round": [1, 1], "n_alive": [10, 10],
                      "n_present": [9, 8], "n_recoverable": [10, 8], "n_fid_1m": [8, 6]})
    r = round_table(f, [1.0])
    assert r["completeness"].iloc[0] == pytest.approx(0.85)
    assert r["fraction_of_ceiling"].iloc[0] == pytest.approx(0.7 / 0.9)
