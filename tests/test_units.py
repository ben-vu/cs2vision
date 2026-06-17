"""
Unit tests for the parts of the real pipeline that run without footage/GPU:
radar projection math, radar registration geometry, frame extraction, sync fit.
Run: python -m tests.test_units
"""
import os
import tempfile

import cv2
import numpy as np

from cs2vision.radarmeta import RadarMeta
from cs2vision.homography import RadarRect, CalibratedRadar, RegisteredRadar, project
from cs2vision.ingest import extract_frames
from cs2vision.sync import fit_sync


def test_radarmeta():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "de_x.txt")
        open(p, "w").write('"de_x"{"pos_x" "-1000" "pos_y" "1000" "scale" "5"}')
        m = RadarMeta.from_txt(p)
        assert np.allclose(m.world_to_map([[-1000, 1000]])[0], [0, 0])
        assert np.allclose(m.world_to_map([[-500, 500]])[0], [100, 100])
    print("  radarmeta projection OK")


def test_calibrated():
    rect = RadarRect(x=30, y=24, w=200, h=200, map_image_size=1024.0)
    H, _ = CalibratedRadar(rect).estimate(None)
    assert np.allclose(project(H, np.array([[100, 100]]))[0], [512, 512])
    print("  calibrated similarity OK")


def test_registration():
    rng = np.random.default_rng(0)
    canon = cv2.GaussianBlur(rng.integers(0, 255, (1024, 1024), np.uint8), (0, 0), 3)
    for _ in range(40):
        cv2.line(canon, tuple(rng.integers(0, 1024, 2)), tuple(rng.integers(0, 1024, 2)),
                 int(rng.uniform(80, 255)), int(rng.uniform(1, 5)))
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "c.png"); cv2.imwrite(cp, canon)
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[24:224, 30:230] = cv2.cvtColor(cv2.resize(canon, (200, 200)), cv2.COLOR_GRAY2BGR)
        reg = RegisteredRadar(RadarRect(30, 24, 200, 200), cp, temporal=False)
        H, _ = reg.estimate(frame)
        est = project(H, np.array([[100, 100]]))[0]
        assert np.hypot(est[0] - 512, est[1] - 512) < 15, est
    print("  ECC registration OK")


def test_ingest_and_sync():
    with tempfile.TemporaryDirectory() as d:
        vp = os.path.join(d, "v.mp4")
        vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
        for i in range(60):
            vw.write(np.full((180, 320, 3), i, np.uint8))
        vw.release()
        man = extract_frames(vp, os.path.join(d, "out"), target_fps=2.0)
        assert man["step"] == 15 and len(man["frames"]) >= 3
    s = fit_sync([(0, 100), (300, 400)])     # a=1, b=100
    assert abs(s.a - 1.0) < 1e-6 and abs(s.b - 100.0) < 1e-6
    assert s.tick_for_frame(50) == 150
    print("  ingest + sync OK")


def test_calibrate():
    from cs2vision.calibrate import (rect_from_corners, verification_overlay,
                                     overlay_gt_on_radar, auto_suggest_rect)
    r = rect_from_corners((30, 24), (230, 224))
    assert (r.x, r.y, r.w, r.h) == (30, 24, 200, 200)
    frame = np.zeros((720, 1280, 3), np.uint8)
    img = overlay_gt_on_radar(frame, r, [dict(team="CT", alive=True, map_pos=[512.0, 512.0])])
    assert img[118:131, 124:137].sum() > 0          # cross drawn near (130,124)
    assert verification_overlay(frame, r).shape == frame.shape
    f2 = np.zeros((720, 1280, 3), np.uint8); f2[24:224, 30:230] = 240
    sug = auto_suggest_rect(f2)
    assert sug and abs(sug.w - 200) < 20 and abs(sug.h - 200) < 20
    print("  calibrate cores OK")


def test_autosync():
    from cs2vision.autosync import align_events
    src_fps, tickrate = 60.0, 64.0
    a0 = tickrate / src_fps
    b_true = -tickrate * 37.5
    rng = np.random.default_rng(3)
    ticks = np.cumsum(rng.uniform(28, 105, 12)) * tickrate + 5000
    bcast = [((T - b_true) / a0) / src_fps + rng.normal(0, 0.15) for T in ticks]
    bcast = bcast[1:] + [12.3, 400.0]               # drop one, add spurious
    s = align_events(bcast, ticks, src_fps, tickrate)
    assert abs(s.a - a0) < 0.01 and abs(s.b - b_true) < tickrate * 0.6
    print("  autosync aligner OK")


def test_channel():
    import tempfile, os as _os
    from cs2vision.channel import channel_ceiling, radar_precision_floor, fraction_of_ceiling
    from cs2vision.radarmeta import RadarMeta
    from cs2vision.homography import RadarRect
    with tempfile.TemporaryDirectory() as d:
        p = _os.path.join(d, "de_n.txt")
        open(p, "w").write('"de_n"{"pos_x" "0" "pos_y" "0" "scale" "7"}')
        meta = RadarMeta.from_txt(p)
    rect = RadarRect(28, 24, 196, 196, 1024.0)
    fl = radar_precision_floor(rect, meta)
    assert 17 < fl["floor_world_units"] < 20            # ~18 units on a scale-7 map
    spread = [dict(team="CT", alive=True, map_pos=[100 + 40 * i, 100]) for i in range(10)]
    assert channel_ceiling([dict(players=spread)], rect, meta)["ceiling_fraction"] == 1.0
    clash = [dict(team="CT", alive=True, map_pos=[500, 500]),
             dict(team="T", alive=True, map_pos=[501, 500])]
    c = channel_ceiling([dict(players=clash)], rect, meta)
    assert c["unresolvable_fraction"] == 1.0           # both collapse into one blip
    assert abs(fraction_of_ceiling(0.5, {"ceiling_fraction": 1.0}) - 0.5) < 1e-9
    print("  channel ceiling OK")


def test_autolabel():
    from cs2vision.autolabel import radar_blip_labels
    from cs2vision.homography import RadarRect
    rect = RadarRect(28, 24, 196, 196, 1024.0)
    players = [dict(team="CT", alive=True, map_pos=[512, 512]),
               dict(team="T", alive=False, map_pos=[0, 0])]
    labels = radar_blip_labels(players, rect, (2560, 1440))
    assert len(labels) == 1 and labels[0][0] == 0      # CT only, dead skipped
    print("  autolabel OK")


if __name__ == "__main__":
    test_radarmeta(); test_calibrated(); test_registration(); test_ingest_and_sync()
    test_calibrate(); test_autosync(); test_channel(); test_autolabel()
    print("ALL UNIT TESTS PASSED")
