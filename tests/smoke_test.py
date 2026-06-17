"""
PLUMBING smoke test. Proves the real code path executes end to end using
trivial dummy inputs. It produces NO scientific results and uses NO real or
simulated match data -- it only checks that detector -> radar -> projection ->
tracking -> scoring and the clip-assembly path are correctly wired.

Run:  python -m tests.smoke_test
"""
import json
import os
import tempfile

import cv2
import numpy as np

from cs2vision.homography import RadarRect, CalibratedRadar
from cs2vision.pipeline import CS2VisionPipeline
from cs2vision.dataset import assemble_clip
from cs2vision.sync import fit_sync


# --- a stub detector: emits one CT blip at a fixed crop location ---------- #
from cs2vision.detector import Detection

class StubDetector:
    """Returns one blip whose frame-px centre is known, so we can predict
    exactly where it must project to."""
    def __init__(self, rect):
        self.rect = rect
    def detect(self, frame):
        # blip at crop-relative (100,100) -> full-frame (rect.x+100, rect.y+100)
        cx, cy = self.rect.x + 100, self.rect.y + 100
        return [Detection("blip_ct", (cx - 2, cy - 2, cx + 2, cy + 2), 0.9)]


def main():
    rect = RadarRect(x=30, y=24, w=200, h=200, map_image_size=1024.0)
    radar = CalibratedRadar(rect)
    det = StubDetector(rect)

    # crop px (100,100) with 200px widget -> canonical map px (512,512)
    frame = np.zeros((720, 1280, 3), np.uint8)
    pipe = CS2VisionPipeline(det, radar)
    out = pipe.process(frame)
    est = out["positions"]["CT"]
    assert len(est) == 1, "expected one projected blip"
    assert np.allclose(est[0], [512, 512], atol=1.0), f"projection wrong: {est[0]}"

    # a GT player at exactly that map position -> fidelity must be 1.0
    gt_players = [dict(team="CT", alive=True, map_pos=[512.0, 512.0])]
    comp, fid, n = CS2VisionPipeline.score_frame(out, gt_players, "detection_only")
    assert (comp, fid, n) == (1.0, 1.0, 1), f"scoring wrong: {(comp, fid, n)}"

    # a GT player 30px away -> matched for completeness but outside fidelity@10
    gt_far = [dict(team="CT", alive=True, map_pos=[512.0 + 30, 512.0])]
    comp2, fid2, _ = CS2VisionPipeline.score_frame(out, gt_far, "detection_only")
    assert comp2 == 0.0 and fid2 == 0.0, f"distance gating wrong: {(comp2, fid2)}"

    # --- clip-assembly path (manifest + sync + gt_json) ------------------- #
    with tempfile.TemporaryDirectory() as d:
        cv2.imwrite(os.path.join(d, "f000000.jpg"), frame)
        json.dump(dict(frames=[dict(file="f000000.jpg", src_frame=0, t_sec=0.0)]),
                  open(os.path.join(d, "manifest.json"), "w"))
        gt_json = os.path.join(d, "gt.json")
        json.dump([dict(tick=128, players=gt_players, smokes=[])], open(gt_json, "w"))
        sync = fit_sync([(0, 128), (1000, 128 + 1000)])   # a=1.0, b=128
        recs = assemble_clip(d, gt_json, sync)
        assert len(recs) == 1 and recs[0]["players"][0]["map_pos"] == [512.0, 512.0]
        assert recs[0]["tick"] == 128

    print("SMOKE TEST PASSED — real pipeline wiring is correct.")
    print("(No results produced: this validates code paths only.)")


if __name__ == "__main__":
    main()
