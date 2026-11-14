"""End-to-end wiring test on synthetic matches (no demo files, no GPU, no YOLO)."""
import json

import pandas as pd

from cs2vision.cli import main


def test_selftest_end_to_end(tmp_path):
    root = tmp_path / "st"
    main(["selftest", "--dir", str(root), "--matches", "3", "--rounds", "2"])

    # sync recovers the true round starts to within a few ticks
    for m in ("synth1", "synth2", "synth3"):
        d = root / "data" / "matches" / m
        s = pd.read_csv(d / "sync.csv").merge(pd.read_csv(d / "truth_sync.csv"), on="round")
        assert (s["start_time"] - s["true_start_time"]).abs().max() < 0.07
        H = json.loads((d / "calib.json").read_text())["H"]
        assert abs(H[0][2] - 24) < 1.0 and abs(H[1][2] - 20) < 1.0   # true rect origin

    assert (root / "work" / "dataset" / "data.yaml").exists()
    a = json.loads((root / "results" / "expA" / "summary.json").read_text())
    assert 0.5 < a["recoverable_pooled"] <= 1.0
    c = pd.read_csv(root / "results" / "expC" / "summary_table.csv")
    assert c.loc[0, "fidelity_1m"] > 0.5
    assert (root / "results" / "RESULTS.md").exists()
