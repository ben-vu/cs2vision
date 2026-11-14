# CS2Vision

CS2Vision reconstructs player positions from the observer radar in Counter-Strike 2 broadcast video. It scores what it recovers against demo ground truth, and reports that score against the ceiling the radar itself imposes.

This build is scoped to one map (de_mirage) and one broadcast overlay style.

**Start with [GUIDE.md](GUIDE.md).** It explains how to go from raw demos and VODs to the result tables.

```bash
pip install -r requirements.txt && pip install -e .
cs2v selftest      # synthetic end-to-end dry run, no real data needed
pytest -q          # unit tests plus a wiring test
```

## Pipeline

1. **Stage 1: ground truth, ingestion and sync.** `parse-demo` extracts ground truth from the demo, and `align` and `ingest` sync the video to it and extract frames.
2. **Stage 2: radar geometry.** `calibrate` marks the radar by hand, then `align` refines that calibration automatically.
3. **Stage 3: channel ceiling (Experiment A).** `exp-a` works from demos alone; no detector or video is needed.
4. **Stage 4: detection and recovery (Experiments B and C).** `autolabel`, `train`, `exp-b` and `exp-c`.

Finally, `report` writes `results/RESULTS.md`. The distribution contains **no** match
media, trained weights, or experiment results: the source can compute outputs after
those inputs are supplied, but it does not authenticate numbers quoted in the thesis.

**Scope of supplied source:** Experiments A–C only. The earlier five-map
Experiment D simulator, its outputs, and its analysis scripts are not in this
repository and cannot be reproduced from this package.

**Metric definitions:** completeness uses a 5 m presence threshold; fidelities
use their stated 1 m, 3 m and 10 m thresholds, with assignment searching through
at least 10 m. Thus 10 m fidelity can be larger than 5 m completeness.

**Inference boundary:** detector and tracking read video frames, but per-match
alignment and calibration in the experiment-building workflow use demo-derived
positions. This is *not* an independently demonstrated video-only deployment
workflow for unpaired, unseen broadcasts.

## Module inventory (for Appendix A)

| Module | Role |
|---|---|
| `units.py` | World units, canonical radar pixels and metres (Appendix C); Mirage overview transform. |
| `groundtruth.py` | demoparser2 extraction of per-tick player state and round boundaries; fast tick lookup. |
| `video.py` | Timestamp-based frame access. |
| `radar.py` | Canonical-to-frame transform, robust affine/homography fit (RANSAC, 3 px), radar-visibility check. |
| `calibrate.py` | One-off hand calibration of the radar rectangle and blip diameter. |
| `blobs.py` | Classical colour-threshold blip detector: used for sync, calibration refinement, label checking, and as the non-learned baseline. |
| `sync.py` | Anchor-based prediction plus per-round automatic sync; alternating calibration refinement; residual and drift reporting. |
| `viz.py` | Check sheets overlaying demo positions on the broadcast radar. |
| `ingest.py` | Samples live-round frames, crops the radar, drops frames where the radar is hidden, writes the manifest. |
| `autolabel.py` | Match-level splits, YOLO labels projected from the demo, automatic agreement, and the 200-image manual check sample. |
| `train.py` | Two-phase YOLOv8 training for each configuration and seed; blip-scale study. |
| `detector.py` | Common detector interface over YOLO and blob detectors. |
| `tracker.py` | Per-team Hungarian tracker with confidence decay and last-known-position coasting. |
| `channel.py` | Precision floor, blip separability, recoverable fraction. |
| `evaluate.py` | Hungarian matching, completeness and fidelity, four-way failure attribution, CIs and paired tests with Cohen's d_z. |
| `pipeline.py` | Per-frame detect, project, track and score loop. |
| `experiments.py` | Experiments A, B and C plus the report. |
| `synth.py` | Synthetic fixture for tests and `selftest` (not a data source). |
| `cli.py` | The `cs2v` command. |

## Reproducibility

- **Seeds.** Every stochastic step takes a seed: the split, the label-check sample, RANSAC, and YOLO training (with deterministic mode on).
- **Splits.** Split assignments are written to `data/splits.csv` and never regenerated unless you pass `--force`.
- **Media.** Broadcast video and demo files are third-party material and are not redistributed. Record the HLTV match URLs, VOD URLs and section times instead.
