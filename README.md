# CS2Vision — real-footage pipeline

This is the simulator-era pilot restructured to run on **real CS2 broadcast
footage and real demo-derived ground truth**. The geometry, tracking, scoring
and detection-evaluation maths are the same as the validated pilot; only the
**inputs** are now real. Nothing here fabricates data — the modules that need
your footage/demos/labels will error clearly if those are missing rather than
inventing anything.

## What runs right now (tested, no data needed)

These have unit tests (`python -m tests.test_units`) and pass:

- `radarmeta.py` — parse Valve overview `.txt`, project world → canonical map px
- `homography.py` — `CalibratedRadar` (exact) and `RegisteredRadar` (ECC) radar → map transforms
- `ingest.py` — extract sampled frames + manifest from an `.mp4`
- `sync.py` — fit video-frame ↔ demo-tick alignment
- `dataset.py`, `pipeline.py`, `evaluation.py`, `experiments.py` — wiring verified by `python -m tests.smoke_test`

## What only you can supply (the actual research work)

| Need | How | Why it isn't done here |
|---|---|---|
| Broadcast video `.mp4` | Download ESL/BLAST/IEM VODs (YouTube/HLTV) | no footage in a sandbox |
| Demo `.dem` files | HLTV.org demo archive, matched to each VOD | "" |
| Valve radar `.txt` + `_radar.png` per map | extract from game files / community mirrors | licensed game assets |
| Annotations | CVAT, export **YOLO 1.1** format | human labelling |
| Trained YOLOv8 weights `.pt` | `train_yolo.py` on a GPU | no GPU / no labels here |
| Radar screen rectangle per overlay | read it off one frame, fill `RadarRect` | per-broadcast calibration |
| Frame↔tick anchors per clip | eyeball 3 shared events (round start, plant) | needs human judgement |

## Install

```bash
pip install opencv-python-headless numpy scipy matplotlib pyyaml demoparser2 ultralytics
```
`ultralytics` (+torch) is only needed for `detector.py`/`train_yolo.py`.
`demoparser2` is the CS2 demo parser (replaces the thesis's Go library; pure-Python, same ground truth).

## End-to-end procedure

**1. Ground truth from the demo.** First discover the exact prop names in your
demo build, then build per-tick GT projected to map pixels:

```bash
python -m cs2vision.groundtruth match.dem --discover          # prints prop names
python -m cs2vision.groundtruth match.dem --radar de_mirage.txt --out gt.json
```
If `--discover` shows different names (e.g. `team_name` vs `team_num`), edit
`WANTED_PROPS`/`TEAM_FIELD` at the top of `groundtruth.py`.

**2. Extract frames from the VOD.**
```bash
python -m cs2vision.ingest match.mp4 frames/clip01 --fps 2
```

**3. Sync.** Open the frame where a round visibly starts, note its `src_frame`
from `frames/clip01/manifest.json`, and the matching `round_start` tick
(`groundtruth.event_ticks('round_start')`). Repeat for ≥3 events, then:
```python
from cs2vision.sync import fit_sync
sync = fit_sync([(src_frame_1, tick_1), (src_frame_2, tick_2), (src_frame_3, tick_3)])
print(sync.residual_seconds())   # report this in the thesis as alignment drift
```

**4. Annotate + train.** Annotate a stratified frame sample in CVAT (classes in
`classes.py`), export YOLO 1.1. Derive anchors and train:
```bash
python -m cs2vision.train_yolo --labels labels/ --k 9     # inspect small-object clusters
python -m cs2vision.train_yolo --data data.yaml --train   # GPU; two-phase schedule
```

**5. Calibrate the radar rectangle.** In any frame, read the radar widget's
pixel box and fill it in:
```python
from cs2vision.homography import RadarRect, CalibratedRadar, RegisteredRadar
rect  = RadarRect(x=28, y=24, w=190, h=190, map_image_size=1024)
radar = CalibratedRadar(rect)                          # static full-map radar (recommended)
# radar = RegisteredRadar(rect, "de_mirage_radar.png") # if the HUD radar pans/zooms
```

**6. Run the experiments on real data.**
```python
from cs2vision.detector import YoloDetector
from cs2vision.experiments import experiment_1, run_pipeline_experiments
from cs2vision.classes import CLASSES

det = YoloDetector("best.pt", CLASSES, imgsz=1280)
clips = [("frames/clip01", "gt.json", sync, "labels/clip01")]   # one tuple per clip

experiment_1(det, clips)                                  # detection mAP
run_pipeline_experiments(det, lambda: CalibratedRadar(rect), clips)  # geometry/smoke/full
```
Results land in `outputs/metrics/`. These are now **real** measurements you ran.

## Honest notes / open problems

- **Units changed.** Pilot thresholds were abstract "map units". Here they are
  **canonical-map pixels** (≈1024 across a map). Re-tune `FIDELITY_THR_PX`,
  `ASSOC_THR_PX`, and the homography jump limits on your validation split, and
  state the px↔metre conversion (from the map `scale`) in the thesis.
- **Radar anchoring is the real research risk.** The pilot's cyan cross-markers
  don't exist on real radars; that's why this uses calibration + image
  registration instead. The `RegisteredRadar` ECC path is the part most likely
  to need iteration on real HUDs (zoom-follow radars, overlay occlusion).
- **Smoke radius on the radar** is estimated from the detected box; validate it
  against the known 144-unit world radius once you have smoke annotations.
- **Sync is semi-manual.** Automating anchor detection (round-start audio,
  bomb-plant frames) is a reasonable thesis sub-contribution but isn't built.
- The detector is anchor-free YOLOv8; the k-means output is reported to justify
  the P2 head and document the small-object scale distribution.
