# CS2Vision — usage guide

A practical, end-to-end guide: what footage to use, **how much** of it, what it
must look like, and how to run the pipeline from raw VOD to scored results.
Pair this with `README_REAL.md` (command reference) and `docs/frame_layout.png`
(annotated frame schematic).

---

## 1. The mental model

There are three things you build, in order. Keep them separate in your head —
they have very different data costs.

1. **Ground truth** — from demo files. Cheap and automatic. No annotation.
2. **The detector** — a YOLOv8 you train on **hand-annotated** frames. This is
   the expensive human-labour part.
3. **The pipeline run** — detector + radar transform + demo GT, scored. Once the
   detector exists, this runs on *any* extracted frames automatically.

The key efficiency consequence: **only the detector needs hand-annotation.**
Experiments 2–4 (geometry, smoke, full pipeline) score against demo-derived
ground truth, so you can run them over whole maps without labelling a thing.
You annotate a few thousand frames *once*, to train the detector; everything
downstream is free.

---

## 2. What the footage must look like

See `docs/frame_layout.png`. A usable clip has all of:

- **1920×1080, 60 fps**, Tier-1 broadcast (ESL / BLAST / IEM). Lower
  resolution or heavy YouTube compression destroys the 8–12 px radar blips the
  geometry depends on.
- **The in-HUD radar visible and full-map**, in a fixed screen position, square,
  north-up, not cropped or zoomed. This is the single most important
  requirement — the radar *is* your spatial anchor.
- **One consistent overlay per clip.** Calibration (radar rectangle) and the
  de-overlay preprocessing are per-template. Don't mix a clip across an overlay
  change (e.g. a sponsor break that shifts the HUD).
- **Live play**, not replays, highlight montages, player-cam interludes, or the
  scoreboard/freeze-time cuts where the radar is hidden. Trim those out.

Avoid: <1080p, replays/highlights, caster face-cams over the play area,
mid-overlay-change segments, and anything where the radar disappears.

You also need, per map: the Valve overview `de_<map>.txt` and
`de_<map>_radar.png` (extract from the game's `pak01` VPK or a community mirror).

---

## 3. How much data

Distinguish three counts — people conflate them and over-scope.

| Count | What it is | Typical size |
|---|---|---|
| **Candidate frames** | frames you extract at 2 fps | ~4,000–5,000 per 40-min map |
| **Annotated frames** | the stratified subset you label for training | see tiers below |
| **Scored frames** | frames you run the pipeline over (Exp 2–4) | as many as you like (free) |

**Annotated-frame targets** (this is the number that costs you time):

| Goal | Annotated frames | Matches / maps | Notes |
|---|---|---|---|
| Plumbing test | 50–100 | 1 short clip | just to see the path run |
| **Minimum viable pilot** | **2,000–5,000** | 8–15 matches, **1 map** (Mirage) | a defensible honours result on one map; matches thesis Sec 2.2.6 |
| Per added map | +1,500–3,000 | 6–12 matches | scale once the Mirage pipeline works |
| Full thesis (5 maps) | ~10,000–20,000 total | 40–80 matches | the realistic honours-scale target |

The thesis's "≈250,000 frames" is the **candidate pool** at 2 fps across 200
maps, *not* frames you annotate. You annotate a **stratified sample** of it
(balanced across maps, economy states, round phases — uniform sampling
over-represents dull full-buy mid-round frames). Start at the pilot tier and
grow only if the validation curve says you need to.

**Why these numbers work:** the visual domain is narrow (one game, fixed art
assets, a handful of HUD layouts), so a COCO-pretrained YOLOv8 domain-adapts
with far less data than an open-world detector. A few thousand well-stratified
frames per map is enough to clear the small-object recall bar; beyond that you
hit diminishing returns fast.

**Splits:** 70 / 15 / 15 train/val/test, **match-stratified** — the test set
must come from matches absent from train/val, or you measure memorisation of
player models and overlay quirks, not generalisation.

**Time budget:** dense CS2 frames carry ~10–25 boxes (10 players + blips + HUD +
utility). Budget ~2–4 min/frame in CVAT early, dropping to ~1 min once your
annotation rules stabilise. So a 3,000-frame pilot is roughly 60–120 hours of
labelling — plan for it, and run the **pilot study on one map first** (thesis
Sec 2.2.6) to lock your class rules before scaling.

---

## 4. Workflow, step by step

```
VOD.mp4 ──ingest──> frames/ + manifest.json
demo.dem ──groundtruth──> gt.json   (positions/smokes/events, in map px)
one frame ──calibrate──> RadarRect
VOD+demo ──autosync──> SyncModel   (or hand-pick anchors with sync.fit_sync)
frames ──CVAT──> labels/  ──train_yolo──> best.pt
                                    │
        detector + radar + sync + gt ──experiments──> outputs/metrics/
```

**4.1 Ground truth (per match).**
```bash
python -m cs2vision.groundtruth demo.dem --discover            # check prop names once
python -m cs2vision.groundtruth demo.dem --radar de_mirage.txt --out gt.json
```

**4.2 Extract frames (per clip).**
```bash
python -m cs2vision.ingest VOD.mp4 frames/clip01 --fps 2
```

**4.3 Calibrate the radar rectangle (once per overlay).**
```python
from cs2vision.calibrate import interactive_calibrate, auto_suggest_rect
rect = interactive_calibrate("frames/clip01/f000000.jpg")   # click TL then BR
# headless? read the box off any viewer and use RadarRect(x,y,w,h) directly,
# or auto_suggest_rect(frame) for a rough starting box to nudge.
```
**Verify it** before trusting it — this is the highest-leverage check in the
whole project:
```python
import cv2
from cs2vision.calibrate import overlay_gt_on_radar
frame = cv2.imread("frames/clip01/f000300.jpg")
# (use gt for the tick matching this frame, via sync + assemble_clip)
cv2.imwrite("check.png", overlay_gt_on_radar(frame, rect, gt_players))
```
The drawn crosses must sit **on top of the real radar blips**. If they're
offset or scaled wrong, your `RadarRect` or radar `.txt` is off — fix it here,
not later.

**4.4 Sync (per clip).** Automatic:
```python
from cs2vision.autosync import autosync_video
from cs2vision.groundtruth import DemoGroundTruth
from cs2vision.radarmeta import RadarMeta
ticks = DemoGroundTruth("demo.dem", RadarMeta.from_txt("de_mirage.txt")).event_ticks("round_start")
sync = autosync_video("VOD.mp4", ticks, src_fps=60.0)
print("alignment drift:", round(sync.residual_seconds()*1000), "ms")   # report this
```
If the audio is too noisy (high drift / "too few consistent events"), fall back
to hand-picking 3 anchors with `sync.fit_sync([(src_frame, tick), ...])`. Aim
for residual drift well under one frame (~16 ms ideal; <100 ms is usable).

**4.5 Annotate + train.** CVAT → YOLO 1.1 export into `labels/`, then:
```bash
python -m cs2vision.train_yolo --labels labels/ --k 9       # inspect small-object clusters
python -m cs2vision.train_yolo --data data.yaml --train     # GPU, two-phase schedule
```

**4.6 Run experiments.**
```python
from cs2vision.detector import YoloDetector
from cs2vision.homography import CalibratedRadar
from cs2vision.experiments import experiment_1, run_pipeline_experiments
from cs2vision.classes import CLASSES

det = YoloDetector("runs/cs2vision/phase2_full/weights/best.pt", CLASSES, imgsz=1280)
clips = [("frames/clip01", "gt.json", sync, "labels/clip01")]

experiment_1(det, clips)                                          # detection mAP
run_pipeline_experiments(det, lambda: CalibratedRadar(rect), clips)  # geometry/smoke/full
```

---

## 5. What good output looks like

- **Calibration check:** GT crosses land on blips (Section 4.3). Non-negotiable.
- **Detection (Exp 1):** per-class AP reported. Expect blips and HUD panels to
  be your strongest classes; in-flight utility and killfeed the weakest. If
  blip AP is low, your geometry will suffer downstream — fix detection first.
- **Geometry (Exp 2):** mean positional error in **map pixels**. Convert to
  metres with the map `scale` for the thesis. The temporal-consistency method
  should beat the per-frame and naïve baselines, especially at the 95th
  percentile (camera-switch frames).
- **Smoke (Exp 3):** completeness should rise with smoke flagging, with a
  significant paired t-test, exactly as the pilot showed — now on real data.
- **Full (Exp 4):** per-round fidelity@10 px + a matched-error distribution.

Report `sync.residual_seconds()` and the calibration check alongside results —
they're your internal-validity evidence (thesis Sec 2.4.1).

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| GT crosses offset from blips | wrong `RadarRect` or wrong radar `.txt` | recalibrate; confirm map name matches the `.txt` |
| GT crosses rotated/mirrored | radar isn't north-up, or Y-sign | check you used a full-map north-up radar; verify `radarmeta` on a known landmark |
| `groundtruth` KeyError on a prop | demoparser2 prop names differ | run `--discover`, edit `WANTED_PROPS`/`TEAM_FIELD` |
| autosync "too few consistent events" | noisy audio onsets | raise `tol_sec`, lower `k`, or hand-pick anchors |
| huge positional error only on some frames | camera-switch / radar occluded by killfeed | expected tail; `RegisteredRadar` + temporal constraint mitigates |
| Nuke positions wrong on lower bombsite | vertical section not selected | ensure demo `Z` is parsed; `radarmeta` picks the altitude band |
| detector misses tiny blips | imgsz too low / no P2 head | train at 1280, keep the P2 config |
| test mAP ≫ val mAP drop on new event | overlay leakage | confirm match-stratified split; add de-overlay preprocessing |

---

## 7. Minimal worked example (do this first)

One map, end to end, before scaling:

1. Pick **3–5 Mirage matches** with clean Tier-1 VODs + matching demos.
2. Build `gt.json` for each (4.1). Extract frames at 2 fps (4.2).
3. Calibrate `rect` on one frame and **pass the overlay check** (4.3).
4. Autosync each clip; confirm drift < 100 ms (4.4).
5. Annotate ~1,500–2,000 stratified frames in CVAT; export YOLO; train (4.5).
6. Run all four experiments (4.6); read `outputs/metrics/`.

If the overlay check passes and Exp 2's temporal method beats the baselines on
real footage, the architecture has transferred off the simulator — that's your
green light to scale to all five maps.

---

## 8. Sanity-check the install

```bash
pip install opencv-python-headless numpy scipy matplotlib pyyaml demoparser2 ultralytics
python -m tests.test_units      # projection, registration, ingest, sync, calibrate, autosync
python -m tests.smoke_test      # full wiring, no data needed
```
Both must pass before you wire in footage.
