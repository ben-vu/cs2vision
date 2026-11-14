# CS2Vision: step-by-step guide to producing your results

This guide takes you from an empty folder to `results/RESULTS.md`, which contains every table your results chapter needs. Each step says what to run, what you should see, and what to do if it goes wrong.

The study is scoped down from the thesis draft to keep it achievable in an honours year:

- **One map (Mirage).** Every command assumes Mirage.
- **One tournament organiser.** Use a single production with a static, full-map observer radar.
- **Radar channel only.** HUD OCR, world-view player detection and smoke modelling are out of scope, matching the narrowed Section 1.4.
- **Three experiments.** A (channel ceiling, RQ1), B (detector, RQ2a) and C (end-to-end against the ceiling, RQ2b).

Section 11 lists the places where your thesis text must change to match what the software does.

---

## 0. What you need

| Item | Notes |
|---|---|
| A computer with a screen | For the one-off calibration step, which opens a window where you draw two boxes. |
| Python 3.10–3.12 | |
| 30–60 GB of disk | About 2–4 GB of video per match. |
| A GPU for training only | Google Colab's free T4 is enough if you use model size `s`. Everything else runs on a laptop CPU. |
| 8–10 Mirage matches | Same organiser, for Experiments B and C. Each needs the demo **and** the broadcast VOD. |
| 20–50 extra Mirage demos (optional) | No video needed. They only feed Experiment A, and more demos give a tighter ceiling estimate. |

### Install

```bash
cd cs2vision
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cs2v -h
```

If you only have a CPU, the `ultralytics` install pulls in PyTorch, which is large. That is expected.

### Prove the install works (5 minutes, no real data)

```bash
cs2v selftest
pytest -q
```

`selftest` builds four synthetic matches with a fake radar, then runs every step except YOLO training. It writes a complete `work/selftest/results/RESULTS.md`. These numbers are meaningless and must **not** go in your thesis; they only show that the machinery runs. Open `work/selftest/data/matches/synth1/check_sheet.png` to see what a good alignment looks like: every coloured ring sits on a dot.

---

## 1. Verify the Mirage radar numbers (once)

`config.yaml` contains three numbers from Valve's radar metadata:

```yaml
map:
  pos_x: -3230
  pos_y: 1713
  scale: 5.0
```

Check them against `resource/overviews/de_mirage.txt` inside the game files. You can open `game/csgo/pak01_dir.vpk` with Source 2 Viewer. Community parsers such as awpy also publish the same values. If Valve changes Mirage's radar in an update, these values change too. Write down the values you used and the game version for your Appendix B.

---

## 2. Choose the matches

1. Pick one organiser, for example ESL Pro League or BLAST. Open a VOD and look at the radar in the top-left corner. It must show the **whole map, fixed in place**, not zoomed on or centred around the spectated player. If the radar moves, pick a different organiser.
2. Pick 8–10 matches in which Mirage was played, from the same event or season so the overlay is identical.
3. For each match:
   - **Demo:** on HLTV, open the match page and use the "GOTV Demo" download. Extract the archive and keep only the Mirage `.dem`.
   - **Video:** download the VOD at 1080p. With `yt-dlp` you can download just the Mirage section:
     ```bash
     yt-dlp -f "bv*[height=1080]+ba/b[height=1080]" --download-sections "*1:23:00-2:10:00" -o video.mp4 <VOD URL>
     ```
     Give yourself a couple of minutes of margin before the pistol round and after the last round.
4. Record the HLTV match URL, the VOD URL and the section times in a spreadsheet. Appendix B promises you ship these instead of the media.

### Folder layout

```
data/
  matches/
    eslpl21_navi_faze_mirage/     <- any name without spaces
      demo.dem                    <- any name ending .dem
      video.mp4                   <- mp4 / mkv / webm
      anchors.csv                 <- you write this (step 4)
    eslpl21_g2_vitality_mirage/
      ...
  demos_only/                     <- optional extra Mirage demos for Experiment A
```

---

## 3. Parse the demos

```bash
cs2v parse-demo              # all matches
```

Each match folder gains `gt_ticks.parquet` (every player's position at every live tick) and `rounds.csv`.

**Check:** open `rounds.csv`. Rounds should be numbered 1…N with no gaps, and N should equal the final score total, for example 13–9 gives 22. If a demo is not Mirage, the command stops with an error that says so.

---

## 4. Write anchors.csv (about 2 minutes per match)

The tool needs to know roughly where round 1 starts in the video. Open the video in any player and find the moment **freeze time ends in round 1**: the round timer switches to 1:55 and players start moving. Write down that video time:

```csv
round,video_time
1,2:14.5
```

- **Precision:** ±2 seconds is plenty, because the tool searches ±4 s and finds the exact start itself.
- **Formats:** `134.5`, `2:14.5` and `1:02:14.5` are all accepted.
- **Edited VODs:** if the VOD is cut between rounds (ad breaks removed), add one more line for the first round after each cut, for example `9,21:40`. An uncut live VOD needs only the round 1 line.

---

## 5. Calibrate the radar (once per organiser)

Pick a video time where the radar is clearly visible, then run:

```bash
cs2v calibrate eslpl21_navi_faze_mirage --time 5:00
```

Two windows open, one after the other:

1. **"Box the WHOLE radar map image":** drag a box around the full extent of the Mirage radar picture, then press Enter. It does not need to be perfect; step 6 refines it.
2. **"Box ONE player blip":** drag a tight box around a single player dot, then press Enter. Its size becomes the blip diameter.

This writes `calib.json`, `radar_ref.png` (the reference image used to detect when the radar is hidden) and `calib_preview.png` (the frame with a grid over the radar; the grid should cover the map image).

For the other matches from the same organiser at the same resolution, reuse the calibration:

```bash
cs2v calibrate eslpl21_g2_vitality_mirage --time 5:00 --copy-from eslpl21_navi_faze_mirage
```

Always run `calibrate` for every match, even when copying, because it also saves that match's `radar_ref.png`.

---

## 6. Align: automatic sync and calibration refinement

```bash
cs2v align                    # all matches; about 2–5 min per match
```

For every round, the tool:

- predicts the start time from your anchor,
- searches ±4 s for the start time at which demo positions best match the dots on the radar,
- uses the best-matching pairs to refine the radar calibration (RANSAC, 3 px threshold), then repeats.

The output looks like this:

```
  round  7: start   1234.56s  residual +0.094s  match err 0.9px  late 1.1px
```

| Column | Meaning | Healthy value |
|---|---|---|
| residual | Refined start minus predicted start | Small and similar across rounds. A jump means a cut in the VOD: add an anchor for that round. |
| match err | Mean distance between dots and demo positions, early in the round | Under 2 px. Over 4 px means the round is flagged not ok and excluded. |
| late | The same distance late in the round | Similar to match err. If it is much bigger, the video is not running in real time, which is rare and worth reporting. |

The final summary line per match (`rounds_ok`, `residual_s_mean_abs`, `match_err_px_median`, `late_err_px_median`) goes into your sync-quality table. `cs2v report` collects it for you.

### If many rounds are not ok

1. Run `cs2v check <match>` and open `check_sheet.png`. Rings are demo positions and white crosses are detected dots.
2. **No crosses on the dots:** the colour thresholds do not suit this overlay. Lower `blob.min_saturation` or `blob.min_value` in `config.yaml`, then re-run `align`.
3. **Crosses on the map background:** raise those thresholds.
4. **Rings offset from the dots by a constant amount:** the initial box was far off. Redo `calibrate` more carefully.
5. **Rings in the right shape but in the wrong places:** the anchor is more than 4 s off. Fix `anchors.csv`, or raise `sync.search_s`.

---

## 7. Check alignment visually (required)

```bash
cs2v check
```

Open every `check_sheet.png`. Every coloured ring should sit on a dot (blue = CT, yellow = T). Put one of these sheets in your thesis as the figure showing that ground truth lines up with the broadcast. Do not move on until this looks right, because everything downstream depends on it.

At this point, also decide on the classes. If CT and T dots on this production are clearly different colours, keep `classes: team`. If the observer radar uses per-player colours that do not separate the teams, set `classes: single` and re-run from step 8. You then report team-agnostic results; say so in the thesis.

---

## 8. Ingest frames

```bash
cs2v ingest
```

This samples every ok round at `video.sample_fps` (default 1 per second), crops the radar and writes `frames/` and `manifest.csv`. It reports how many frames had a visible radar. Frames showing replays, scoreboards or transitions are dropped automatically, and their share is your "radar occluded" exclusion rate.

Expect roughly 1,500–2,500 frames per match.

---

## 9. Split, auto-label, and check the labels

```bash
cs2v split          # writes data/splits.csv (70/15/15 by match, seeded)
cs2v autolabel      # writes work/dataset/ and results/label_check/
```

1. **Review the split.** Open `data/splits.csv`. You may edit it, but decide before training and never change it afterwards. With 10 matches you get 6 train, 2 val and 2 test.
2. **Read the automatic check.** `autolabel` prints `blob_agreement` per split: the share of generated boxes with a detected dot at their centre. On real footage expect about 0.85–0.97; the misses are mostly dots merged with another player's or covered by an overlay.
3. **Do the manual check (about 1 hour).** This replaces the inter-rater reliability study. Open `results/label_check/`, which holds 200 enlarged radar crops with their boxes drawn. For each image, count the boxes that do not enclose a visible dot and type the number in the `bad_boxes` column of `label_check.csv`. Put 0 if all are correct; do not leave blanks. `cs2v report` turns this into your label error rate.

---

## 10. Train, then run the experiments

### 10a. Train (GPU)

Three configurations are defined in `config.yaml`:

| name | P2 head | input size | role |
|---|---|---|---|
| `full` | yes | 640 | primary model |
| `no_p2` | no | 640 | ablation: is the P2 head needed? |
| `lowres` | yes | 320 | ablation: is the higher resolution needed? |

Each configuration is trained with a two-phase schedule: the backbone is frozen for `epochs_phase1`, then the whole network is trained at a lower learning rate. Running all three configurations with three seeds each is 9 runs.

| Setup | Time for 9 runs |
|---|---|
| `model_size: m`, 20 + 40 epochs, about 12k training crops, Colab T4 | Roughly 12–20 h |
| `model_size: s`, 15 + 30 epochs | Roughly 4–8 h |

If time is tight, train `full` with 3 seeds and the ablations with 1 seed each:

```bash
cs2v train --configs full --seeds 0,1,2
cs2v train --configs no_p2,lowres --seeds 0
```

Finished models go to `work/weights/<config>_s<seed>.pt`. Training skips any model that already exists, so you can stop and resume.

#### On Colab

1. Zip `cs2vision/` (code, `config.yaml` and `work/dataset/`), but leave out `data/matches`, which is only needed for Experiments A and C.
2. Upload the zip, then run:
   ```
   !unzip -q cs2vision.zip && cd cs2vision && pip install -q -e . ultralytics && cs2v train
   ```
   Set `train.device: "0"` in `config.yaml` first.
3. Download `work/weights/` back into your local `cs2vision/work/weights/`.

The dataset path is fixed up automatically on the new machine.

### 10b. Experiment A: channel ceiling (CPU, minutes)

```bash
cs2v exp-a --demos data/demos_only        # --demos is optional
```

This uses the radar size and blip diameter from your calibration together with every parsed demo. Outputs go to `results/expA/`: `summary.md`, `summary.json`, `per_round.csv`, `by_phase.csv` and `fig_ceiling.png`.

### 10c. Tune on validation, then Experiment B and C on test

**Do this before touching the test split.** Look at `cs2v exp-c --split val`. If you want to adjust `tracker.gate_m`, `tracker.decay` or `tracker.min_conf`, do it now, on validation only. Record the final values in the thesis. After that, run:

```bash
cs2v exp-b          # detector on the test split: per-class AP, ablations, latency
cs2v exp-c          # full pipeline on the test split, including the blob baseline
cs2v report         # -> results/RESULTS.md
```

Experiment C evaluates these variants:

| Variant | What it is |
|---|---|
| `full`, `no_p2`, `lowres` | Each detector (seed 0) with tracking. |
| `full_raw` | The primary detector with no tracking, which isolates what tracking adds. |
| `blob_baseline` | The classical colour-threshold detector. It replaces the pilot's classical detector as a non-learned reference. |

It then runs paired t-tests over rounds against `full`, giving the mean difference, 95% CI, t, p and Cohen's d_z. It also produces the failure attribution (channel / detection / localisation / association) and a breakdown by round phase.

Do not re-run Experiment C on the test split after changing settings in order to get a better number. If you have to re-run it, report that in the thesis.

### Reading the "fraction of ceiling"

**Reproduction notice:** This ZIP provides no real match data, model weights, raw
metrics, or the code for the earlier five-map Experiment D. The commands in this
guide produce new measurements once you provide the requisite inputs. They
cannot establish that the thesis's existing numerical tables came from a prior
execution. The ten-match split defaults to 7 training, 1 validation and 2 test
matches after deterministic rounding. Completeness is matched within 5 m,
whereas the 10 m fidelity metric uses a separate 10 m cutoff.


The ceiling counts a player as unrecoverable when their dot's centre is closer than one blip diameter to another player's dot, in a single frame. This is a per-frame rule, not a proof, so a fraction of ceiling slightly **above 1** is possible and is itself a finding. There are two reasons:

- **Tracking:** the tracker can carry a player through a brief merge by using their position from before it.
- **Detection:** a trained detector can sometimes split two dots that only partly overlap.

The synthetic self-test already shows this: YOLO with tracking reached about 1.09 of the per-frame ceiling. Experiment A therefore also reports `merge_sensitivity.csv`, the recoverable fraction when the merge distance is 0.5, 0.75, 1.0 or 1.25 blip diameters. Report the headline at 1.0 (`eval.merge_frac`), show the sensitivity table, and compare the `full` and `full_raw` rows of Experiment C. That comparison tells you how much of any excess comes from tracking.

Do not claim that merged players are "irrecoverable by any method"; your Abstract and Section 2.3.1 currently say this. The defensible claim is that they are unrecoverable **from a single frame** under the stated merge rule.

---

## 11. Where each result goes in your thesis

Every number should be copied from `results/RESULTS.md` or the CSV named below, never typed from memory. Keep the whole `results/` folder; Appendix B promises the raw metric files.

| Thesis location | Source | What to report |
|---|---|---|
| 2.2 Dataset (matches, rounds, frames) | `work/dataset/summary.csv`; frame counts in each `manifest.csv` | Matches per split, images, boxes, and the % of frames dropped for a hidden radar. |
| 2.2 Synchronisation quality | "Synchronisation" table in RESULTS.md | Rounds ok / total, mean and max \|residual\|, median match and late error in px (and m: px × `metres_per_px` from Exp A). |
| 2.2.3 Label verification | "Manual label check" in RESULTS.md, plus `blob_agreement` | Label error rate from the 200 images; automatic agreement per split. |
| New 4.1 Experiment A | `results/expA/summary.md`, `fig_ceiling.png` | Precision floor (world units and m per radar px), recoverable fraction with 95% CI, by-phase table. |
| New 4.2 Experiment B | `results/expB/summary.md` | mAP50 and mAP50-95 (mean ± SD over seeds), per-class AP, ablation deltas, latency, blip-size table. |
| New 4.3 Experiment C | `results/expC/summary.md`, `fig_expC.png` | Completeness, fidelity at 1 m (plus 3 m and 10 m), fraction of ceiling with CI, error mean/median/p95, paired tests, failure attribution, by-phase. |
| Abstract and 5.5 Conclusion | Exp A and Exp C headline rows | The headline sentence has the form: "the pipeline recovers X% of players within 1 m where Y% are recoverable in principle, i.e. Z% of the ceiling." |

### What happens to the current Chapters 3–4

Replace the synthetic pilot with the real study:

- **Chapter 3 (Implementation):** describe this framework, stages 1–4, using the module list in `README.md`.
- **Chapter 4 (Results):** Experiments A, B and C.

The Stage-0 synthetic pilot can be reduced to a short paragraph or moved to an appendix. The `selftest` fixture fixes the smoke-hiding flaw noted in Section 5.2: every alive player is always drawn.

### Text that must change to match the software

1. **Scope (1.4, 2.2.1, 2.5).** One map (Mirage) and one organiser, with about 8–10 matches rather than 200 broadcasts across five maps and three organisers. External validity is limited accordingly. Say this plainly in 2.4.2 and 5.2.
2. **Detector input (2.3.2, 2.3.5).** YOLO runs on the radar crop, not the full 1080p frame. The ablation is 640 vs 320 input on the crop, not 1280 vs 640 on the frame. The P2-head ablation is unchanged.
3. **Box-scale study (2.3.5).** Auto-labels all have one fixed size (the blip diameter), so k-means clustering is replaced by `box_scale.csv`, which shows how many pixels a blip covers at each input size relative to the P2 and P3 strides.
4. **Training schedule (2.3.5).** Two phases: a frozen backbone (first 10 layers) with AdamW, then the full network at a lower learning rate, both with cosine decay. There is no per-layer discriminative learning rate, and no learning-rate range test unless you run one yourself. Horizontal and vertical flips and hue jitter are disabled; say why (map asymmetry, team colour).
5. **Synchronisation (2.2.2).** One hand-marked anchor per uncut VOD, then an automatic per-round search that matches radar dots to demo positions. The residual is reported per round. This replaces "three reference events and a least-squares fit".
6. **Calibration (homography.py and calibrate.py in Appendix A).** A hand-drawn rectangle refined by an affine fit (RANSAC, 3 px) on dot-to-demo correspondences. There is no image-registration fallback for moving radars; such productions are excluded.
7. **Annotation (2.2.3, 2.4.1, 2.6.2).** Remove every remaining mention of CVAT, a second coder, Cohen's kappa or an IoU target (they survive in 2.4.1 and 2.6.2 of your draft). Replace them with the automatic agreement plus the 200-image manual check.
8. **Metrics (2.3.3).**
   - Completeness = share of alive players with a predicted position within 5 m (the presence gate).
   - Fidelity = share within 1 m, with 3 m and 10 m as secondary thresholds.
   - Matching is Hungarian, per team.
   - The failure attribution has **four** categories: channel (unresolvable on the radar), detection, localisation and association.
9. **Baseline.** The non-learned comparator is the colour-threshold blob detector with per-team colour prototypes fitted on the training split. Drop the Q3Vision numeric benchmark, or keep it qualitative only.
10. **Appendix A and B.** Replace them with the module list in `README.md`, and record the versions (`pip freeze > results/requirements_frozen.txt`), `config.yaml`, the Mirage metadata values and the game version.
11. **Appendix D.** Your current declaration says the implementation work is your own. Since the code was written with AI assistance, update it to say so.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `calibrate` window does not open | You installed `opencv-python-headless`. Run `pip uninstall opencv-python-headless && pip install opencv-python`. Alternatively, skip the GUI with `--rect x,y,w,h --blip-px 11`, reading the pixel values off a screenshot. |
| `parse-demo`: "no round_freeze_end / round_end events" | The demo is corrupt or incomplete. Download it again. |
| `parse-demo`: wrong round count | The demo includes a knife round or a restart. Duplicates are removed automatically, keeping the last one; check against the scoreboard. |
| `align`: every round has match err ≈ cap and ok = False | The dots are not being detected. See step 6, "If many rounds are not ok". |
| Only some rounds fail | A cut in the VOD, or the radar hidden for most of the window. Add an anchor for that round, or accept the exclusion and report it. |
| `ingest` keeps < 80% of frames | Lower `video.radar_visible_min_ncc` slightly (e.g. 0.35). Confirm with `check` that the frames you now keep really show the radar. |
| Training out of memory | Lower `train.batch`, or use `model_size: s`. |
| `exp-b`: "no weights found" | Copy the `.pt` files into `work/weights/`, or pass `--weights-dir`. |
| Very low `fraction_of_ceiling` but good mAP | Look at the failure attribution. If localisation dominates, the calibration RMS is too high: re-run `align` and check `calib.json` `rms_px`. If association dominates, tune the tracker on validation. |
