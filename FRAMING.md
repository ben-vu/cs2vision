# CS2Vision — research reframing scaffold

This is a **structural scaffold**, not thesis prose. It restructures Chapter 1
away from "a coaching tool" toward *understanding the limits of broadcast-only
state reconstruction* and *methods transferable to esports without an
authoritative feed*. Write the actual text yourself, in your own words, and
take the scope change to your supervisor — it reshapes the Introduction and the
research questions, and it should be reflected in your AI-use statement
(structuring/outlining).

---

## 1. The core idea (what the two frames show)

`broadcast.png` and `demo.png` are the same instant. The broadcast has **already
reconstructed** much of the game state for the viewer: the observer radar plots
every player, the team panels show HP/economy/weapons, an enemy is even
highlighted with a name + HP tag. The demo frame is the **authoritative** source
(everything, exactly, by parsing).

So the project is not "detect players from video" — the broadcast surfaces
players already. It is: **treat the broadcast as a lossy information channel and
measure how much authoritative state can be recovered from it, and where that is
impossible in principle.** The demo provides the authoritative reference to
measure against; the contribution is the *gap* and its *causes*.

This separates two things that the old framing conflated:

- **Channel limit** — state not present in the broadcast at all, unrecoverable
  by any method (e.g. two players closer than the radar can resolve collapse
  into one blip; sub-radar-pixel precision simply is not in the signal).
- **Method error** — state that *is* in the signal but the pipeline missed.

`channel.py` computes the limit; the pipeline's measured accuracy is then
honestly reported **as a fraction of that ceiling**.

---

## 2. Significance — rewritten angles

Drop "professional teams currently lack a tool…". Replace with:

- **Scientific (primary).** First quantification of the *information ceiling* of
  a professional esports broadcast for game-state reconstruction: how much
  authoritative state is recoverable from the broadcast signal, decomposed by
  channel (spatial radar, status HUD, occlusion) and by cause of loss. The
  ceiling is a property of the broadcast medium, independent of method, and is
  itself the novel result.
- **Methodological / transferable.** Radar/HUD-anchored recovery and
  occlusion-as-structured-missingness depend only on a broadcast exposing a
  minimap + status overlays, not on CS2 specifics. They transfer to any title
  with a broadcast but no authoritative feed (Valorant, Overwatch 2, mobile
  MOBAs) and, more abstractly, to any operator-controlled-broadcast domain where
  structured state must be inferred from the produced signal.
- **Practical (demoted to one sentence).** A by-product is that broadcast-only
  state can support downstream analysis where demos are unavailable — but this
  is a consequence, not the contribution, and should not anchor the framing.

---

## 3. Research questions — restructured

Demote detection. Make the channel the object of study.

- **Primary RQ.** To what extent is the authoritative game state of a
  professional CS2 match recoverable from the broadcast signal alone, and what
  are the limits of that recovery?

- **RQ-spatial.** How accurately can player positions be recovered from the
  broadcast observer radar, and how close does that come to the radar's
  information-theoretic precision ceiling (one radar-widget pixel)?
- **RQ-status.** To what degree can per-player status (health, economy) be
  recovered by OCR of the broadcast HUD, measured against authoritative values?
- **RQ-residual.** What game state is irrecoverable from the broadcast (off-radar
  situations, world-view occlusion, sub-resolution overlap), and how large is
  that residual?
- **RQ-transfer.** Do radar/HUD-anchored methods generalise to broadcasts of
  other titles that lack an authoritative feed?

Player detection (old RQ1) becomes a **component enabler** mentioned in the
methods (you still detect radar blips and HUD regions — that is still computer
vision), not a headline question. World-view player-model detection is
auxiliary and optional (`autolabel.world_view=False` by default).

---

## 4. Objectives — reshaped

- **RO1.** Build authoritative ground truth + the channel-limit tooling from
  demos (done: `groundtruth.py`, `channel.py`).
- **RO2.** Characterise the broadcast spatial-channel ceiling across maps and
  matches (precision floor, recoverable fraction) — **needs only demos, no
  detector, no annotation.**
- **RO3.** Recover spatial state from the radar (detect blips → radar homography
  → map coords) and report accuracy *as a fraction of the RO2 ceiling*.
- **RO4.** Recover status state by OCR of the HUD panels; score against
  authoritative HP/economy.
- **RO5.** Quantify the irrecoverable residual and its causes; discuss transfer
  to feed-less titles.

(Compare to the old RO list, which led with "train a YOLO player detector".)

---

## 5. A finding the reframe surfaces (use it)

On the production **observer** radar, all alive players are shown regardless of
smoke. So smoke is an occlusion limit on the **world-view** channel, **not** the
radar channel — the radar spatial channel is smoke-robust. The original
broadcast framing (and the old simulator) assumed smoke hides radar blips; on
real broadcasts it does not. This is worth stating explicitly: it means the
dominant spatial channel degrades far less under utility than expected, and the
real residual comes from radar resolution and sub-pixel overlap, not smoke.

Verify this on your own footage before asserting it (radar configs vary by
producer); `channel.py` has an `assume_full_radar` switch for exactly this.

---

## 6. Real numbers already available (this Nuke demo)

From `furia-vs-vitality-m3-nuke.dem`, with a representative Nuke radar:

- radar precision floor ≈ **18 world units** (≈ 2.6 canonical-map px) per blip —
  the hard ceiling on positional precision from this broadcast.
- **≈ 99%** of alive players are uniquely recoverable from the radar at its
  resolution; **≈ 1%** collapse below radar separability and are irrecoverable
  from the broadcast regardless of method.

These are exactly the kind of result the reframed thesis reports. Re-run across
maps/matches (cheap — only demos needed) to generalise the ceiling.

---

## 7. Data you need (reframed — much less than before)

The reframe + demo auto-labelling slashes the data cost, because the headline
result (the ceiling, RO2/RO5) needs **no annotation and no detector at all**.

| Purpose | Needs | Quantity |
|---|---|---|
| **Channel-ceiling study** (RO2/RO5, the headline) | demos + radar `.txt` only | as many matches as you can parse — aim for **all 5 active-duty maps × 5–10 matches each**. Free; no labelling. |
| **Spatial recovery detector** (RO3: blips) | demo-auto-labelled frames | ~**3,000–6,000 frames/map**, auto-generated (≈1–2 demos/map at 2 fps; this demo alone yields ~3,600). Zero manual labelling. |
| **Status OCR** (RO4) | a few hundred HUD crops + demo HP/economy | ~**300–800 frames** for OCR calibration; values auto-checked against the demo. |
| Auxiliary world-view players (optional) | demo-projected boxes | only if you keep it; not required. |

Net: one or two demos per map for the detector, and as many demos as you like
for the ceiling — versus the ~10–20k *hand*-annotated frames the broadcast
framing implied. The expensive manual-annotation campaign is gone.

How many maps: all five active-duty maps for the ceiling study (it is cheap and
strengthens the transfer claim). The detector/recovery side can start on one map
(Nuke or Mirage) and scale.

---

## 8. What changed in the code

- `classes.py` — classes grouped by channel (spatial/status/occlusion/event) +
  players marked auxiliary.
- `channel.py` — **new**: precision floor, separability, recoverable fraction,
  method-as-fraction-of-ceiling.
- `experiments.py` — `experiment_channel()` (new headline, GT-only);
  `experiment_1()` now reports per-channel and `primary_map50` vs
  `auxiliary_map50`, so player detection is no longer the headline number.
- `autolabel.py` — world-view player labels default **off**.

What to build next for the new framing: a `status` OCR readout module (EasyOCR
behind an interface, scored against demo HP/economy) for RO4. Say the word and
I'll add it the way the detector wrapper is structured.
