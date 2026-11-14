"""Experiments A (ceiling), B (detector) and C (end-to-end), thesis Section 2.3.

Each experiment writes CSVs, figures and a summary.md into results/exp<X>/.
`cs2v report` stitches the summaries into results/RESULTS.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .autolabel import gt_boxes_for_frame  # noqa: E402
from .blobs import BlobDetector  # noqa: E402
from .channel import SENSITIVITY, ceiling_from_gt, precision_floor  # noqa: E402
from .config import class_names, p  # noqa: E402
from .detector import make_detector  # noqa: E402
from .evaluate import fmt_ci, mean_ci, paired, round_table  # noqa: E402
from .groundtruth import parse_demo  # noqa: E402
from .match import Match, list_matches  # noqa: E402
from .pipeline import run_match  # noqa: E402
from .units import MapMeta  # noqa: E402

PHASE_ORDER = ["early", "mid", "late", "post_plant"]


def _out(cfg, name) -> Path:
    d = p(cfg, "results") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _md_table(df: pd.DataFrame, floatfmt=".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _splits(cfg) -> pd.DataFrame:
    return pd.read_csv(p(cfg, "splits"))


# ======================================================================================
# Experiment A
# ======================================================================================
def experiment_a(cfg, calib_match: str | None = None, extra_demos: str | None = None,
                 step_ticks: int = 16):
    meta = MapMeta.from_cfg(cfg)
    matches = list_matches(cfg)
    calib_match = calib_match or next((m for m in matches
                                       if (Match(cfg, m).calib_path).exists()), None)
    if calib_match is None:
        raise SystemExit("Experiment A needs one calibrated match for the radar size and "
                         "blip diameter (run calibrate + align first)")
    calib = Match(cfg, calib_match).calib()
    scale_px = calib.frame_px_per_canon
    sources = []
    for m in matches:
        d = Match(cfg, m).dir
        if (d / "gt_ticks.parquet").exists():
            sources.append((m, d))
    if extra_demos:
        cache = p(cfg, "runs").parent / "demo_cache"
        for dem in sorted(Path(extra_demos).glob("*.dem")):
            d = cache / dem.stem
            if not (d / "gt_ticks.parquet").exists():
                print(f"parsing {dem.name}")
                try:
                    parse_demo(dem, d, meta)
                except Exception as e:  # wrong map, corrupt file...
                    print(f"  skipped: {e}")
                    continue
            sources.append((dem.stem, d))
    samples = []
    for sid, d in sources:
        gt = pd.read_parquet(d / "gt_ticks.parquet")
        rounds = pd.read_csv(d / "rounds.csv")
        samples.append(ceiling_from_gt(gt, rounds, scale_px, calib.blip_px, cfg, step_ticks,
                                       sid))
    s = pd.concat(samples, ignore_index=True)
    out = _out(cfg, "expA")
    s.to_csv(out / "samples.csv", index=False)

    s["n_rec"] = s["recoverable"] * s["n_alive"]
    per_round = s.groupby(["match", "round"]).agg(n_alive=("n_alive", "sum"),
                                                  n_rec=("n_rec", "sum")).reset_index()
    per_round["recoverable"] = per_round["n_rec"] / per_round["n_alive"]
    per_round.to_csv(out / "per_round.csv", index=False)
    floor = precision_floor(meta.scale, scale_px)
    rr = mean_ci(per_round["recoverable"])
    phase_rows = []
    for ph in PHASE_ORDER:
        sub = s[s["phase"] == ph]
        if sub.empty:
            continue
        g = sub.groupby(["match", "round"])[["n_rec", "n_alive"]].sum()
        pr = g["n_rec"] / g["n_alive"]
        ci = mean_ci(pr)
        phase_rows.append({"phase": ph, "rounds": ci["n"], "samples": len(sub),
                           "recoverable_mean": ci["mean"], "ci_lo": ci["ci_lo"],
                           "ci_hi": ci["ci_hi"],
                           "samples_with_merge": (sub["n_unresolvable"] > 0).mean()})
    phase = pd.DataFrame(phase_rows)
    phase.to_csv(out / "by_phase.csv", index=False)

    summary = {
        "sources": len(sources), "rounds": int(len(per_round)), "samples": int(len(s)),
        "radar_frame_px_per_canon_px": scale_px, "radar_side_px": scale_px * meta.radar_size,
        "blip_px": calib.blip_px, "blip_m": float(calib.blip_px * floor["metres_per_px"]),
        **floor,
        "recoverable_pooled": float(s["n_rec"].sum() / s["n_alive"].sum()),
        "recoverable_per_round_mean": rr["mean"], "ci_lo": rr["ci_lo"], "ci_hi": rr["ci_hi"],
        "recoverable_same_team_rule": float((s["recoverable_same_team_rule"] * s["n_alive"])
                                            .sum() / s["n_alive"].sum()),
        "samples_with_any_merge": float((s["n_unresolvable"] > 0).mean()),
    }
    sens = pd.DataFrame([{"merge_distance_x_blip": f,
                          "recoverable_pooled": float((s[f"recoverable_merge_{f:g}"] * s["n_alive"])
                                                      .sum() / s["n_alive"].sum())}
                         for f in SENSITIVITY])
    sens.to_csv(out / "merge_sensitivity.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(per_round["recoverable"], bins=30, color="#4a78b5")
    ax[0].set_xlabel("recoverable fraction (per round)")
    ax[0].set_ylabel("rounds")
    if len(phase):
        ax[1].bar(phase["phase"], phase["recoverable_mean"],
                  yerr=[phase["recoverable_mean"] - phase["ci_lo"],
                        phase["ci_hi"] - phase["recoverable_mean"]], color="#e0a030", capsize=4)
        ax[1].set_ylim(max(0, phase["ci_lo"].min() - 0.02), 1.0)
    ax[1].set_ylabel("recoverable fraction")
    fig.tight_layout()
    fig.savefig(out / "fig_ceiling.png", dpi=200)
    plt.close(fig)

    md = [
        "## Experiment A: broadcast radar channel ceiling (RQ1)", "",
        f"Sources: {summary['sources']} demos, {summary['rounds']} rounds, "
        f"{summary['samples']} tick samples (every {step_ticks} ticks).", "",
        f"Radar on screen: {summary['radar_side_px']:.0f} px wide; "
        f"blip diameter {calib.blip_px:.1f} px ({summary['blip_m']:.2f} m).", "",
        f"**Precision floor:** {floor['world_units_per_px']:.2f} world units = "
        f"{floor['metres_per_px']:.3f} m per broadcast radar pixel.", "",
        f"**Recoverable fraction:** pooled {summary['recoverable_pooled']:.4f}; per-round mean "
        f"{fmt_ci(rr, 4)} (95% CI, n = {rr['n']} rounds).", "",
        f"Same-team-only merge rule: {summary['recoverable_same_team_rule']:.4f}. "
        f"Samples with at least one merged pair: {summary['samples_with_any_merge']:.3f}.", "",
        "By round phase (per-round means, 95% CI):", "", _md_table(phase, ".4f"), "",
        "Sensitivity to the merge rule (distance below which two blips count as one):", "",
        _md_table(sens, ".4f"), "",
    ]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return summary


# ======================================================================================
# Experiment B
# ======================================================================================
def _latency(model, images, imgsz, device, repeats=3):
    for im in images[:5]:
        model.predict(im, imgsz=imgsz, device=device, verbose=False)
    per = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for im in images:
            model.predict(im, imgsz=imgsz, device=device, verbose=False)
        per.append((time.perf_counter() - t0) / max(1, len(images)))
    return float(np.median(per)) * 1000


def experiment_b(cfg, weights_dir: str | None = None, n_latency=100):
    from ultralytics import YOLO

    from .train import box_scale_study, data_yaml

    out = _out(cfg, "expB")
    wdir = Path(weights_dir) if weights_dir else p(cfg, "runs").parent / "weights"
    data = data_yaml(cfg)
    names = class_names(cfg)
    test_imgs = sorted((p(cfg, "dataset") / "images" / "test").glob("*.png"))[:n_latency]
    test_imgs = [cv2.imread(str(f)) for f in test_imgs]
    dev = cfg["train"]["device"]
    rows, cls_rows = [], []
    for w in sorted(wdir.glob("*.pt")):
        name, seed = w.stem.rsplit("_s", 1)
        imgsz = cfg["train"]["configs"].get(name, {}).get("imgsz", 640)
        model = YOLO(str(w))
        m = model.val(data=data, split="test", imgsz=imgsz, batch=cfg["train"]["batch"],
                      device=dev, plots=False, verbose=False, project=str(out / "val"),
                      name=w.stem, exist_ok=True)
        lat = _latency(model, test_imgs, imgsz, dev) if test_imgs else np.nan
        rows.append({"config": name, "seed": int(seed), "mAP50": float(m.box.map50),
                     "mAP50_95": float(m.box.map), "precision": float(m.box.mp),
                     "recall": float(m.box.mr), "latency_ms": lat})
        for i, ci in enumerate(m.box.ap_class_index):
            cls_rows.append({"config": name, "seed": int(seed), "class": names[int(ci)],
                             "AP50": float(m.box.ap50[i]), "AP50_95": float(m.box.ap[i])})
        print(rows[-1])
    if not rows:
        raise SystemExit(f"no weights found in {wdir}; run `cs2v train` first")
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    per_cls = pd.DataFrame(cls_rows)
    per_cls.to_csv(out / "per_class_runs.csv", index=False)

    agg = runs.groupby("config").agg(seeds=("seed", "count"), mAP50=("mAP50", "mean"),
                                     mAP50_sd=("mAP50", "std"), mAP50_95=("mAP50_95", "mean"),
                                     mAP50_95_sd=("mAP50_95", "std"),
                                     latency_ms=("latency_ms", "median")).reset_index()
    agg["unstable_sd_gt_1pt"] = agg["mAP50_sd"] > 0.01
    agg.to_csv(out / "summary_table.csv", index=False)
    cls_agg = per_cls.groupby(["config", "class"]).agg(
        AP50=("AP50", "mean"), AP50_sd=("AP50", "std"), AP50_95=("AP50_95", "mean")
    ).reset_index()
    cls_agg.to_csv(out / "per_class.csv", index=False)
    scale = box_scale_study(cfg)
    scale.to_csv(out / "box_scale.csv", index=False)

    prim = cfg["train"]["primary"]
    abl = []
    if prim in set(runs["config"]):
        base = runs[runs["config"] == prim]
        for name in sorted(set(runs["config"]) - {prim}):
            other = runs[runs["config"] == name]
            abl.append({"ablation": name, "delta_mAP50": other["mAP50"].mean() - base["mAP50"].mean(),
                        "delta_mAP50_95": other["mAP50_95"].mean() - base["mAP50_95"].mean()})
    abl = pd.DataFrame(abl)
    md = ["## Experiment B: radar blip detection (RQ2a)", "",
          "Test split, mean over seeds (SD across seeds). mAP is Ultralytics' COCO-style "
          "101-point interpolated AP. Latency = median ms per radar crop, batch 1.", "",
          _md_table(agg), "", "Per class:", "", _md_table(cls_agg), "",
          f"Ablations relative to `{prim}` (negative = worse than primary):", "",
          _md_table(abl) if len(abl) else "(no ablation runs)", "",
          "Blip size at each training resolution:", "", _md_table(scale, ".2f"), ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return agg


# ======================================================================================
# Experiment C
# ======================================================================================
def fit_blob_baseline(cfg, out_path: Path, max_frames=400, seed=0):
    """Learn per-class colour prototypes for the blob baseline from the TRAIN split."""
    sp = _splits(cfg)
    rng = np.random.default_rng(seed)
    samples, blip = [], None
    train_ids = list(sp[sp["split"] == "train"]["match"])
    for mid in train_ids:
        m = Match(cfg, mid)
        calib, gt, man = m.calib(), m.gt(), m.manifest()
        blip = calib.blip_px
        man = man[man["visible"]]
        take = man.iloc[rng.permutation(len(man))[:max_frames // max(1, len(train_ids))]]
        for _, r in take.iterrows():
            img = cv2.imread(str(m.dir / r["file"]))
            if img is not None:
                xy, cls, _ = gt_boxes_for_frame(gt, calib, r)
                samples.append((img, xy, cls))
    det = BlobDetector.from_cfg(cfg, blip)
    counts = det.fit_colours(samples)
    det.save(out_path)
    print(f"blob baseline colour prototypes fitted: {counts}")
    return det


def experiment_c(cfg, specs: list[str] | None = None, split="test", include_blob=True):
    out = _out(cfg, "expC" if split == "test" else f"expC_{split}")
    sp = _splits(cfg)
    match_ids = list(sp[sp["split"] == split]["match"])
    if not match_ids:
        raise SystemExit(f"no matches in split '{split}'")
    wdir = p(cfg, "runs").parent / "weights"
    prim = cfg["train"]["primary"]
    variants = []  # (label, spec, mode)
    if specs:
        variants += [(Path(s).stem, s, "tracked") for s in specs]
    else:
        for name in cfg["train"]["configs"]:
            w = wdir / f"{name}_s0.pt"
            if w.exists():
                variants.append((name, str(w), "tracked"))
        pw = wdir / f"{prim}_s0.pt"
        if pw.exists():
            variants.append((f"{prim}_raw", str(pw), "raw"))
    if include_blob:
        bpath = p(cfg, "runs").parent / "blob_baseline.json"
        if not bpath.exists():
            fit_blob_baseline(cfg, bpath)
        variants.append(("blob_baseline", f"blob:{bpath}", "tracked"))
    if not variants:
        raise SystemExit("nothing to evaluate: no weights in work/weights and blob disabled")

    thresholds = cfg["eval"]["fidelity_m"]
    h = thresholds[0]
    rounds_by_variant, summ, errs_by_variant, frames_by_variant = {}, [], {}, {}
    for label, spec, mode in variants:
        print(f"[Experiment C] {label} ({mode})")
        frames, errs, lats = [], [], []
        for mid in match_ids:
            m = Match(cfg, mid)
            det = make_detector(spec, cfg, blip_px=m.calib().blip_px)
            f, e, lat = run_match(m, det, mode=mode, verbose=True)
            frames.append(f)
            errs.append(e)
            lats.append(lat)
        frames = pd.concat(frames, ignore_index=True)
        errs = np.concatenate(errs) if errs else np.zeros(0)
        lats = np.concatenate(lats) if lats else np.zeros(0)
        frames.to_csv(out / f"frames_{label}.csv", index=False)
        rt = round_table(frames, thresholds)
        rt.to_csv(out / f"rounds_{label}.csv", index=False)
        rounds_by_variant[label], errs_by_variant[label] = rt, errs
        frames_by_variant[label] = frames
        row = {"variant": label, "mode": mode, "rounds": len(rt)}
        for col in ["completeness"] + [f"fidelity_{t:g}m" for t in thresholds] + \
                   ["recoverable", "fraction_of_ceiling"]:
            ci = mean_ci(rt[col])
            row[col] = ci["mean"]
            row[col + "_ci"] = f"[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]"
        row.update({"err_mean_m": errs.mean() if len(errs) else np.nan,
                    "err_median_m": np.median(errs) if len(errs) else np.nan,
                    "err_p95_m": np.percentile(errs, 95) if len(errs) else np.nan,
                    "latency_ms_median": 1000 * np.median(lats) if len(lats) else np.nan})
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(out / "summary_table.csv", index=False)

    # paired comparisons against the primary tracked variant (rounds are the unit)
    ref = prim if prim in rounds_by_variant else summary["variant"].iloc[0]
    comp = []
    for label, rt in rounds_by_variant.items():
        if label == ref:
            continue
        a = rounds_by_variant[ref].set_index(["match", "round"])
        b = rt.set_index(["match", "round"])
        j = a.join(b, lsuffix="_ref", rsuffix="_other", how="inner")
        for metric in (f"fidelity_{h:g}m", "completeness"):
            r = paired(j[f"{metric}_ref"], j[f"{metric}_other"])
            comp.append({"comparison": f"{label} vs {ref}", "metric": metric, **r})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "paired_tests.csv", index=False)

    # failure attribution and phase breakdown for the reference variant
    fr = frames_by_variant[ref]
    fail_cols = ["fail_channel", "fail_detection", "fail_localisation", "fail_association"]
    fails = fr[fail_cols].sum()
    attribution = pd.DataFrame({"stage": [c.replace("fail_", "") for c in fail_cols],
                                "players": fails.values,
                                "share_of_failures": fails.values / max(1, fails.sum()),
                                "share_of_alive": fails.values / max(1, fr["n_alive"].sum())})
    attribution.to_csv(out / "failure_attribution.csv", index=False)
    phase_rows = []
    for ph in PHASE_ORDER:
        sub = fr[fr["phase"] == ph]
        if sub.empty:
            continue
        g = sub.groupby(["match", "round"]).sum(numeric_only=True)
        fid = mean_ci(g[f"n_fid_{h:g}m"] / g["n_alive"])
        rec = mean_ci(g["n_recoverable"] / g["n_alive"])
        phase_rows.append({"phase": ph, "rounds": fid["n"], f"fidelity_{h:g}m": fid["mean"],
                           "ci": f"[{fid['ci_lo']:.3f}, {fid['ci_hi']:.3f}]",
                           "recoverable": rec["mean"]})
    phase = pd.DataFrame(phase_rows)
    phase.to_csv(out / "by_phase.csv", index=False)

    _plots_c(out, summary, errs_by_variant, attribution, h)
    show = summary[["variant", "mode", "rounds", "completeness", f"fidelity_{h:g}m",
                    f"fidelity_{h:g}m_ci", "recoverable", "fraction_of_ceiling",
                    "fraction_of_ceiling_ci"]]
    extra = summary[["variant"] + [f"fidelity_{t:g}m" for t in thresholds[1:]] +
                    ["err_mean_m", "err_median_m", "err_p95_m", "latency_ms_median"]]
    md = [f"## Experiment C: end-to-end recovery against the ceiling (RQ2b), split = {split}",
          "", f"Matches: {', '.join(match_ids)}. Unit of analysis: round. Headline threshold "
          f"{h:g} m; presence gate {cfg['eval']['presence_gate_m']:g} m.", "",
          _md_table(show), "", "Looser thresholds and positional error (metres):", "",
          _md_table(extra), "", f"Paired comparisons against `{ref}` (difference = other - ref):",
          "", _md_table(comp[["comparison", "metric", "n", "mean_a", "mean_b", "diff", "ci_lo",
                              "ci_hi", "t", "p", "dz"]]) if len(comp) else "(none)", "",
          f"Failure attribution for `{ref}` at {h:g} m:", "", _md_table(attribution), "",
          f"By round phase for `{ref}`:", "", _md_table(phase) if len(phase) else "(none)", ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return summary


def _plots_c(out, summary, errs_by_variant, attribution, h):
    col = f"fidelity_{h:g}m"
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    lo = [float(s.strip("[]").split(",")[0]) for s in summary[col + "_ci"]]
    hi = [float(s.strip("[]").split(",")[1]) for s in summary[col + "_ci"]]
    y = summary[col].to_numpy()
    ax[0].bar(summary["variant"], y, yerr=[y - lo, np.array(hi) - y], capsize=4,
              color="#4a78b5")
    ax[0].plot([-0.5, len(y) - 0.5], [summary["recoverable"].mean()] * 2, "k--", lw=1,
               label="ceiling")
    ax[0].set_ylabel(f"fidelity @ {h:g} m (per-round mean)")
    ax[0].tick_params(axis="x", rotation=30)
    ax[0].legend()
    for label, e in errs_by_variant.items():
        if len(e):
            xs = np.sort(e)
            ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=label)
    ax[1].axvline(h, color="k", ls="--", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("positional error (m)")
    ax[1].set_ylabel("cumulative share")
    ax[1].legend(fontsize=7)
    ax[2].bar(attribution["stage"], attribution["share_of_failures"], color="#e0a030")
    ax[2].set_ylabel("share of failures")
    fig.tight_layout()
    fig.savefig(out / "fig_expC.png", dpi=200)
    plt.close(fig)


# ======================================================================================
# report
# ======================================================================================
def report(cfg):
    res = p(cfg, "results")
    parts = ["# CS2Vision results", "",
             "Generated by `cs2v report`. Every number below comes from the CSV files in "
             "the same folder; nothing is typed in by hand.", ""]
    ds = p(cfg, "dataset") / "summary.csv"
    if ds.exists():
        parts += ["## Dataset", "", _md_table(pd.read_csv(ds)), ""]
    sync_rows = []
    from .sync import sync_summary
    for mid in list_matches(cfg):
        m = Match(cfg, mid)
        if m.sync_path.exists():
            sync_rows.append({"match": mid, **sync_summary(m.sync())})
    if sync_rows:
        parts += ["## Synchronisation", "", _md_table(pd.DataFrame(sync_rows)), ""]
    lc = res / "label_check" / "label_check.csv"
    if lc.exists():
        d = pd.read_csv(lc)
        bad = pd.to_numeric(d["bad_boxes"], errors="coerce")
        if bad.notna().all() and len(d):
            parts += ["## Manual label check", "",
                      f"{len(d)} images, {int(d['n_boxes'].sum())} boxes, "
                      f"{int(bad.sum())} judged wrong: label error rate "
                      f"{bad.sum() / max(1, d['n_boxes'].sum()):.4f}.", ""]
        else:
            parts += ["## Manual label check", "", "label_check.csv not fully scored yet.", ""]
    for e in ("expA", "expB", "expC"):
        f = res / e / "summary.md"
        if f.exists():
            parts += [f.read_text(), ""]
    (res / "RESULTS.md").write_text("\n".join(parts))
    print(f"wrote {res / 'RESULTS.md'}")
