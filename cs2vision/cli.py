"""cs2v: command-line entry point. Run `cs2v <command> -h` for options.

Typical order (see GUIDE.md):
  parse-demo -> calibrate -> align -> check -> ingest   (per match)
  split -> autolabel -> train -> exp-a -> exp-b -> exp-c -> report
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import load_config, p
from .match import Match, list_matches


def _matches(cfg, ids):
    if ids in (None, [], ["all"]):
        return [Match(cfg, m) for m in list_matches(cfg)]
    return [Match(cfg, m) for m in ids]


def cmd_parse_demo(cfg, a):
    from .groundtruth import parse_demo
    from .units import MapMeta
    meta = MapMeta.from_cfg(cfg)
    for m in _matches(cfg, a.match):
        if (m.dir / "gt_ticks.parquet").exists() and not a.force:
            print(f"[{m.id}] already parsed (use --force to redo)")
            continue
        gt_path, rounds_path = parse_demo(m.demo, m.dir, meta)
        import pandas as pd
        r = pd.read_csv(rounds_path)
        print(f"[{m.id}] {len(r)} rounds -> {gt_path.name}, {rounds_path.name}")


def cmd_calibrate(cfg, a):
    from .calibrate import calibrate
    m = Match(cfg, a.match[0])
    rect = tuple(int(v) for v in a.rect.split(",")) if a.rect else None
    src = Match(cfg, a.copy_from) if a.copy_from else None
    calibrate(m, a.time, rect=rect, blip_px=a.blip_px, copy_from=src)


def cmd_align(cfg, a):
    from .sync import align, sync_summary
    for m in _matches(cfg, a.match):
        _, sync = align(m, iterations=a.iterations, model=a.model)
        print(f"[{m.id}] {sync_summary(sync)}")


def cmd_check(cfg, a):
    from .viz import check_sheet
    for m in _matches(cfg, a.match):
        check_sheet(m, n=a.n)


def cmd_ingest(cfg, a):
    from .ingest import ingest_match
    for m in _matches(cfg, a.match):
        ingest_match(m)


def cmd_split(cfg, a):
    from .autolabel import make_splits
    df = make_splits(cfg, seed=a.seed, overwrite=a.force)
    print(df.to_string(index=False))
    print(f"-> {p(cfg, 'splits')} (edit by hand if you want a different assignment)")


def cmd_autolabel(cfg, a):
    from .autolabel import build_dataset
    build_dataset(cfg, n_check=a.n_check, seed=a.seed)


def cmd_train(cfg, a):
    from .train import train_all
    configs = a.configs.split(",") if a.configs else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    for w in train_all(cfg, configs, seeds):
        print(w)


def cmd_exp_a(cfg, a):
    from .experiments import experiment_a
    experiment_a(cfg, a.calib_match, a.demos, a.step_ticks)


def cmd_exp_b(cfg, a):
    from .experiments import experiment_b
    experiment_b(cfg, a.weights_dir)


def cmd_exp_c(cfg, a):
    from .experiments import experiment_c
    experiment_c(cfg, a.weights or None, split=a.split, include_blob=not a.no_blob)


def cmd_report(cfg, a):
    from .experiments import report
    report(cfg)


def cmd_selftest(cfg, a):
    """Build synthetic matches and run the whole pipeline on them (no YOLO)."""
    import yaml

    from .synth import make_synthetic_match
    from .units import MapMeta
    root = Path(a.dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    (root / "data" / "matches").mkdir(parents=True)
    scfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    scfg["paths"] = {"matches": "data/matches", "splits": "data/splits.csv",
                     "dataset": "work/dataset", "runs": "work/runs", "results": "results"}
    (root / "config.yaml").write_text(yaml.safe_dump(scfg, sort_keys=False))
    cfg2 = load_config(root / "config.yaml")
    meta = MapMeta.from_cfg(cfg2)
    for i in range(a.matches):
        mid = f"synth{i + 1}"
        info = make_synthetic_match(root / "data" / "matches" / mid, meta, seed=i,
                                    n_rounds=a.rounds)
        print(f"built {mid}")
        m = Match(cfg2, mid)
        x, y, w, h = info["rect"]
        from .calibrate import calibrate
        calibrate(m, info["true_starts"][0] + 5, rect=(x + 3, y - 2, w - 4, h + 5),
                  blip_px=info["blip_px"])
    steps = [["align"], ["check"], ["ingest"], ["split", "--force"], ["autolabel"],
             ["exp-a"], ["exp-c", "--split", "test"], ["report"]]
    for s in steps:
        print(f"\n$ cs2v {' '.join(s)}")
        main(["--config", str(root / "config.yaml"), *s])
    print(f"\nselftest finished. Outputs in {root}")


def build_parser():
    ap = argparse.ArgumentParser(prog="cs2v", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, match=True, many=True):
        sp = sub.add_parser(name, help=help_)
        if match:
            sp.add_argument("match", nargs="*" if many else 1,
                            help="match folder name(s); omit or 'all' for every match")
        sp.set_defaults(fn=fn)
        return sp

    s = add("parse-demo", cmd_parse_demo, "demo -> gt_ticks.parquet + rounds.csv")
    s.add_argument("--force", action="store_true")
    s = add("calibrate", cmd_calibrate, "mark the radar and a blip on one frame", many=False)
    s.add_argument("--time", required=True, help="video time of a frame with the radar visible")
    s.add_argument("--rect", help="x,y,w,h of the radar image, skips the GUI box")
    s.add_argument("--blip-px", type=float, help="blip diameter in px, skips the GUI box")
    s.add_argument("--copy-from", help="reuse the calibration of another match (same overlay)")
    s = add("align", cmd_align, "auto-sync every round and refine the calibration")
    s.add_argument("--iterations", type=int, default=2)
    s.add_argument("--model", choices=["affine", "homography"], default="affine")
    s = add("check", cmd_check, "write check_sheet.png with demo positions over the radar")
    s.add_argument("-n", type=int, default=12)
    add("ingest", cmd_ingest, "sample live-round frames and crop the radar")
    s = add("split", cmd_split, "assign matches to train/val/test", match=False)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s = add("autolabel", cmd_autolabel, "build the YOLO dataset and the label-check sample",
            match=False)
    s.add_argument("--n-check", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s = add("train", cmd_train, "two-phase YOLOv8 training", match=False)
    s.add_argument("--configs", help="comma list, default all in config.yaml")
    s.add_argument("--seeds", help="comma list, default train.seeds")
    s = add("exp-a", cmd_exp_a, "Experiment A: channel ceiling", match=False)
    s.add_argument("--calib-match", help="match whose calibration gives radar/blip size")
    s.add_argument("--demos", help="folder of extra Mirage demos (no video needed)")
    s.add_argument("--step-ticks", type=int, default=16)
    s = add("exp-b", cmd_exp_b, "Experiment B: detector on the test split", match=False)
    s.add_argument("--weights-dir")
    s = add("exp-c", cmd_exp_c, "Experiment C: end-to-end vs ceiling", match=False)
    s.add_argument("--weights", nargs="*", help="weights files; default work/weights/*_s0.pt")
    s.add_argument("--split", default="test")
    s.add_argument("--no-blob", action="store_true", help="skip the blob baseline")
    add("report", cmd_report, "collect all summaries into results/RESULTS.md", match=False)
    s = add("selftest", cmd_selftest, "synthetic end-to-end dry run", match=False)
    s.add_argument("--dir", default="work/selftest")
    s.add_argument("--matches", type=int, default=4)
    s.add_argument("--rounds", type=int, default=4)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    cfg = load_config(a.config)
    a.fn(cfg, a)


if __name__ == "__main__":
    main()
