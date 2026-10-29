"""SSL fine-tuning benchmark (Gate 5, T5.7).

The Gate 3 picker (pipeline v2, multi-class, scale-conditioned) trained from three starting
points, with few and with all training tomograms:

* ``scratch``: random initialisation;
* ``ssl_v2``: warm start from SSL v2 (pretrained on all training tomograms, no labels);
* ``ssl_v1``: warm start from legacy SSL v1 (same tomograms; its chunking needs one species
  for the fixed-particle rescale, ``--ssl_v1_class``).

For ``n`` training tomograms (``--n_train``, e.g. ``1,2,all``) and each seed, ``n`` tomograms
are drawn (seeded) from the training set, plus one more for validation (the pipeline splits
train/val by tomogram and needs two tomograms). Every arm is trained with the same epochs on
the same subset and scored on the held-out ``--test_tomos`` (per-class F-beta, beta 4, at the
global threshold 0.3 and at the calibrated thresholds; ``benchmarks/common.py``).

Acceptance for humans (05_AGENT_SSL.md): recommend SSL v2 as the default initialisation only
if the macro F-beta gain over scratch is >= 0.03 with 1 or 2 training tomograms and >= -0.01
with all; ``report.json`` lists the gains.

Usage (from the repository root)::

    python benchmarks/ssl_finetune.py --tomo_dir data/czii/tomograms \\
        --star data/czii/coords/all_objects.star --test_tomos TS_5_4,TS_69_2 \\
        [--ssl_v2 ssl_v2.ckpt] [--ssl_v1 ssl_v1.ckpt] [--epochs 30] [--seeds 0,1,2] [--n_train 1,2,all]
    python benchmarks/ssl_finetune.py --tiny   # smoke run on small phantoms, 1 epoch, 1 seed
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.common import (  # noqa: E402
    ClassInfo,
    TestTomogram,
    TrainSet,
    TrainSettings,
    _log,
    evaluate_v2_arm,
    isolated_main_config,
    save_json,
    train_v2,
)
from benchmarks.ssl_common import pretrain_ssl, subset_trainset, tomogram_stems  # noqa: E402
from benchmarks.ssl_probe import DEFAULT_DIAMETERS, DEFAULT_NAMES, TINY_NAMES, TINY_SPECIES, parse_classes  # noqa: E402

ARMS = ("scratch", "ssl_v2", "ssl_v1")
MODES = ("global_0.3", "calibrated")


def _with_epochs(settings: TrainSettings, n_epochs: Optional[int]) -> TrainSettings:
    return settings if n_epochs is None else dataclasses.replace(settings, n_epochs=n_epochs)


def _macro(per_class: Dict) -> float:
    return float(np.mean([v["fbeta"] for v in per_class.values()]))


def run(out_dir: Path, data: TrainSet, tests: Sequence[TestTomogram], classes: Sequence[ClassInfo],
        settings_by_seed: Dict[int, TrainSettings], n_train: Sequence[str], ssl_v2: Optional[Path],
        ssl_v1: Optional[Path], v1_class: ClassInfo, arms: Sequence[str],
        ssl_epochs: Optional[Dict[int, Optional[int]]] = None) -> Dict:
    ssl_epochs = ssl_epochs or {}
    first = next(iter(settings_by_seed.values()))
    if "ssl_v2" in arms:
        ssl_v2 = ssl_v2 or pretrain_ssl(out_dir / "pretrain_ssl_v2", data, 2, _with_epochs(first, ssl_epochs.get(2)))
    if "ssl_v1" in arms:
        ssl_v1 = ssl_v1 or pretrain_ssl(out_dir / "pretrain_ssl_v1", data, 1, _with_epochs(first, ssl_epochs.get(1)),
                                       v1_class=v1_class)
    starts = {"scratch": None, "ssl_v2": ssl_v2, "ssl_v1": ssl_v1}
    stems = tomogram_stems(data.tomo_dir)
    runs = []
    for n in n_train:
        k = len(stems) if n == "all" else min(int(n) + 1, len(stems))
        for seed, settings in settings_by_seed.items():
            chosen = sorted(np.random.default_rng(seed).choice(stems, size=k, replace=False).tolist())
            label = f"n{n}_seed{seed}"
            run_dir = out_dir / label
            subset = subset_trainset(data.tomo_dir, data.star_path, chosen, run_dir / "train")
            arm_res = {}
            for arm in arms:
                _log(f"{label}: training {arm} (warm start {starts[arm]})")
                ckpt = train_v2(run_dir / arm, run_dir / "v2_data", subset, classes, settings, True, starts[arm])
                with isolated_main_config():
                    arm_res[arm] = evaluate_v2_arm(ckpt, tests, classes, run_dir / arm / "predictions")
                arm_res[arm]["macro_fbeta"] = {m: _macro(arm_res[arm]["scoring"][m]["per_class"]) for m in MODES}
            runs.append({"label": label, "n_train": n, "seed": seed, "tomograms": chosen, "arms": arm_res})
    return {"runs": runs, "summary": summarise(runs, arms),
            "ssl_v2_checkpoint": str(ssl_v2) if ssl_v2 else None, "ssl_v1_checkpoint": str(ssl_v1) if ssl_v1 else None}


def summarise(runs: List[Dict], arms: Sequence[str]) -> Dict:
    """Mean and sd of the macro F-beta per ``n_train`` and arm, and the gain over scratch."""
    out = {}
    for n in dict.fromkeys(r["n_train"] for r in runs):
        rows = [r for r in runs if r["n_train"] == n]
        entry = {}
        for mode in MODES:
            m = {}
            for arm in arms:
                vals = [r["arms"][arm]["macro_fbeta"][mode] for r in rows]
                m[arm] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "n_runs": len(vals)}
                if arm != "scratch" and "scratch" in arms:
                    gains = [r["arms"][arm]["macro_fbeta"][mode] - r["arms"]["scratch"]["macro_fbeta"][mode]
                             for r in rows]
                    m[arm]["gain_over_scratch"] = float(np.mean(gains))
            entry[mode] = m
        out[str(n)] = entry
    return out


def render_markdown(report: Dict) -> str:
    lines = [f"# SSL fine-tuning benchmark ({'tiny' if report['tiny'] else 'full'})", "",
             "Macro F-beta (beta 4) of the pipeline-v2 picker, mean ± sd over seeds; gain = arm − scratch.", ""]
    for mode in MODES:
        lines += [f"## {mode}", "", "| n_train | arm | macro F-beta | gain |", "|---|---|---|---|"]
        for n, entry in report["result"]["summary"].items():
            for arm, v in entry[mode].items():
                gain = v.get("gain_over_scratch")
                lines.append(f"| {n} | {arm} | {v['mean']:.3f} ± {v['sd']:.3f} | "
                             f"{'' if gain is None else f'{gain:+.3f}'} |")
        lines.append("")
    lines += ["Acceptance (humans): SSL v2 gain >= 0.03 with 1 or 2 training tomograms and >= -0.01 with all."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiny", action="store_true", help="Smoke run on small phantoms, 1 epoch, 1 seed")
    parser.add_argument("--tomo_dir", type=Path, default=None)
    parser.add_argument("--star", type=Path, default=None)
    parser.add_argument("--test_tomos", default=None, help="Comma-separated tomogram stems held out for scoring")
    parser.add_argument("--class_diameters", default=DEFAULT_DIAMETERS)
    parser.add_argument("--class_names", default=DEFAULT_NAMES)
    parser.add_argument("--ssl_v2", type=Path, default=None)
    parser.add_argument("--ssl_v1", type=Path, default=None)
    parser.add_argument("--ssl_v1_class", type=int, default=4)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--n_train", default="1,2,all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--ssl_v1_epochs", type=int, default=None, help="SSL v1 pretraining epochs (default: --epochs)")
    parser.add_argument("--ssl_v2_epochs", type=int, default=None,
                        help="SSL v2 pretraining epochs (default: --epochs). An SSL v2 epoch has ~3x fewer samples than v1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--calibration_min_particles", type=int, default=20)
    parser.add_argument("--class_head", choices=["sigmoid", "softmax"], default="sigmoid",
                        help="Class head of the pipeline-v2 pickers (train.particles.class_head)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    name = f"ssl_finetune_{'tiny' if args.tiny else 'full'}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out or (REPO / "benchmarks" / "out" / name)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    n_train = [s.strip() for s in args.n_train.split(",") if s.strip()]

    if args.tiny:
        from utils.phantoms import make_dataset

        species = TINY_SPECIES
        train = make_dataset(out_dir / "data_train", n_tomos=3, shape=(64, 128, 128), apix=10.0, species=species,
                             wedge_deg=60.0, snr=1.0, seed=0)
        test = make_dataset(out_dir / "data_test", n_tomos=1, shape=(64, 128, 128), apix=10.0, species=species,
                            wedge_deg=60.0, snr=1.0, seed=500, tomo_prefix="TEST_")
        data = TrainSet(train.tomo_dir, train.star_path)
        tests = [TestTomogram(path=p, apix=10.0, coords=c) for p, c in zip(test.tomo_paths, test.coords)]
        classes = [ClassInfo(n, TINY_NAMES[n], d) for n, d, _, _ in species]
        seeds = [0]
        n_train = ["1"] if args.n_train == "1,2,all" else n_train
        mk = lambda s: TrainSettings(n_epochs=1, batch_size=2, tiny=True, seed=s, calibration_min_particles=1, class_head=args.class_head,  # noqa: E731
                                     swin=dict(feature_size=12, use_checkpoint=False))
        v1_class = classes[1]
    else:
        if args.tomo_dir is None or args.star is None or not args.test_tomos:
            parser.error("--tomo_dir, --star and --test_tomos are required (or --tiny)")
        from benchmarks.czii_multiclass import split_by_tomogram

        test_tomos = [s.strip() for s in args.test_tomos.split(",") if s.strip()]
        data, tests = split_by_tomogram(args.tomo_dir, args.star, test_tomos, out_dir / "data")
        classes = parse_classes(args.class_diameters, args.class_names)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        mk = lambda s: TrainSettings(n_epochs=args.epochs, batch_size=args.batch_size, seed=s,  # noqa: E731
                                     calibration_min_particles=args.calibration_min_particles,
                                     class_head=args.class_head)
        v1_class = next(c for c in classes if c.number == args.ssl_v1_class)

    result = run(out_dir, data, tests, classes, {s: mk(s) for s in seeds}, n_train, args.ssl_v2, args.ssl_v1,
                 v1_class, arms, ssl_epochs=None if args.tiny else {1: args.ssl_v1_epochs, 2: args.ssl_v2_epochs})
    report = {"benchmark": "ssl_finetune", "tiny": bool(args.tiny),
              "classes": [{"number": c.number, "name": c.name, "diameter_ang": c.diameter_ang} for c in classes],
              "settings": {"tomo_dir": str(data.tomo_dir), "star": str(data.star_path),
                           "test_tomograms": [str(t.path) for t in tests], "n_train": n_train, "seeds": seeds,
                           "arms": arms, "ssl_v1_class": v1_class.number,
                           "note": "n training tomograms + 1 validation tomogram per run (pipeline split)"},
              "result": result}
    path = save_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(render_markdown(report))
    print(f"Wrote {path} and {path.with_name('report.md')}")


if __name__ == "__main__":
    main()
