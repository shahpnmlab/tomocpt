"""Synthetic multi-size benchmark (Gate 4, T4.6).

Phantom tomograms (``tests/utils/phantoms.py``) at 10 A/px with a +-60 deg missing wedge and
four species from 80 to 500 A, so two working voxel sizes (levels) are needed: solid spheres
of 80, 250 and 500 A and hollow shells of 250 A (same size as the 250 A spheres, different
shape). Two SNR levels and three seeds; every (SNR, seed) pair is one run with its own
training and test tomograms, on which the four arms of ``benchmarks/common.py`` are trained
with the same epochs and scored.

Usage (from the repository root)::

    python benchmarks/synthetic_multisize.py            # full benchmark (hours on one GPU)
    python benchmarks/synthetic_multisize.py --tiny     # smoke run: all arms, 1 epoch, 1 run

Output: ``benchmarks/out/<run>/report.json`` and ``report.md`` (``benchmarks/report.py``).
Arm (d) warm-starts from the public TomoCPT model given with ``--pretrained`` (or
``$TOMOCPT_ZENODO_CKPT``). A tiny run uses a smaller network than the public model, so it
always warm-starts arm (d) from the tiny v1 model of arm (a) instead; so does a full run
without the public model. ``report.json`` records the source under ``warm_start``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.common import (  # noqa: E402
    ClassInfo,
    TestTomogram,
    TrainSet,
    TrainSettings,
    public_checkpoint,
    run_arms,
)
from benchmarks.report import write_report  # noqa: E402

APIX = 10.0
WEDGE_DEG = 60.0
#: (class number, name, diameter in A, hollow)
SPECIES = (
    (1, "sphere80", 80.0, False),
    (2, "sphere250", 250.0, False),
    (3, "shell250", 250.0, True),
    (4, "sphere500", 500.0, False),
)
FULL = dict(shape=(96, 256, 256), counts=(20, 6, 6, 3), n_train=6, n_test=3, snrs=(1.0, 0.25),
            seeds=(0, 1, 2), epochs=30)
TINY = dict(shape=(64, 128, 128), counts=(4, 2, 2, 1), n_train=2, n_test=1, snrs=(1.0,),
            seeds=(0,), epochs=1)


def classes() -> List[ClassInfo]:
    return [ClassInfo(number=n, name=name, diameter_ang=d) for n, name, d, _ in SPECIES]


def make_split(root: Path, prefix: str, n: int, shape, counts, snr: float, seed: int):
    from utils.phantoms import make_dataset

    species = tuple((num, d, int(k), hollow) for (num, _, d, hollow), k in zip(SPECIES, counts))
    return make_dataset(root, n_tomos=n, shape=tuple(shape), apix=APIX, species=species,
                        wedge_deg=WEDGE_DEG, snr=snr, seed=seed, tomo_prefix=prefix)


def run(out_dir: Path, tiny: bool, pretrained: Optional[str], arms: Sequence[str], epochs: Optional[int]) -> Path:
    spec = TINY if tiny else FULL
    settings = TrainSettings(
        n_epochs=int(epochs or spec["epochs"]),
        batch_size=2 if tiny else 8,
        tiny=tiny,
        calibration_min_particles=1 if tiny else 10,
        swin=dict(feature_size=12, use_checkpoint=False) if tiny else {},
    )
    pretrained_path = public_checkpoint(pretrained)
    runs = []
    for snr in spec["snrs"]:
        for seed in spec["seeds"]:
            label = f"snr{snr:g}_seed{seed}"
            run_dir = out_dir / label
            train = make_split(run_dir / "data_train", "TS_", spec["n_train"], spec["shape"], spec["counts"],
                               snr, seed=1000 * seed)
            test = make_split(run_dir / "data_test", "TEST_", spec["n_test"], spec["shape"], spec["counts"],
                              snr, seed=1000 * seed + 500)
            tests = [TestTomogram(path=p, apix=APIX, coords=c) for p, c in zip(test.tomo_paths, test.coords)]
            res = run_arms(run_dir, TrainSet(train.tomo_dir, train.star_path), tests, classes(), settings,
                           pretrained_path, arms)
            runs.append({"label": label, "group": f"snr={snr:g}", "snr": snr, "seed": seed,
                         "warm_start": res["warm_start"], "arms": res["arms"]})
    raw = {
        "benchmark": "synthetic_multisize",
        "tiny": bool(tiny),
        "classes": [{"number": c.number, "name": c.name, "diameter_ang": c.diameter_ang} for c in classes()],
        "settings": {"apix": APIX, "wedge_deg": WEDGE_DEG, "species": [list(s) for s in SPECIES],
                     "shape": list(spec["shape"]), "counts": list(spec["counts"]),
                     "n_train_tomograms": spec["n_train"], "n_test_tomograms": spec["n_test"],
                     "snrs": list(spec["snrs"]), "seeds": list(spec["seeds"]), "n_epochs": settings.n_epochs,
                     "batch_size": settings.batch_size, "swin": settings.swin},
        "warm_start": runs[0]["warm_start"] if runs else None,
        "notes": [
            "Scores: tomocpt.metrics matching within half the class radius, counts summed over the "
            "test tomograms of a run; mean and sd over runs (seeds x SNR levels; arms_by_group per SNR).",
            "Wall time per tomogram includes the model load; arm (a) sums its species models.",
            "Arm (a) has no cross-class suppression (separate models, today's practice).",
        ],
        "runs": runs,
    }
    return write_report(raw, out_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiny", action="store_true", help="Smoke run: all arms, 1 epoch, one small run")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default benchmarks/out/<run>)")
    parser.add_argument("--pretrained", default=None, help="Public TomoCPT checkpoint for arm (d)")
    parser.add_argument("--arms", default="a,b,c,d", help="Comma-separated arms to run")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs of every arm")
    args = parser.parse_args(argv)
    name = f"synthetic_multisize_{'tiny' if args.tiny else 'full'}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out or (REPO / "benchmarks" / "out" / name)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    path = run(out_dir, args.tiny, args.pretrained, arms, args.epochs)
    print(f"Wrote {path} and {path.with_name('report.md')}")


if __name__ == "__main__":
    main()
