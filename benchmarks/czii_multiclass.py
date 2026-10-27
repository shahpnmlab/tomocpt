"""CZII 10440 multi-class benchmark (Gate 4, T4.6).

Six species of the CZII cryo-ET object identification challenge (EMPIAR/CZ portal dataset
10440, 10.012 A/px), split by tomogram. The data are not shipped: download them with
``scripts/download_portal_data.py`` and pass the paths. Every seed is one run; the four arms
of ``benchmarks/common.py`` are trained with the same epochs on the training tomograms and
scored on the test tomograms.

Inputs:

* ``--tomo_dir``: tomograms (``<name>.mrc``);
* ``--star``: one RELION STAR file with every particle of every tomogram
  (``rlnMicrographName`` = tomogram stem, ``rlnCoordinateX/Y/Z`` in voxels of the tomogram,
  ``rlnClassNumber``);
* ``--test_tomos``: comma-separated tomogram stems held out for scoring (the rest train;
  pipeline v1/v2 split it again into train/val by tomogram).

Defaults use the public CZII diameters (A): 1 beta-amylase 130, 2 beta-galactosidase 180,
3 thyroglobulin 260, 4 ribosome 300, 5 apo-ferritin 120, 6 virus-like-particle 270.

Usage::

    python benchmarks/czii_multiclass.py --tomo_dir data/czii/tomograms \\
        --star data/czii/coords/all_objects.star --test_tomos TS_5_4,TS_69_2 \\
        [--pretrained tomocpt_public.ckpt] [--epochs 30] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.common import (  # noqa: E402
    ClassInfo,
    TestTomogram,
    TrainSet,
    TrainSettings,
    public_checkpoint,
    run_arms,
)
from benchmarks.report import write_report  # noqa: E402

DEFAULT_DIAMETERS = "1:130,2:180,3:260,4:300,5:120,6:270"
DEFAULT_NAMES = "1:beta-amylase,2:beta-galactosidase,3:thyroglobulin,4:ribosome,5:apo-ferritin,6:virus-like-particle"


def parse_classes(diameters: str, names: str) -> List[ClassInfo]:
    from tomocpt.labels.multiclass import parse_class_diameters, parse_class_names

    d = parse_class_diameters(diameters)
    n = parse_class_names(names) if names else {}
    return [ClassInfo(number=k, name=n.get(k, f"class{k}"), diameter_ang=float(v)) for k, v in sorted(d.items())]


def split_by_tomogram(tomo_dir: Path, star: Path, test_tomos: Sequence[str], work: Path):
    """Training tomograms (symlinks) and STAR rows in ``work/train``; test tomograms with
    their true coordinates (``{class_number: xyz}``, tomogram voxels)."""
    import mrcfile
    import starfile

    data = starfile.read(star)
    particles = data["particles"] if isinstance(data, dict) else data
    name_col = "rlnMicrographName" if "rlnMicrographName" in particles else "rlnTomoName"
    stems = particles[name_col].astype(str).map(lambda s: Path(s).stem)
    test_set = set(test_tomos)

    train_dir = work / "train" / "tomograms"
    train_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(tomo_dir).glob("*.mrc")):
        if path.stem not in test_set:
            link = train_dir / path.name
            if not link.exists():
                os.symlink(path.resolve(), link)
    train_star = work / "train" / "particles.star"
    out = dict(data) if isinstance(data, dict) else {"particles": data}
    out["particles"] = particles[~stems.isin(test_set)]
    starfile.write(out, train_star, overwrite=True)

    tests = []
    for stem in test_tomos:
        path = Path(tomo_dir) / f"{stem}.mrc"
        if not path.is_file():
            raise FileNotFoundError(f"Test tomogram {path} not found")
        with mrcfile.open(str(path), permissive=True, header_only=True) as mrc:
            apix = float(mrc.voxel_size.x)
        rows = particles[stems == stem]
        coords = {int(c): g[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(np.float64)
                  for c, g in rows.groupby("rlnClassNumber")}
        tests.append(TestTomogram(path=path, apix=apix, coords=coords))
    return TrainSet(train_dir, train_star), tests


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tomo_dir", type=Path, required=True)
    parser.add_argument("--star", type=Path, required=True)
    parser.add_argument("--test_tomos", required=True, help="Comma-separated tomogram stems held out for scoring")
    parser.add_argument("--class_diameters", default=DEFAULT_DIAMETERS)
    parser.add_argument("--class_names", default=DEFAULT_NAMES)
    parser.add_argument("--pretrained", default=None, help="Public TomoCPT checkpoint for arm (d)")
    parser.add_argument("--arms", default="a,b,c,d")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--calibration_min_particles", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = args.out or (REPO / "benchmarks" / "out" / f"czii_multiclass_{time.strftime('%Y%m%d_%H%M%S')}")
    classes = parse_classes(args.class_diameters, args.class_names)
    test_tomos = [s.strip() for s in args.test_tomos.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    pretrained = public_checkpoint(args.pretrained)
    data, tests = split_by_tomogram(args.tomo_dir, args.star, test_tomos, out_dir / "data")

    runs = []
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        settings = TrainSettings(n_epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                                 calibration_min_particles=args.calibration_min_particles)
        res = run_arms(out_dir / f"seed{seed}", data, tests, classes, settings, pretrained, arms)
        runs.append({"label": f"seed{seed}", "seed": seed, "warm_start": res["warm_start"], "arms": res["arms"]})
    raw = {
        "benchmark": "czii_multiclass",
        "tiny": False,
        "classes": [{"number": c.number, "name": c.name, "diameter_ang": c.diameter_ang} for c in classes],
        "settings": {"tomo_dir": str(args.tomo_dir), "star": str(args.star), "test_tomos": test_tomos,
                     "n_epochs": args.epochs, "batch_size": args.batch_size, "seeds": args.seeds},
        "warm_start": runs[0]["warm_start"] if runs else None,
        "notes": ["Seeds label repeated training runs; TomoCPT training is not seeded, and the "
                  "pipeline's train/val split is fixed."],
        "runs": runs,
    }
    path = write_report(raw, out_dir)
    print(f"Wrote {path} and {path.with_name('report.md')}")


if __name__ == "__main__":
    main()
