"""SSL embedding probe (Gate 5, T5.7).

Dense embeddings read at the true particle centres, k-NN species classification with a
leave-one-tomogram-out split. Compares three networks:

* ``ssl_v2``: SSL v2 (wedge- and scale-aware) pretrained on the tomograms, no labels;
* ``ssl_v1``: legacy SSL v1 pretrained on the same tomograms (its chunking needs one species
  for the fixed-particle rescale, ``--ssl_v1_class``; the SSL losses use no label);
* ``random_v2``: an untrained SSL v2 network with the same settings.

Features: the ``decoder2`` output (32^3, FiLM-conditioned for v2) and, for v2 models, the
dense projection head; both are reported, per species and balanced. SSL v2 / random networks
read the tomogram at ``apix_ref = clip(apix_native, apix_ref_min, apix_ref_max)``, SSL v1 at
its own training scale (the v1 species at ``desired_particle_pixel_size`` voxels).

Usage (from the repository root)::

    # CZII (data from scripts/download_portal_data.py); pretrains both SSL versions unless
    # checkpoints are given
    python benchmarks/ssl_probe.py --tomo_dir data/czii/tomograms --star data/czii/coords/all_objects.star \\
        [--ssl_v2 ssl_v2.ckpt] [--ssl_v1 ssl_v1.ckpt] [--epochs 30]
    python benchmarks/ssl_probe.py --tiny      # smoke run on small phantoms, 1 epoch

Output: ``benchmarks/out/<run>/report.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.common import ClassInfo, TrainSet, TrainSettings, _device, _log, save_json  # noqa: E402
from benchmarks.ssl_common import (  # noqa: E402
    Embedder,
    coords_of,
    embeddings_at,
    knn_probe,
    pretrain_ssl,
    tomogram_stems,
)

DEFAULT_DIAMETERS = "1:130,2:180,3:260,4:300,5:120,6:270"
DEFAULT_NAMES = "1:beta-amylase,2:beta-galactosidase,3:thyroglobulin,4:ribosome,5:apo-ferritin,6:virus-like-particle"
#: tiny phantoms: (class, diameter A, count, hollow); two species of the same size (solid / hollow)
TINY_SPECIES = ((1, 100.0, 6, False), (2, 200.0, 3, False), (3, 200.0, 3, True), (4, 300.0, 2, False))
TINY_NAMES = {1: "solid100", 2: "solid200", 3: "hollow200", 4: "solid300"}


def parse_classes(diameters: str, names: str) -> List[ClassInfo]:
    from tomocpt.labels.multiclass import parse_class_diameters, parse_class_names

    d = parse_class_diameters(diameters)
    n = parse_class_names(names) if names else {}
    return [ClassInfo(number=k, name=n.get(k, f"class{k}"), diameter_ang=float(v)) for k, v in sorted(d.items())]


def probe(out_dir: Path, data: TrainSet, classes: Sequence[ClassInfo], settings: TrainSettings,
          ssl_v2: Optional[Path], ssl_v1: Optional[Path], v1_class: ClassInfo,
          arms: Sequence[str]) -> Dict:
    t0 = time.time()
    if "ssl_v2" in arms or "random_v2" in arms:
        ssl_v2 = ssl_v2 or pretrain_ssl(out_dir / "pretrain_ssl_v2", data, 2, settings)
    if "ssl_v1" in arms:
        ssl_v1 = ssl_v1 or pretrain_ssl(out_dir / "pretrain_ssl_v1", data, 1, settings, v1_class=v1_class)
    pretrain_s = time.time() - t0
    stems = tomogram_stems(data.tomo_dir)
    coords = {s: coords_of(data.star_path, s) for s in stems}
    names = {c.number: c.name for c in classes}
    device = _device()
    results = {}
    for arm in arms:
        ckpt = ssl_v1 if arm == "ssl_v1" else ssl_v2
        _log(f"probe: {arm} ({ckpt})")
        emb = Embedder(arm, ckpt, device, seed=settings.seed, v1_diameter_ang=v1_class.diameter_ang)
        per_feature: Dict[str, List] = {}
        labels = []
        for s in stems:
            feats, labs = embeddings_at(emb, data.tomo_dir / f"{s}.mrc", coords[s])
            for k, v in feats.items():
                per_feature.setdefault(k, []).append(v)
            labels.append(labs)
        del emb
        arm_res = {}
        for feat, embs in per_feature.items():
            r = knn_probe(embs, labels)
            arm_res[feat] = {"balanced_accuracy": r["balanced"],
                             "per_class": {names.get(c, str(c)): a for c, a in r["per_class"].items()}}
        best = max(arm_res, key=lambda k: arm_res[k]["balanced_accuracy"] or 0.0)
        results[arm] = {"checkpoint": str(ckpt), "best_feature": best, "features": arm_res,
                        "n_particles": int(sum(len(lab) for lab in labels))}
    return {"arms": results, "pretrain_wall_time_s": pretrain_s,
            "ssl_v2_checkpoint": str(ssl_v2) if ssl_v2 else None, "ssl_v1_checkpoint": str(ssl_v1) if ssl_v1 else None}


def render_markdown(report: Dict) -> str:
    lines = [f"# SSL embedding probe ({'tiny' if report['tiny'] else 'full'})", "",
             "Leave-one-tomogram-out 5-NN (cosine) on dense embeddings at the true centres; "
             "balanced accuracy (mean of per-species accuracies).", "",
             "| network | feature | balanced acc. | per species |", "|---|---|---|---|"]
    for arm, r in report["result"]["arms"].items():
        for feat, f in r["features"].items():
            per = ", ".join(f"{k} {v:.2f}" for k, v in f["per_class"].items())
            bal = f["balanced_accuracy"]
            lines.append(f"| {arm} | {feat} | {bal:.3f} | {per} |" if bal is not None else f"| {arm} | {feat} | - | |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiny", action="store_true", help="Smoke run on small phantoms, 1 epoch")
    parser.add_argument("--tomo_dir", type=Path, default=None)
    parser.add_argument("--star", type=Path, default=None)
    parser.add_argument("--class_diameters", default=DEFAULT_DIAMETERS)
    parser.add_argument("--class_names", default=DEFAULT_NAMES)
    parser.add_argument("--ssl_v2", type=Path, default=None, help="SSL v2 checkpoint (pretrained here if absent)")
    parser.add_argument("--ssl_v1", type=Path, default=None, help="SSL v1 checkpoint (pretrained here if absent)")
    parser.add_argument("--ssl_v1_class", type=int, default=4,
                        help="Class number whose diameter sets SSL v1's fixed-particle rescale (default ribosome)")
    parser.add_argument("--arms", default="ssl_v2,ssl_v1,random_v2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    name = f"ssl_probe_{'tiny' if args.tiny else 'full'}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out or (REPO / "benchmarks" / "out" / name)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.tiny:
        from utils.phantoms import make_dataset

        ds = make_dataset(out_dir / "data", n_tomos=3, shape=(96, 128, 128), apix=10.0, species=TINY_SPECIES,
                          wedge_deg=60.0, snr=1.0, seed=0)
        data = TrainSet(ds.tomo_dir, ds.star_path)
        classes = [ClassInfo(n, TINY_NAMES[n], d) for n, d, _, _ in TINY_SPECIES]
        settings = TrainSettings(n_epochs=1, batch_size=2, tiny=True, seed=args.seed,
                                 swin=dict(feature_size=12, use_checkpoint=False))
        v1_class = classes[1]
    else:
        if args.tomo_dir is None or args.star is None:
            parser.error("--tomo_dir and --star are required (or --tiny)")
        data = TrainSet(args.tomo_dir, args.star)
        classes = parse_classes(args.class_diameters, args.class_names)
        settings = TrainSettings(n_epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
        v1_class = next(c for c in classes if c.number == args.ssl_v1_class)

    result = probe(out_dir, data, classes, settings, args.ssl_v2, args.ssl_v1, v1_class, arms)
    report = {"benchmark": "ssl_probe", "tiny": bool(args.tiny),
              "classes": [{"number": c.number, "name": c.name, "diameter_ang": c.diameter_ang} for c in classes],
              "settings": {"tomo_dir": str(data.tomo_dir), "star": str(data.star_path), "n_epochs": settings.n_epochs,
                           "batch_size": settings.batch_size, "swin": settings.swin, "ssl_v1_class": v1_class.number,
                           "knn_k": 5, "split": "leave-one-tomogram-out"},
              "result": result}
    path = save_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(render_markdown(report))
    print(f"Wrote {path} and {path.with_name('report.md')}")


if __name__ == "__main__":
    main()
