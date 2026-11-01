"""Gate 6 benchmark helpers: filament centreline scores.

``centerline_scores(pred_polylines, true_polylines, tolerance_vox)`` is length-weighted:
precision is the fraction of predicted length within ``tolerance_vox`` of a true centreline,
recall the fraction of true length within ``tolerance_vox`` of a predicted one. Polylines are
``[K, 3]`` ``x, y, z`` voxels.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _densify(p: np.ndarray, step: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    if len(p) < 2:
        return p
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + t * (b - a))
    return np.concatenate(out)


def _point_to_polylines(points: np.ndarray, polylines: Sequence[np.ndarray]) -> np.ndarray:
    """Exact distance from each point to the nearest segment of any polyline."""
    best = np.full(len(points), np.inf)
    for p in polylines:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        if len(p) == 1:
            best = np.minimum(best, np.linalg.norm(points - p[0], axis=1))
            continue
        for a, b in zip(p[:-1], p[1:]):
            ab = b - a
            L2 = float(ab @ ab)
            if L2 == 0:
                d = np.linalg.norm(points - a, axis=1)
            else:
                t = np.clip((points - a) @ ab / L2, 0.0, 1.0)
                d = np.linalg.norm(points - (a + t[:, None] * ab), axis=1)
            best = np.minimum(best, d)
    return best


def _covered_length(src: Sequence[np.ndarray], ref: Sequence[np.ndarray], tol: float, step: float):
    """``(length of src within tol of ref, total length of src)``; sub-segments of ``step``."""
    total, near = 0.0, 0.0
    for p in src:
        d = _densify(p, step)
        if len(d) < 2:
            continue
        mid = 0.5 * (d[:-1] + d[1:])
        seg = np.linalg.norm(np.diff(d, axis=0), axis=1)
        total += float(seg.sum())
        if ref:
            near += float(seg[_point_to_polylines(mid, ref) <= tol].sum())
    return near, total


def centerline_scores(pred_polylines: Sequence[np.ndarray], true_polylines: Sequence[np.ndarray],
                      tolerance_vox: float, step: float = 0.25) -> Dict[str, float]:
    """``{"precision", "recall", "f1"}`` (length-weighted, see the module docstring)."""
    pred = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in pred_polylines if len(p)]
    true = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in true_polylines if len(p)]
    tp_p, len_p = _covered_length(pred, true, float(tolerance_vox), step)
    tp_t, len_t = _covered_length(true, pred, float(tolerance_vox), step)
    precision = tp_p / len_p if len_p > 0 else 0.0
    recall = tp_t / len_t if len_t > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "pred_length": float(len_p), "true_length": float(len_t)}


# ------------------------------------------------------------------------------------------
# T6.5 benchmark driver
# ------------------------------------------------------------------------------------------
"""
Usage (data from ``experiments/fetch_filament_data.py``: ``<root>/<dataset id>/tomograms/<run>.mrc``
and ``<root>/<dataset id>/masks/<object>/<run>.mrc``)::

    python benchmarks/filaments.py --data_root data/filaments --train_sets 10490,10477 \\
        --test_sets 10513 --objects microtubule --epochs 60 --out benchmarks/out/filaments_<date> \\
        [--particles_star all_particles.star --particle_classes 1:ribosome:300] [--max_runs 40]

Arms: ``filaments`` (filament classes only) and, with ``--particles_star``, ``joint``
(filaments + particles in one model) and ``particles`` (particle classes only). Each held-out
run of a training set (``--test_frac``, split by run, seeded) and every run of a test set is
scored: centreline precision / recall / F1 (length-weighted, within 0.5 x diameter), point F1 at
``filament_spacing_ang``, particle F-beta, time and peak GPU memory. Test-set numbers (labels
not ground truth, e.g. DS-10513) are reported separately. The particle STAR must name tomograms
as staged here: ``<dataset id>_<run>``.
"""

DEFAULT_DIAMETERS = {"microtubule": 250.0, "actin": 70.0}


def _stage(data_root, sets, objects, out, test_frac, seed, max_runs, all_test=False):
    """Symlink tomograms and masks into ``out/{train,test}``; names ``<dataset>_<run>``."""
    import os
    from pathlib import Path

    rng = np.random.default_rng(seed)
    staged = {"train": [], "test": []}
    for ds in sets:
        root = Path(data_root) / str(ds)
        runs = sorted(p.stem for p in (root / "tomograms").glob("*.mrc"))
        runs = [r for r in runs if all((root / "masks" / o / f"{r}.mrc").is_file() for o in objects)]
        if max_runs:
            runs = runs[: int(max_runs)]
        order = rng.permutation(len(runs))
        n_test = len(runs) if all_test else max(1, int(round(test_frac * len(runs))))
        test_idx = set(order[:n_test].tolist())
        for i, r in enumerate(runs):
            split = "test" if i in test_idx else "train"
            name = f"{ds}_{r}"
            dst = Path(out) / split / "tomograms" / f"{name}.mrc"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink((root / "tomograms" / f"{r}.mrc").resolve(), dst)
            for o in objects:
                m = Path(out) / split / "masks" / o / f"{name}.mrc"
                m.parent.mkdir(parents=True, exist_ok=True)
                if not m.exists():
                    os.symlink((root / "masks" / o / f"{r}.mrc").resolve(), m)
            staged[split].append({"name": name, "dataset": str(ds), "tomogram": str(dst)})
    return staged


def _train_arm(work, stage_dir, fil_classes, part_classes, particles_star, settings):
    from pathlib import Path

    from benchmarks.common import _find_checkpoint, _run_cli, _train_section, _write_config

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    classes = list(part_classes) + list(fil_classes)
    prep = {
        "raw_data_dir": str(Path(stage_dir) / "tomograms"),
        "training_data_dir": str(work / "training_data"),
        # the CLI requires the option; labels.run reads the literal "None" as "no particle points"
        "coordinate_files": str(particles_star) if part_classes else "None",
        "coordinate_file_type": "star",
        "particle_length_ang": repr(max(float(d) for _, _, d in classes)),
        "class_id": "all",
        "multiclass": True,
        "class_diameters": ",".join(f"{n}:{float(d):g}" for n, _, d in classes),
        "class_names": ",".join(f"{n}:{name}" for n, name, _ in classes),
        "label_border_policy": "clip",
    }
    if fil_classes:
        prep["filament_masks"] = ",".join(f"{n}:{Path(stage_dir) / 'masks' / name}" for n, name, _ in fil_classes)
    train = _train_section(work, settings, 2, conditioning=True)
    cfg = {"shared": {"training_data_dir": str(work / "training_data"), "chunks_dir": str(work / "chunks")},
           "prepData": prep, "train": train,
           "infer": {"predictions_dir": str(work / "predictions"),
                     "calibration_min_particles": int(settings.calibration_min_particles)}}
    config = _write_config(work / "config.yaml", cfg)
    if not (work / "training_data").is_dir():
        _run_cli("prepare_data", config, work / "prepare_data.log")
    _run_cli("train", config, work / "train.log")
    return _find_checkpoint(work / "model")


def _point_f1(pred_polys, true_polys, spacing_vox, radius_vox):
    from benchmarks.common import _counts
    from tomocpt.labels.filaments import resample_polyline

    def pts(polys):
        out = [resample_polyline(p, spacing_vox) for p in polys if len(p) >= 2]
        return np.concatenate(out) if out else np.zeros((0, 3))

    tp, fp, fn = _counts(pts(pred_polys), pts(true_polys), radius_vox)
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / max(1, 2 * tp + fp + fn)}


def _evaluate(ckpt, tests, objects_dir_of, fil_names, particles_star, out_dir):
    import time
    from pathlib import Path

    import mrcfile
    import pandas as pd
    import torch

    from benchmarks.common import _counts, _device, _gpu_peak_mb, _infer_config, _reset_gpu_peak, isolated_main_config
    from tomocpt.labels.filaments import mask_to_centerlines
    from tomocpt.networks.inference import InferenceModel
    from tomocpt.predict.calibration import read_calibration
    from tomocpt.predict.filaments import is_filament, trace_class_maps
    from tomocpt.predict.multiclass import peaks_from_class_maps, predict_class_maps, resolve_thresholds

    device = _device()
    cfg = _infer_config(Path(out_dir))  # v2 defaults: spacing 0.5 x d, no cross-class suppression
    with isolated_main_config():
        _reset_gpu_peak()
        model = InferenceModel.load_from_checkpoint(str(ckpt), map_location=device, strict=True).eval()
        table = model.class_table()
        thr = {c: v[0] for c, v in resolve_thresholds(table, cfg, read_calibration(ckpt)).items()}
        stars = None
        if particles_star:
            import starfile
            d = starfile.read(str(particles_star))
            stars = d["particles"] if isinstance(d, dict) else d
        per_tomo, t_maps = [], 0.0
        for t in tests:
            t0 = time.perf_counter()
            with torch.no_grad():
                maps, apix = predict_class_maps(model, Path(t["tomogram"]), cfg, device,
                                                classes=list(range(len(table))))
            fils = trace_class_maps(maps, table, apix, cfg, thr, t["name"])
            peaks, _ = peaks_from_class_maps(maps, table, apix, cfg, thr, t["name"], device)
            t_maps += time.perf_counter() - t0
            rec = {"name": t["name"], "dataset": t["dataset"], "filaments": {}, "particles": {}}
            for c, row in enumerate(table):
                if is_filament(row):
                    with mrcfile.open(str(objects_dir_of(t, row.name)), permissive=True) as m:
                        mask = np.asarray(m.data) > 0
                    true = mask_to_centerlines(mask, apix, row.diameter_ang, 0.5 * row.diameter_ang)
                    pred = [np.asarray(f["points"]) for f in fils.get(c, [])]
                    tol = 0.5 * row.diameter_ang / apix
                    s = centerline_scores(pred, true, tol)
                    s["points"] = _point_f1(pred, true, cfg.filament_spacing_ang / apix, tol)
                    s["n_pred"], s["n_true"] = len(pred), len(true)
                    rec["filaments"][row.name] = s
                elif stars is not None:
                    g = stars[(stars["rlnMicrographName"].astype(str) == t["name"])
                              & (stars["rlnClassNumber"].astype(int) == int(row.number))]
                    true = g[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float)
                    p = peaks.get(c, np.zeros((0, 4)))
                    rec["particles"][row.name] = dict(zip(("tp", "fp", "fn"), _counts(
                        p[:, [2, 1, 0]], true, 0.5 * row.diameter_ang / apix)))
            per_tomo.append(rec)
        peak = _gpu_peak_mb()
        del model
    return {"per_tomogram": per_tomo, "seconds_per_tomogram": t_maps / max(1, len(tests)),
            "peak_gpu_memory_mb": peak, "thresholds": {table[c].name: v for c, v in thr.items()}}


def _summary(ev, beta=4.0):
    """Length-weighted centreline P/R/F1 and point F1 per filament class, particle F-beta per
    class, grouped by dataset."""
    out = {}
    for rec in ev["per_tomogram"]:
        ds = out.setdefault(rec["dataset"], {"filaments": {}, "particles": {}})
        for name, s in rec["filaments"].items():
            a = ds["filaments"].setdefault(name, {"tp_p": 0.0, "len_p": 0.0, "tp_t": 0.0, "len_t": 0.0,
                                                  "tp": 0, "fp": 0, "fn": 0})
            a["tp_p"] += s["precision"] * s["pred_length"]
            a["len_p"] += s["pred_length"]
            a["tp_t"] += s["recall"] * s["true_length"]
            a["len_t"] += s["true_length"]
            for k in ("tp", "fp", "fn"):
                a[k] += s["points"][k]
        for name, s in rec["particles"].items():
            a = ds["particles"].setdefault(name, {"tp": 0, "fp": 0, "fn": 0})
            for k in ("tp", "fp", "fn"):
                a[k] += s[k]
    for ds in out.values():
        for a in ds["filaments"].values():
            P = a["tp_p"] / a["len_p"] if a["len_p"] else 0.0
            R = a["tp_t"] / a["len_t"] if a["len_t"] else 0.0
            a.update(precision=P, recall=R, f1=2 * P * R / (P + R) if P + R else 0.0,
                     point_f1=2 * a["tp"] / max(1, 2 * a["tp"] + a["fp"] + a["fn"]))
        for a in ds["particles"].values():
            b2 = beta * beta
            a["fbeta"] = (1 + b2) * a["tp"] / max(1, (1 + b2) * a["tp"] + b2 * a["fn"] + a["fp"])
    return out


def main(argv=None):
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from benchmarks.common import TrainSettings, _log, save_json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_sets", default="10490,10477")
    ap.add_argument("--test_sets", default="10513")
    ap.add_argument("--objects", default="microtubule")
    ap.add_argument("--diameters", default=None, help="object=diameter_A,... (default microtubule 250, actin 70)")
    ap.add_argument("--particles_star", default=None)
    ap.add_argument("--particle_classes", default=None, help="number:name:diameter_A,...")
    ap.add_argument("--arms", default="filaments,joint,particles")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--max_runs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="feature_size 12 (smoke runs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    objects = [o.strip() for o in a.objects.split(",") if o.strip()]
    diam = dict(DEFAULT_DIAMETERS)
    for kv in (a.diameters or "").split(","):
        if "=" in kv:
            k, v = kv.split("=")
            diam[k.strip()] = float(v)
    part = []
    for e in (a.particle_classes or "").split(","):
        if e.strip():
            n, name, d = e.split(":")
            part.append((int(n), name, float(d)))
    base = max([n for n, _, _ in part] + [0])
    fil = [(base + 100 + i, o, diam[o]) for i, o in enumerate(objects)]
    train_sets = [s for s in a.train_sets.split(",") if s.strip()]
    test_sets = [s for s in a.test_sets.split(",") if s.strip()]
    staged = _stage(a.data_root, train_sets, objects, out / "stage", a.test_frac, a.seed, a.max_runs)
    if test_sets:
        extra = _stage(a.data_root, test_sets, objects, out / "stage_testsets", 1.0, a.seed, a.max_runs, all_test=True)
        staged["testsets"] = extra["test"]
    _log(f"train runs {len(staged['train'])}, held-out {len(staged['test'])}, test sets {len(staged.get('testsets', []))}")
    settings = TrainSettings(n_epochs=a.epochs, batch_size=a.batch_size, tiny=a.tiny,
                             swin={"feature_size": 12} if a.tiny else {}, class_head="softmax")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not a.particles_star:
        arms = [x for x in arms if x == "filaments"]
    spec = {"filaments": (fil, []), "joint": (fil, part), "particles": ([], part)}
    report = {"args": vars(a), "classes": {"filaments": fil, "particles": part}, "splits": staged, "arms": {}}
    for arm in arms:
        f_cls, p_cls = spec[arm]
        t0 = time.time()
        ckpt = _train_arm(out / arm, out / "stage" / "train", f_cls, p_cls, a.particles_star, settings)
        res = {"checkpoint": str(ckpt), "train_seconds": time.time() - t0}

        def mask_of(t, name, root):
            return Path(root) / "masks" / name / f"{t['name']}.mrc"

        for key, root in (("held_out", out / "stage" / "test"), ("test_sets", out / "stage_testsets" / "test")):
            tests = staged["test"] if key == "held_out" else staged.get("testsets", [])
            if not tests:
                continue
            ev = _evaluate(ckpt, tests, lambda t, n, r=root: mask_of(t, n, r), [n for _, n, _ in f_cls],
                           a.particles_star, out / arm / f"pred_{key}")
            res[key] = {"summary": _summary(ev), **ev}
        report["arms"][arm] = res
        save_json(out / "filaments_benchmark.json", report)
        _log(f"arm {arm}: {json.dumps({k: v.get('summary') for k, v in res.items() if isinstance(v, dict)}, default=str)[:2000]}")
    save_json(out / "filaments_benchmark.json", report)
    return report


if __name__ == "__main__":
    main()
