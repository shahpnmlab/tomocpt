"""Gate 6 benchmark driver for filaments. The centreline metric (``centerline_scores``) lives in
``tomocpt.metrics`` (Gate 6b) and is re-exported here."""

from __future__ import annotations

import sys
from pathlib import Path

# Run as a script from a worktree: this checkout's package, not the installed one.
_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from tomocpt.metrics import _covered_length, _densify, _point_to_polylines, centerline_scores  # noqa: E402,F401


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

Fixed runs of one set (``--train_sets 10490 --test_sets "" --train_runs sample_1,... --val_runs
sample_6,sample_7``): the validation runs are the ``held_out`` report. They also choose the tracing
threshold, so those scores are slightly optimistic. Pictures go to ``<arm>/views/``
(``benchmarks/filament_views.py``).
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


def _stage_fixed(data_root, ds, objects, out, train_runs, val_runs):
    """Fixed runs of one dataset: all linked under ``out/train`` (the chunking splits them by
    ``train.val_stems``). Returns ``{"train": [...], "test": [the validation runs]}``."""
    import os
    from pathlib import Path

    overlap = sorted(set(train_runs) & set(val_runs))
    if overlap:
        raise ValueError(f"Runs in both the training and the validation split: {overlap}")
    root = Path(data_root) / str(ds)
    staged = {"train": [], "test": []}
    for split, runs in (("train", train_runs), ("test", val_runs)):
        for r in runs:
            sources = {Path("tomograms") / f"{ds}_{r}.mrc": root / "tomograms" / f"{r}.mrc"}
            sources.update({Path("masks") / o / f"{ds}_{r}.mrc": root / "masks" / o / f"{r}.mrc" for o in objects})
            for rel, src in sources.items():
                if not src.is_file():
                    raise FileNotFoundError(f"Run {r} of dataset {ds}: {src} is missing")
                dst = Path(out) / "train" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.symlink(src.resolve(), dst)
            staged[split].append({"name": f"{ds}_{r}", "dataset": str(ds),
                                  "tomogram": str(Path(out) / "train" / "tomograms" / f"{ds}_{r}.mrc")})
    return staged


def _stage_negative_masks(data_root, ds, names, out, runs):
    """Links ``<root>/<ds>/masks/<name>/<run>.mrc`` to ``out/train/masks/<name>/<ds>_<run>.mrc``
    (structures that are never a class, e.g. membranes; ``prepData.negative_masks``)."""
    import os
    from pathlib import Path

    for name in names:
        for r in runs:
            src = Path(data_root) / str(ds) / "masks" / name / f"{r}.mrc"
            if not src.is_file():
                raise FileNotFoundError(f"Run {r}: no {name} mask at {src}")
            dst = Path(out) / "train" / "masks" / name / f"{ds}_{r}.mrc"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink(src.resolve(), dst)


def _stage_centerlines(root, ds, objects, out, runs):
    """Links ``<root>/<object>/<run>.mod`` to ``out/train/centerlines/<object>/<dataset>_<run>.mod``."""
    import os
    from pathlib import Path

    for o in objects:
        for r in runs:
            src = Path(root) / o / f"{r}.mod"
            if not src.is_file():
                raise FileNotFoundError(f"Run {r}: no centrelines at {src} (scripts/masks_to_centerlines.py)")
            dst = Path(out) / "train" / "centerlines" / o / f"{ds}_{r}.mod"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink(src.resolve(), dst)


def apply_recipe(train: dict) -> dict:
    """The filament recipe (``HANDOFF_filaments.md`` section 4) on a ``train`` config section."""
    train["chunk_sampling"] = "repeat_factor"
    train.setdefault("augmentation", {}).update(rotation_mode="flips", flip_p=0.5)
    train.setdefault("particles", {}).update(cross_class_neg_weight=0.0)
    train.update(early_stop_rel_delta=0.007, early_stop_patience=10, lr_plateau_match_early_stop=True,
                 lr_plateau_patience=5, weight_averaging="ema")
    return train


def _apply_train_overrides(train: dict, overrides) -> dict:
    """``key.sub=value`` entries (values parsed as YAML) set on the ``train`` section, last wins."""
    import yaml

    for kv in overrides or ():
        key, _, val = kv.partition("=")
        if not key or not _:
            raise ValueError(f"--train_set expects key=value, got '{kv}'")
        node, parts = train, key.strip().split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
    return train


def _train_arm(work, stage_dir, fil_classes, part_classes, particles_star, settings, val_stems=None,
               use_centerlines=False, band_max_vox=None, snapshot_chunks=0, negative_masks=(),
               contrastive_weight=0.0, train_overrides=()):
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
    if fil_classes and use_centerlines:
        prep["filament_coords"] = ",".join(f"{n}:{Path(stage_dir) / 'centerlines' / name}" for n, name, _ in fil_classes)
    elif fil_classes:
        prep["filament_masks"] = ",".join(f"{n}:{Path(stage_dir) / 'masks' / name}" for n, name, _ in fil_classes)
    if negative_masks:
        prep["negative_masks"] = ",".join(f"{name}:{Path(stage_dir) / 'masks' / name}" for name in negative_masks)
    train = apply_recipe(_train_section(work, settings, 2, conditioning=True))
    if val_stems:
        train["val_stems"] = ",".join(val_stems)
    if band_max_vox is not None:
        train["scale"] = {"band_max_vox": float(band_max_vox)}
    if snapshot_chunks:
        train["snapshot_chunks"] = int(snapshot_chunks)
    if contrastive_weight:
        train["discriminative"] = {"contrastive_weight": float(contrastive_weight)}
    _apply_train_overrides(train, train_overrides)
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


def _draw(what, fn, *args, **kwargs):
    """Pictures never end a run."""
    from benchmarks.common import _log

    try:
        return fn(*args, **kwargs)
    except Exception as e:
        _log(f"Could not draw {what}: {e}")
        return None


def _evaluate(ckpt, tests, objects_dir_of, fil_names, particles_star, out_dir, views_dir=None, true_of=None):
    import time
    from pathlib import Path

    import mrcfile
    import pandas as pd
    import torch

    from benchmarks import filament_views as V
    from benchmarks.common import _counts, _device, _gpu_peak_mb, _infer_config, _reset_gpu_peak, isolated_main_config
    from tomocpt.labels.filaments import mask_to_centerlines, read_filament_coords
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
                    true = (read_filament_coords(true_of(t, row.name), apix) if true_of is not None
                            else mask_to_centerlines(mask, apix, row.diameter_ang, 0.5 * row.diameter_ang))
                    pred = [np.asarray(f["points"]) for f in fils.get(c, [])]
                    tol = 0.5 * row.diameter_ang / apix
                    s = centerline_scores(pred, true, tol)
                    s["points"] = _point_f1(pred, true, cfg.filament_spacing_ang / apix, tol)
                    s["n_pred"], s["n_true"] = len(pred), len(true)
                    rec["filaments"][row.name] = s
                    if views_dir is not None:
                        def draw_view():
                            with mrcfile.open(str(t["tomogram"]), permissive=True) as m:
                                tomo = np.asarray(m.data, dtype=np.float32)
                            title = (f"{t['name']}, {row.name}: threshold {thr[c]:.2f}, precision {s['precision']:.3f}, "
                                     f"recall {s['recall']:.3f}, F1 {s['f1']:.3f}, {len(pred)} traced / {len(true)} true")
                            V.save_filament_view(Path(views_dir) / f"{t['name']}_{row.name}.png", tomo,
                                                 V.target_map(true, mask.shape, row.diameter_ang, apix),
                                                 maps[c].detach().float().cpu().numpy(), pred, true, tol, title)
                        _draw(f"{t['name']} {row.name}", draw_view)
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
    from benchmarks.filament_views import dump_training_images, save_threshold_curve
    from tomocpt.predict.calibration import read_calibration

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
    ap.add_argument("--n_cpus", type=int, default=2, help="Data loading and preprocessing workers")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--train_runs", default=None, help="Fixed training runs of the one training set, comma-separated")
    ap.add_argument("--val_runs", default=None, help="Fixed validation runs (with --train_runs); reported as held_out")
    ap.add_argument("--centerlines", default=None,
                    help="<dir>/<object>/<run>.mod as the true centrelines, for labels and scores, instead of tracing "
                         "the masks (scripts/masks_to_centerlines.py; with --train_runs)")
    ap.add_argument("--band_max_vox", type=float, default=None,
                    help="Upper edge of the size band (default: the config default, 32). 37 puts a 250 A microtubule "
                         "and 70 A actin on one level at 6.8 A/px")
    ap.add_argument("--snapshot_chunks", type=int, default=0,
                    help="train.snapshot_chunks: chunks per class and split saved as volumes every epoch for "
                         "scripts/train_dashboard.py (2 is about 1.8 MB per epoch); 0 = off")
    ap.add_argument("--negative_masks", default=None,
                    help="Mask objects that are never a class, e.g. 'membrane' (<root>/<set>/masks/<name>/<run>.mrc; "
                         "prepData.negative_masks, negatives of the contrastive loss; with --train_runs)")
    ap.add_argument("--contrastive_weight", type=float, default=0.0,
                    help="train.discriminative.contrastive_weight (Gate 7; 0 = off; the CZII Gate 7 runs used 0.1)")
    ap.add_argument("--max_runs", type=int, default=None)
    ap.add_argument("--train_set", action="append", default=[],
                    help="key=value on the train section, e.g. sampling=on_the_fly (repeatable)")
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
    val_stems = None
    if a.centerlines and not a.train_runs:
        raise ValueError("--centerlines needs --train_runs")
    if a.negative_masks and not a.train_runs:
        raise ValueError("--negative_masks needs --train_runs")
    if bool(a.train_runs) != bool(a.val_runs):
        raise ValueError("--train_runs and --val_runs go together")
    if a.train_runs:
        if len(train_sets) != 1:
            raise ValueError("--train_runs needs exactly one training set")
        names = lambda text: [r.strip() for r in text.split(",") if r.strip()]  # noqa: E731
        staged = _stage_fixed(a.data_root, train_sets[0], objects, out / "stage", names(a.train_runs), names(a.val_runs))
        val_stems = [t["name"] for t in staged["test"]]
        if a.centerlines:
            _stage_centerlines(a.centerlines, train_sets[0], objects, out / "stage", names(a.train_runs) + names(a.val_runs))
        if a.negative_masks:
            _stage_negative_masks(a.data_root, train_sets[0], names(a.negative_masks), out / "stage",
                                  names(a.train_runs) + names(a.val_runs))
    else:
        staged = _stage(a.data_root, train_sets, objects, out / "stage", a.test_frac, a.seed, a.max_runs)
    if test_sets:
        extra = _stage(a.data_root, test_sets, objects, out / "stage_testsets", 1.0, a.seed, a.max_runs, all_test=True)
        staged["testsets"] = extra["test"]
    _log(f"train runs {len(staged['train'])}, held-out {len(staged['test'])}, test sets {len(staged.get('testsets', []))}")
    settings = TrainSettings(n_epochs=a.epochs, batch_size=a.batch_size, n_cpus=a.n_cpus, tiny=a.tiny,
                             swin={"feature_size": 12} if a.tiny else {}, class_head="softmax")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not a.particles_star:
        arms = [x for x in arms if x == "filaments"]
    spec = {"filaments": (fil, []), "joint": (fil, part), "particles": ([], part)}
    report = {"args": vars(a), "classes": {"filaments": fil, "particles": part}, "splits": staged, "arms": {}}
    for arm in arms:
        f_cls, p_cls = spec[arm]
        t0 = time.time()
        ckpt = _train_arm(out / arm, out / "stage" / "train", f_cls, p_cls, a.particles_star, settings, val_stems,
                          use_centerlines=bool(a.centerlines), band_max_vox=a.band_max_vox,
                          snapshot_chunks=a.snapshot_chunks,
                          negative_masks=[m.strip() for m in (a.negative_masks or "").split(",") if m.strip()],
                          contrastive_weight=a.contrastive_weight, train_overrides=a.train_set)
        res = {"checkpoint": str(ckpt), "train_seconds": time.time() - t0}
        views = out / arm / "views"
        _draw("training panels", dump_training_images, out / arm / "model", views / "training")
        calibration = read_calibration(ckpt) or {}
        res["calibration"] = calibration.get("classes")
        for n, name, _ in f_cls:
            entry = (calibration.get("classes") or {}).get(n)
            if entry:
                _draw(f"threshold curve of {name}", save_threshold_curve, views / f"threshold_curve_{name}.png", entry,
                      f"{name}, validation tomograms")

        def mask_of(t, name, root):
            return Path(root) / "masks" / name / f"{t['name']}.mrc"

        def centerlines_of(t, name):
            return out / "stage" / "train" / "centerlines" / name / f"{t['name']}.mod"

        held_out_root = out / "stage" / ("train" if val_stems else "test")
        for key, root in (("held_out", held_out_root), ("test_sets", out / "stage_testsets" / "test")):
            tests = staged["test"] if key == "held_out" else staged.get("testsets", [])
            if not tests:
                continue
            ev = _evaluate(ckpt, tests, lambda t, n, r=root: mask_of(t, n, r), [n for _, n, _ in f_cls],
                           a.particles_star, out / arm / f"pred_{key}", views_dir=views / key,
                           true_of=centerlines_of if a.centerlines and key == "held_out" else None)
            res[key] = {"summary": _summary(ev), **ev}
        report["arms"][arm] = res
        save_json(out / "filaments_benchmark.json", report)
        _log(f"arm {arm}: {json.dumps({k: v.get('summary') for k, v in res.items() if isinstance(v, dict)}, default=str)[:2000]}")
    save_json(out / "filaments_benchmark.json", report)
    return report


if __name__ == "__main__":
    main()
