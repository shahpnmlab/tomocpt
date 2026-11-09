"""Wall time of the non-training stages (spec 8a): chunking, predict and post-fit calibration, with
inclusive per-function timers (synchronised; nested functions are counted in their callers too).

    python -m benchmarks.perf.pipeline predict   --ckpt weights.ckpt --chunks_dir <dir> [--n 1] --out x.json
    python -m benchmarks.perf.pipeline calibrate --ckpt weights.ckpt --chunks_dir <dir> --out x.json
    python -m benchmarks.perf.pipeline chunk     --config recipe.yaml --tmp <dir> --out x.json

``predict`` / ``calibrate`` use the validation tomograms of ``chunk_layout.json`` and the checkpoint's
own infer settings. ``calibrate`` works on a temporary copy of the checkpoint (it writes into it).
"""
import argparse
import contextlib
import cProfile
import importlib
import io
import pstats
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch

from benchmarks.perf.common import applied_config, environment, git_commit, load_config, write_json

#: (module, attribute) pairs timed inclusively.
TIMED = [
    ("tomocpt.predict.multiclass", "load_mrc"),
    ("tomocpt.predict.multiclass", "resample_volume"),
    ("tomocpt.predict.multiclass", "symmetrize_and_pad"),
    ("tomocpt.predict.multiclass", "_predict_level"),
    ("tomocpt.predict.multiclass", "predict_class_maps"),
    ("tomocpt.predict.multiclass", "peaks_from_class_maps"),
    ("tomocpt.predict.multiclass", "trace_class_maps"),
    ("tomocpt.predict.multiclass", "save_class_maps"),
    ("tomocpt.predict.helpers", "_extract_peaks_from_mask"),
    ("tomocpt.predict.helpers", "peak_local_max_torch"),
    ("tomocpt.predict.helpers", "_refine_peaks_subpixel_gpu"),
    ("tomocpt.predict.helpers", "cross_class_suppression"),
    ("tomocpt.predict.filaments", "trace_class_maps"),
    ("tomocpt.labels.filaments", "_trace_and_fit"),
    ("tomocpt.labels.filaments", "_region_polylines"),
    ("tomocpt.predict.calibration", "filament_threshold_scores"),
    ("tomocpt.predict.calibration", "peak_profile_accumulate"),
    ("tomocpt.predict.calibration", "calibrate_thresholds"),
    ("tomocpt.dataManager.chunking", "process_single_tomo_task_v2"),
]


@contextlib.contextmanager
def timed_functions(pairs=TIMED):
    """Replace each function by a timing wrapper; yields ``{name: {"s", "calls"}}``."""
    acc = defaultdict(lambda: {"s": 0.0, "calls": 0})
    saved = []
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for mod_name, attr in pairs:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, attr, None)
        if fn is None or getattr(fn, "_g8_wrapped", False):
            continue
        name = f"{mod_name.rsplit('.', 1)[-1]}.{attr}"

        def wrapper(*a, _fn=fn, _name=name, **k):
            sync()
            t0 = time.perf_counter()
            try:
                return _fn(*a, **k)
            finally:
                sync()
                acc[_name]["s"] += time.perf_counter() - t0
                acc[_name]["calls"] += 1

        wrapper._g8_wrapped = True
        saved.append((mod, attr, fn))
        setattr(mod, attr, wrapper)
    try:
        yield acc
    finally:
        for mod, attr, fn in saved:
            setattr(mod, attr, fn)


def _cprofile_top(prof, n=30) -> str:
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(n)
    return s.getvalue()


def _load_model(ckpt, device):
    from tomocpt.networks.inference import InferenceModel
    from tomocpt.predict.ensemble import isolated_checkpoint_load

    with isolated_checkpoint_load():
        return InferenceModel.load_from_checkpoint(str(ckpt), map_location=device, strict=True).eval()


def _val_tomograms(chunks_dir):
    from tomocpt.dataManager.metadata import read_chunk_layout

    vt = (read_chunk_layout(chunks_dir) or {}).get("val_tomograms") or []
    if not vt:
        raise ValueError(f"{chunks_dir}/chunk_layout.json lists no val_tomograms")
    return vt


def _infer_cfg(model, overrides=()):
    import copy

    from benchmarks.perf.common import apply_overrides

    cfg = copy.deepcopy(model.hparams.config)
    infer = cfg.infer
    infer.use_cuda = torch.cuda.is_available()
    infer.save_prediction_confidence_map = False
    infer.save_predicted_coords = True
    infer.invert_contrast = False
    infer.masks_dir = None
    apply_overrides(cfg, overrides)
    return infer


def time_predict(ckpt, chunks_dir, n=1, overrides=(), profile=False) -> dict:
    from tomocpt.predict.multiclass import infer_tomos_v2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt, device)
    infer = _infer_cfg(model, overrides)
    tomos = [Path(e["tomogram_path"]) for e in _val_tomograms(chunks_dir)][: int(n)]
    infer_tomos_v2(model, tomos[:1], 0, 0, ckpt, infer, device)  # warm-up (cudnn autotune, allocator)
    prof = cProfile.Profile() if profile else None
    with timed_functions() as acc:
        if prof:
            prof.enable()
        t0 = time.perf_counter()
        res = infer_tomos_v2(model, tomos, 0, 0, ckpt, infer, device)
        wall = time.perf_counter() - t0
        if prof:
            prof.disable()
    out = {"stage": "predict", "wall_s": wall, "per_tomogram_s": wall / max(1, len(tomos)),
           "tomograms": [str(t) for t in tomos], "functions": dict(acc),
           "n_picks": {r["tomo"] if "tomo" in r else str(i): {str(c): len(v) for c, v in r.get("class_coords", {}).items()}
                       for i, r in enumerate(res)},
           "settings": {"batch": int(infer.predictions_batch_size), "tta": bool(infer.use_tta)}}
    if prof:
        out["cprofile_top"] = _cprofile_top(prof)
    return out


def time_calibration(ckpt, chunks_dir, overrides=(), profile=False) -> dict:
    from tomocpt.predict.calibration import calibrate_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt, device)
    infer = _infer_cfg(model, overrides)
    infer.classes = None
    del model
    vt = _val_tomograms(chunks_dir)
    with tempfile.TemporaryDirectory(prefix="tomocpt_perf_cal_") as tmp:
        copy_path = Path(tmp) / Path(ckpt).name
        shutil.copy2(ckpt, copy_path)
        prof = cProfile.Profile() if profile else None
        with timed_functions() as acc:
            if prof:
                prof.enable()
            t0 = time.perf_counter()
            calibrate_checkpoint(copy_path, vt, infer)
            wall = time.perf_counter() - t0
            if prof:
                prof.disable()
    out = {"stage": "calibration", "wall_s": wall, "per_tomogram_s": wall / len(vt),
           "tomograms": [str(e["tomogram_path"]) for e in vt], "functions": dict(acc)}
    if prof:
        out["cprofile_top"] = _cprofile_top(prof)
    return out


def time_chunking(config, tmp_dir, overrides=()) -> dict:
    cfg = load_config(config, overrides)
    out_dir = Path(tmp_dir) / "chunks"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cfg.train.chunks_dir = out_dir
    cfg.shared.chunks_dir = out_dir
    with applied_config(cfg) as mc:
        from tomocpt.training.helpers import _prepare_data_if_needed

        with timed_functions() as acc:
            t0 = time.perf_counter()
            _prepare_data_if_needed(mc.train)
            wall = time.perf_counter() - t0
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    return {"stage": "chunking", "wall_s": wall, "n_files": len(files),
            "n_volume_chunks": sum(1 for p in files if p.name.endswith(".nii.gz") and "volumes" in str(p.parent)),
            "bytes": int(sum(p.stat().st_size for p in files)), "functions": dict(acc)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("predict", "calibrate", "chunk"))
    p.add_argument("--ckpt")
    p.add_argument("--chunks_dir")
    p.add_argument("--config")
    p.add_argument("--tmp", default=None)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--profile", action="store_true")
    p.add_argument("--keep", action="store_true", help="chunk: keep the chunks written under --tmp")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.stage == "predict":
        r = time_predict(a.ckpt, a.chunks_dir, a.n, a.overrides, a.profile)
    elif a.stage == "calibrate":
        r = time_calibration(a.ckpt, a.chunks_dir, a.overrides, a.profile)
    else:
        tmp = a.tmp or tempfile.mkdtemp(prefix="tomocpt_perf_chunk_")
        try:
            r = time_chunking(a.config, tmp, a.overrides)
        finally:
            if not a.keep:
                shutil.rmtree(Path(tmp) / "chunks", ignore_errors=True)
    r.update({"commit": git_commit(), "env": environment()})
    write_json(a.out, r)
    top = sorted(r["functions"].items(), key=lambda kv: -kv[1]["s"])[:12]
    print(f"{r['stage']}: wall {r['wall_s']:.1f} s")
    for k, v in top:
        print(f"  {k}: {v['s']:.2f} s ({v['calls']} calls)")


if __name__ == "__main__":
    main()
