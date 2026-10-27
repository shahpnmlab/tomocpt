"""Shared machinery of the Gate 4 benchmarks (T4.6).

A benchmark *run* trains the four arms on one training set and scores them on held-out
test tomograms:

* (a) one pipeline-v1 model per species (today's practice);
* (b) one pipeline-v2 multi-class model, scale-conditioned;
* (c) (b) without conditioning;
* (d) (b) warm-started (``fine_tune_from``) from the public TomoCPT model.

Training goes through the real CLI (``python -m tomocpt.main prepare_data|train
--config_file``, one subprocess per command, so every arm starts from a clean global
config). Pipeline-v2 training ends with the automatic threshold calibration (T4.5).
Prediction runs in this process (``tomocpt.predict``), so wall time and peak GPU memory are
measured around the prediction itself (model load included).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
ARMS = ("a", "b", "c", "d")
GLOBAL_THRESHOLD = 0.3
BETA = 4.0


@dataclass
class ClassInfo:
    number: int
    name: str
    diameter_ang: float


@dataclass
class TrainSet:
    """Tomograms and one RELION STAR file (``rlnClassNumber`` per particle)."""
    tomo_dir: Path
    star_path: Path


@dataclass
class TestTomogram:
    path: Path
    apix: float
    #: ``{class_number: (N, 3) xyz}`` in native voxels.
    coords: Dict[int, np.ndarray]


@dataclass
class TrainSettings:
    n_epochs: int
    batch_size: int = 2
    tiny: bool = False
    #: Label of the run (TomoCPT training is not seeded; repeated runs differ).
    seed: int = 0
    n_cpus: int = 2
    calibration_min_particles: int = 5
    #: Overrides of ``train.network.SwinUNETR`` (tiny runs use ``feature_size=12``).
    swin: Dict[str, object] = field(default_factory=dict)


def _log(msg: str) -> None:
    print(f"[benchmark {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_cli(command: str, config_path: Path, log_path: Path, extra: Sequence[str] = ()) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "tomocpt.main", command, "--config_file", str(config_path), *extra]
    with open(log_path, "w") as log:
        res = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        tail = Path(log_path).read_text()[-4000:]
        raise RuntimeError(f"`{' '.join(cmd)}` failed (exit {res.returncode}); log {log_path}:\n{tail}")


def _find_checkpoint(model_dir: Path) -> Path:
    ckpts = sorted(Path(model_dir).rglob("checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise RuntimeError(f"No checkpoint written under {model_dir}")
    return ckpts[-1]


def _train_section(work: Path, settings: TrainSettings, pipeline_version: int,
                   conditioning: bool = False, fine_tune_from: Optional[Path] = None) -> dict:
    use_gpu = torch.cuda.is_available()
    train = {
        "training_data_dir": str(work / "training_data"),
        "chunks_dir": str(work / "chunks"),
        "model_dir": str(work / "model"),
        "experiment_name": "bench",
        "n_epochs": int(settings.n_epochs),
        "batch_size": int(settings.batch_size),
        "use_gpus": use_gpu,
        "USE_CUDA_FOR_DATA": use_gpu,
        "n_cpus_for_dataloading": int(settings.n_cpus),
        "n_cpus_for_preprocessing": int(settings.n_cpus),
        "data_pipeline_version": int(pipeline_version),
        "network": {"SwinUNETR": dict(settings.swin)},
    }
    if not use_gpu:
        train["network"]["TORCH_FLOAT_PRECISION"] = "32"
    if pipeline_version == 2:
        train["network"]["scale_conditioning"] = bool(conditioning)
        train["augmentation"] = {"label_interpolation": "linear"}
    if fine_tune_from is not None:
        train["fine_tune_from"] = str(fine_tune_from)
    return train


def _write_config(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def train_v1_species(work: Path, data: TrainSet, cls: ClassInfo, settings: TrainSettings) -> Path:
    """Arm (a), one species: legacy ``prepare_data`` (one dataset, ``class_id`` = the class
    number, ``particle_length_ang`` = its diameter) and ``train`` (pipeline v1)."""
    work.mkdir(parents=True, exist_ok=True)
    cfg = {
        "shared": {"training_data_dir": str(work / "training_data"), "chunks_dir": str(work / "chunks")},
        "prepData": {
            "raw_data_dir": str(data.tomo_dir),
            "training_data_dir": str(work / "training_data"),
            "coordinate_files": str(data.star_path),
            "coordinate_file_type": "star",
            "particle_length_ang": repr(float(cls.diameter_ang)),
            "class_id": str(cls.number),
        },
        "train": _train_section(work, settings, 1),
        "infer": {"predictions_dir": str(work / "predictions")},
    }
    config = _write_config(work / "config.yaml", cfg)
    _run_cli("prepare_data", config, work / "prepare_data.log")
    _run_cli("train", config, work / "train.log")
    return _find_checkpoint(work / "model")


def train_v2(work: Path, data_dir: Path, data: TrainSet, classes: Sequence[ClassInfo], settings: TrainSettings,
             conditioning: bool, fine_tune_from: Optional[Path] = None) -> Path:
    """Arms (b)-(d): multi-class ``prepare_data`` (into ``data_dir``, shared by the arms;
    run once) and pipeline-v2 ``train`` into ``work``. The training run calibrates the class
    thresholds of its best checkpoint."""
    work.mkdir(parents=True, exist_ok=True)
    cfg = {
        "shared": {"training_data_dir": str(data_dir / "training_data"), "chunks_dir": str(data_dir / "chunks")},
        "prepData": {
            "raw_data_dir": str(data.tomo_dir),
            "training_data_dir": str(data_dir / "training_data"),
            "coordinate_files": str(data.star_path),
            "coordinate_file_type": "star",
            "particle_length_ang": repr(max(float(c.diameter_ang) for c in classes)),
            "class_id": "all",
            "multiclass": True,
            "class_diameters": ",".join(f"{c.number}:{float(c.diameter_ang):g}" for c in classes),
            "class_names": ",".join(f"{c.number}:{c.name}" for c in classes),
            "label_border_policy": "clip",
        },
        "train": _train_section(work, settings, 2, conditioning, fine_tune_from),
        "infer": {"predictions_dir": str(work / "predictions"),
                  "calibration_min_particles": int(settings.calibration_min_particles)},
    }
    cfg["train"]["training_data_dir"] = str(data_dir / "training_data")
    cfg["train"]["chunks_dir"] = str(data_dir / "chunks")
    config = _write_config(work / "config.yaml", cfg)
    if not (data_dir / "training_data").is_dir():
        _run_cli("prepare_data", config, work / "prepare_data.log")
    _run_cli("train", config, work / "train.log")
    return _find_checkpoint(work / "model")


# --------------------------------------------------------------------------------------
# prediction and scoring
# --------------------------------------------------------------------------------------

@contextlib.contextmanager
def isolated_main_config():
    """Restores the global ``mainConfig`` after the block. ``BaseModel.__init__`` writes a
    loaded checkpoint's train settings into ``mainConfig.train``, and a later load in the
    same process takes them as CLI changes (legacy behaviour): without this, loading the
    unconditioned arm (c) after the conditioned arm (b) builds a conditioned network."""
    from tomocpt.mainConfig import mainConfig

    saved = copy.deepcopy(mainConfig)
    try:
        yield
    finally:
        _assign_inplace(mainConfig, saved)


def _assign_inplace(target, source) -> None:
    for f in dataclasses.fields(type(source)):
        sv = getattr(source, f.name)
        tv = getattr(target, f.name, None)
        if dataclasses.is_dataclass(sv) and dataclasses.is_dataclass(tv) and type(sv) is type(tv):
            _assign_inplace(tv, sv)
        else:
            setattr(target, f.name, copy.deepcopy(sv))


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _infer_config(predictions_dir: Path, **kw):
    from tomocpt.defaultConfigs.infer_config import InferConfig

    cfg = InferConfig()
    cfg.predictions_dir = Path(predictions_dir)
    cfg.use_cuda = torch.cuda.is_available()
    cfg.save_prediction_confidence_map = False
    cfg.save_predicted_coords = True
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated()) / 2 ** 20


def _counts(pred_xyz: np.ndarray, true_xyz: np.ndarray, radius: float):
    from tomocpt.metrics import match_points

    pred_xyz = np.asarray(pred_xyz, dtype=np.float64).reshape(-1, 3)
    true_xyz = np.asarray(true_xyz, dtype=np.float64).reshape(-1, 3)
    pairs, fp, fn = match_points(pred_xyz, true_xyz, radius)
    return len(pairs), len(fp), len(fn)


def _score(picks: Dict[int, List[np.ndarray]], tests: Sequence[TestTomogram], classes: Sequence[ClassInfo]) -> Dict:
    """Per-class precision/recall/F-beta over all test tomograms (counts summed).
    ``picks[number][i]``: ``[N, 4]`` rows ``z, y, x, score`` of test tomogram ``i``."""
    from tomocpt.metrics import prf_beta

    out = {}
    for c in classes:
        tp = fp = fn = 0
        for i, t in enumerate(tests):
            p = np.asarray(picks.get(c.number, [np.zeros((0, 4))] * len(tests))[i]).reshape(-1, 4)
            radius = 0.5 * float(c.diameter_ang) / (2.0 * t.apix)  # match radius = half the radius
            a, b, d = _counts(p[:, [2, 1, 0]], t.coords.get(c.number, np.zeros((0, 3))), radius)
            tp, fp, fn = tp + a, fp + b, fn + d
        s = prf_beta(tp, fp, fn, beta=BETA)
        out[c.name] = {"fbeta": s["fbeta"], "precision": s["precision"], "recall": s["recall"],
                       "tp": tp, "fp": fp, "fn": fn}
    return out


def evaluate_v1_arm(ckpts: Dict[int, Path], tests: Sequence[TestTomogram], classes: Sequence[ClassInfo],
                    pred_dir: Path) -> Dict:
    """Arm (a): every species model on every test tomogram (``infer_tomos``, no Dask), at
    the global threshold. The time per tomogram is summed over the species models."""
    from tomocpt.predict.helpers import infer_tomos

    device_gpu = 0 if torch.cuda.is_available() else None
    picks: Dict[int, List[np.ndarray]] = {}
    total_time, peak = 0.0, None
    for c in classes:
        cfg = _infer_config(pred_dir / f"class_{c.number}", length=float(c.diameter_ang),
                            confidence_threshold=GLOBAL_THRESHOLD)
        cfg.predictions_dir.mkdir(parents=True, exist_ok=True)
        _reset_gpu_peak()
        t0 = time.perf_counter()
        results = infer_tomos([t.path for t in tests], device_gpu, 0, str(ckpts[c.number]), cfg)
        total_time += time.perf_counter() - t0
        g = _gpu_peak_mb()
        peak = g if peak is None else max(peak, g or 0.0)
        by_name = {r["name"]: np.asarray(r["coords"], dtype=np.float64).reshape(-1, 4) for r in results}
        picks[c.number] = [by_name.get(t.path.stem, np.zeros((0, 4))) for t in tests]
    per_class = _score(picks, tests, classes)
    for c in classes:
        per_class[c.name].update(threshold=GLOBAL_THRESHOLD, threshold_source="confidence_threshold")
    return {
        "wall_time_per_tomogram_s": total_time / max(1, len(tests)),
        "peak_gpu_memory_mb": peak,
        "cross_class_suppression": {},
        "scoring": {"global_0.3": {"per_class": per_class}},
    }


def evaluate_v2_arm(ckpt: Path, tests: Sequence[TestTomogram], classes: Sequence[ClassInfo], pred_dir: Path) -> Dict:
    """Arms (b)-(d): per-class maps once per test tomogram, then peaks (with cross-class
    suppression) at the global threshold and at the calibrated thresholds."""
    from tomocpt.networks.inference import InferenceModel
    from tomocpt.predict.calibration import read_calibration
    from tomocpt.predict.multiclass import peaks_from_class_maps, predict_class_maps, resolve_thresholds

    device = _device()
    cfg = _infer_config(pred_dir, confidence_threshold=GLOBAL_THRESHOLD, cross_class_nms=True)
    calibration = read_calibration(ckpt)
    _reset_gpu_peak()
    t0 = time.perf_counter()
    model = InferenceModel.load_from_checkpoint(str(ckpt), map_location=device, strict=True).eval()
    load_time = time.perf_counter() - t0
    table = model.class_table()
    modes = {
        "global_0.3": {c: (GLOBAL_THRESHOLD, "confidence_threshold") for c in range(len(table))},
        "calibrated": resolve_thresholds(table, cfg, calibration),
    }
    picks = {m: {row.number: [] for row in table} for m in modes}
    suppression = {m: {} for m in modes}
    map_time = 0.0
    for t in tests:
        t1 = time.perf_counter()
        maps, apix = predict_class_maps(model, t.path, cfg, device, classes=list(range(len(table))))
        map_time += time.perf_counter() - t1
        for mode, thr in modes.items():
            t2 = time.perf_counter()
            peaks, counts = peaks_from_class_maps(maps, table, apix, cfg, {c: v[0] for c, v in thr.items()},
                                                  t.path.stem, device)
            if mode == "global_0.3":
                map_time += time.perf_counter() - t2
            for c, p in peaks.items():
                picks[mode][table[c].number].append(p)
            for (kept, removed), n in counts.items():
                key = f"{table[kept].name}->{table[removed].name}"
                suppression[mode][key] = suppression[mode].get(key, 0) + int(n)
    peak = _gpu_peak_mb()
    del model
    scoring = {}
    for mode, thr in modes.items():
        per_class = _score(picks[mode], tests, classes)
        for c, row in enumerate(table):
            per_class[row.name].update(threshold=float(thr[c][0]), threshold_source=thr[c][1])
        scoring[mode] = {"per_class": per_class}
    # Wall time: model load + maps + peaks at one threshold set (the second set reuses the maps).
    return {
        "wall_time_per_tomogram_s": (load_time + map_time) / max(1, len(tests)),
        "peak_gpu_memory_mb": peak,
        "cross_class_suppression": suppression["global_0.3"],
        "cross_class_suppression_calibrated": suppression["calibrated"],
        "calibration": calibration,
        "scoring": scoring,
    }


# --------------------------------------------------------------------------------------
# one run: all arms
# --------------------------------------------------------------------------------------

def public_checkpoint(pretrained: Optional[str]) -> Optional[Path]:
    """The public TomoCPT checkpoint: ``--pretrained`` or ``$TOMOCPT_ZENODO_CKPT``."""
    for p in (pretrained, os.environ.get("TOMOCPT_ZENODO_CKPT")):
        if p and Path(p).is_file():
            return Path(p)
    return None


def run_arms(run_dir: Path, data: TrainSet, tests: Sequence[TestTomogram], classes: Sequence[ClassInfo],
             settings: TrainSettings, pretrained: Optional[Path], arms: Sequence[str] = ARMS) -> Dict:
    """Trains and scores the requested arms on one training set. Returns
    ``{"arms": {arm: result}, "warm_start": {...}}``.

    Arm (d) warm-starts from ``pretrained`` (the public model). Without it, or in a tiny run
    (whose network is smaller than the public model's), it warm-starts from arm (a)'s model
    of the first class, a pipeline-v1 checkpoint trained in this run, and says so in
    ``warm_start``.
    """
    run_dir = Path(run_dir)
    out: Dict[str, Dict] = {}
    v1_ckpts: Dict[int, Path] = {}
    if "a" in arms or "d" in arms:
        for c in classes:
            _log(f"arm a: training the v1 model of {c.name}")
            v1_ckpts[c.number] = train_v1_species(run_dir / "a" / f"class_{c.number}", data, c, settings)
    if "a" in arms:
        _log("arm a: predicting")
        with isolated_main_config():
            out["a"] = evaluate_v1_arm(v1_ckpts, tests, classes, run_dir / "a" / "predictions")

    data_dir = run_dir / "v2_data"
    if pretrained is not None and not settings.tiny:
        warm = {"source": str(pretrained), "fallback": False}
    else:
        reason = "tiny run (network smaller than the public model)" if settings.tiny else "public model not available"
        warm = {"source": str(v1_ckpts.get(classes[0].number, "")), "fallback": True, "reason": reason}
    for arm, cond, ft in (("b", True, None), ("c", False, None), ("d", True, "warm")):
        if arm not in arms:
            continue
        fine_tune = Path(warm["source"]) if ft else None
        _log(f"arm {arm}: training the v2 model (conditioning={cond}, warm start={fine_tune})")
        ckpt = train_v2(run_dir / arm, data_dir, data, classes, settings, cond, fine_tune)
        _log(f"arm {arm}: predicting")
        with isolated_main_config():
            out[arm] = evaluate_v2_arm(ckpt, tests, classes, run_dir / arm / "predictions")
    return {"arms": out, "warm_start": warm}


def save_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
