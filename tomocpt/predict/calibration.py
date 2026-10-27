"""Per-class threshold calibration of pipeline-v2 checkpoints (Gate 4, T4.5).

One global ``confidence_threshold`` does not suit several classes, whose score
distributions differ. After v2 training, the thresholds are chosen on the validation
tomograms (maximum F-beta per class) and stored in the checkpoint under the top-level key
``"tomocpt_calibration"``; ``predict`` uses them unless ``infer.class_thresholds`` is set.
"""

import copy
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.logger import get_logger
from tomocpt.metrics import match_points, prf_beta

logger = get_logger()

CALIBRATION_KEY = "tomocpt_calibration"
CALIBRATION_VERSION = 1
#: Default threshold grid: 0.05, 0.10, ..., 0.95.
DEFAULT_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))
#: Match radius as a fraction of the class radius (as ``tomocpt.metrics.score_classes``).
MATCH_RADIUS_FRAC = 0.5


def calibrate_thresholds(
    picks: Dict[Any, List],
    truths: Dict[Any, List[Optional[np.ndarray]]],
    radii: Dict[Any, float],
    beta: float = 4.0,
    min_particles: int = 20,
    grid: Optional[Sequence[float]] = None,
) -> Dict[Any, Dict[str, Any]]:
    """Chooses the threshold of each class that maximises F-beta.

    ``picks[c]``: per tomogram ``(xyz [N, 3], score [N])``; ``truths[c]``: per tomogram
    ``xyz [M, 3]``, or ``None`` where class ``c`` is not available (that tomogram is skipped,
    so its picks count as neither true nor false positives); ``radii[c]``: class radius in
    the unit of the coordinates. Each threshold of ``grid`` is scored with
    ``tomocpt.metrics.match_points`` (radius ``0.5 * radii[c]``) and ``prf_beta`` summed over
    the tomograms; ties pick the higher threshold. A class with fewer than ``min_particles``
    true particles, or whose best F-beta is 0 (no threshold finds a true particle), gets
    ``"threshold": None`` (it falls back at prediction time).

    Returns ``{c: {"threshold", "fbeta", "n_true"}}``.
    """
    grid = sorted(float(t) for t in (DEFAULT_GRID if grid is None else grid))
    out = {}
    for c in picks:
        tomos = [
            (np.asarray(xyz, dtype=np.float64).reshape(-1, 3), np.asarray(score, dtype=np.float64).reshape(-1),
             np.asarray(t, dtype=np.float64).reshape(-1, 3))
            for (xyz, score), t in zip(picks[c], truths[c]) if t is not None
        ]
        n_true = int(sum(len(t) for _, _, t in tomos))
        radius = MATCH_RADIUS_FRAC * float(radii[c])
        best_thr, best_f = None, -1.0
        for thr in grid:
            tp = fp = fn = 0
            for xyz, score, t in tomos:
                pairs, fp_idx, fn_idx = match_points(xyz[score >= thr], t, radius)
                tp, fp, fn = tp + len(pairs), fp + len(fp_idx), fn + len(fn_idx)
            f = prf_beta(tp, fp, fn, beta=beta)["fbeta"]
            # The grid is ascending, so ">=" keeps the higher threshold on ties.
            if f >= best_f - 1e-12:
                best_thr, best_f = thr, f
        # No threshold finds a true particle (F-beta 0 everywhere): nothing to calibrate.
        calibrated = n_true >= int(min_particles) and best_f > 0.0
        out[c] = {
            "threshold": best_thr if calibrated else None,
            "fbeta": float(best_f),
            "n_true": n_true,
        }
    return out


def read_calibration(ckpt_path) -> Optional[dict]:
    """The ``"tomocpt_calibration"`` record of a checkpoint, or None."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return ckpt.get(CALIBRATION_KEY)


def _true_coords(coords_path, class_number: int) -> np.ndarray:
    df = pd.read_csv(coords_path)
    return df.loc[df["class_number"] == class_number, ["x", "y", "z"]].to_numpy(dtype=np.float64)


def calibrate_checkpoint(ckpt_path, val_tomograms: List[dict], infer_cfg: InferConfig) -> dict:
    """Calibrates the class thresholds of a pipeline-v2 checkpoint and stores them in it.

    ``val_tomograms``: the ``val_tomograms`` entries of ``chunk_layout.json``
    (``tomogram_path``, ``coords_path`` with columns ``x, y, z, class_number, diameter_ang``
    in native voxels, ``class_available`` per class). Each tomogram runs through the v2
    predict path (no Dask) with every class at the lowest grid threshold, after cross-class
    suppression if ``infer_cfg.cross_class_nms``. The record ``{"version", "beta", "grid",
    "classes", "val_tomograms"}`` is written under ``"tomocpt_calibration"`` (temporary file,
    then rename); the other checkpoint keys are kept. Returns the record.
    """
    from tomocpt.networks.inference import InferenceModel
    from tomocpt.predict.multiclass import peaks_from_class_maps, predict_class_maps

    warnings.filterwarnings("ignore", message=".*Using TorchIO images without a torchio.SubjectsLoader.*",
                            category=UserWarning)
    if infer_cfg.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = InferenceModel.load_from_checkpoint(str(ckpt_path), map_location=device, strict=True).eval()
    if not model.is_pipeline_v2():
        raise ValueError(f"{ckpt_path} is not a pipeline-v2 checkpoint; nothing to calibrate.")
    table = model.class_table()
    grid = list(DEFAULT_GRID)
    lowest = {c: min(grid) for c in range(len(table))}
    all_classes = list(range(len(table)))

    picks = {row.number: [] for row in table}
    truths = {row.number: [] for row in table}
    radii: Dict[int, float] = {}
    for entry in val_tomograms:
        tomo = Path(entry["tomogram_path"])
        tomo_cfg = infer_cfg
        if entry.get("apix_native"):
            # The voxel size the chunking used (header or prepData.apix_override).
            tomo_cfg = copy.copy(infer_cfg)
            tomo_cfg.apix_override = float(entry["apix_native"])
        maps, apix = predict_class_maps(model, tomo, tomo_cfg, device, classes=all_classes)
        peaks, _ = peaks_from_class_maps(maps, table, apix, infer_cfg, lowest, tomo.stem, device)
        del maps
        available = entry.get("class_available") or [True] * len(table)
        for c, row in enumerate(table):
            # Radii in native voxels, as the coordinates (the same for every tomogram of
            # one voxel size; the last one wins otherwise).
            radii[row.number] = float(row.diameter_ang) / (2.0 * apix)
            p = peaks.get(c, np.zeros((0, 4)))
            picks[row.number].append((p[:, [2, 1, 0]], p[:, 3]))
            truths[row.number].append(_true_coords(entry["coords_path"], row.number) if available[c] else None)

    classes = calibrate_thresholds(picks, truths, radii, beta=infer_cfg.calibration_beta,
                                   min_particles=infer_cfg.calibration_min_particles, grid=grid)
    record = {
        "version": CALIBRATION_VERSION,
        "beta": float(infer_cfg.calibration_beta),
        "grid": grid,
        "classes": classes,
        "val_tomograms": [str(e["tomogram_path"]) for e in val_tomograms],
    }
    for row in table:
        e = classes[row.number]
        thr = "None (falls back)" if e["threshold"] is None else f"{e['threshold']:.2f}"
        logger.info(f"Calibration, class {row.name}: threshold {thr}, F-beta {e['fbeta']:.3f}, "
                    f"{e['n_true']} true particle(s)")

    del model
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt[CALIBRATION_KEY] = record
    tmp = Path(str(ckpt_path) + ".calibration.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, ckpt_path)
    return record
