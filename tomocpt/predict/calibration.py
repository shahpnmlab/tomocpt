"""Per-class threshold calibration of pipeline-v2 checkpoints (Gate 4, T4.5).

One global ``confidence_threshold`` does not suit several classes, whose score
distributions differ. After v2 training, the thresholds are chosen on the validation
tomograms (maximum F-beta per class) and stored in the checkpoint under the top-level key
``"tomocpt_calibration"``; ``predict`` uses them unless ``infer.class_thresholds`` is set.
"""

import contextlib
import copy
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.labels.multiclass import PARTICLE
from tomocpt.logger import get_logger
from tomocpt.metrics import match_points, prf_beta

logger = get_logger()

CALIBRATION_KEY = "tomocpt_calibration"
CALIBRATION_VERSION = 1
#: Default threshold grid: 0.05, 0.10, ..., 0.95.
DEFAULT_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))
#: Match radius as a fraction of the class radius (as ``tomocpt.metrics.score_classes``).
MATCH_RADIUS_FRAC = 0.5
#: Tracing thresholds of filament classes (Gate 6b): tracing is expensive, so a coarser grid.
FILAMENT_GRID = tuple(round(0.1 * k, 1) for k in range(1, 10))
#: Centreline tolerance as a fraction of the class diameter (as ``benchmarks/filaments.py``).
FILAMENT_TOLERANCE_FRAC = 0.5
#: A threshold above this fraction of the volume is not traced: skeletonising a near-full volume
#: takes minutes and no such trace can be right.
FILAMENT_MAX_REGION_FRAC = 0.25
#: Columns of a score row: covered predicted, predicted, covered true, true length (A), true filaments.
N_FILAMENT_COLUMNS = 5


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


#: State shared with forked tracing workers (set just before the pool starts; Gate 8e.1).
_TRACE_STATE: Dict[str, Any] = {}


def _filament_threshold_row(thr: float) -> np.ndarray:
    """Scores of one threshold (``_TRACE_STATE`` holds the map and settings)."""
    from tomocpt.labels import filaments as LF
    from tomocpt.metrics import _covered_length

    st = _TRACE_STATE
    prob, apix, tol, step, true = st["prob"], st["apix"], st["tol"], st["step"], st["true"]
    pred = []
    if float(np.mean(prob > float(thr))) <= FILAMENT_MAX_REGION_FRAC:
        fits = LF._trace_and_fit(prob, float(thr), apix, st["diameter"], st["min_length"], st["spacing"],
                                 st["smoothing"])
        pred = [np.asarray(pts, dtype=np.float64) for pts, _, _ in fits]
    near_p, len_p = _covered_length(pred, true, tol, step)
    near_t = _covered_length(true, pred, tol, step)[0] if pred else st["near_t_empty"]
    return np.array((near_p * apix, len_p * apix, near_t * apix, st["len_t"] * apix, len(true)), dtype=np.float64)


#: Maps smaller than this (voxels) are traced in this process: forking costs more than it saves.
PARALLEL_TRACE_MIN_VOXELS = 4_000_000


def _trace_workers(n_thresholds: int, workers: Optional[int], n_voxels: int) -> int:
    if workers is not None:
        return max(1, int(workers))
    if not hasattr(os, "fork") or n_voxels < PARALLEL_TRACE_MIN_VOXELS:
        return 1
    return max(1, min(int(n_thresholds), (os.cpu_count() or 2) // 2, 16))


def filament_threshold_scores(prob, true_polylines, diameter_ang: float, apix: float, infer_cfg,
                              grid: Sequence[float], workers: Optional[int] = None) -> np.ndarray:
    """Scores of one filament channel of one tomogram at each threshold of ``grid``: ``[G, 5]``
    (see ``N_FILAMENT_COLUMNS``; lengths in A, so tomograms of different voxel sizes pool).

    ``prob``: ``[D, H, W]`` probability map in native voxels; ``true_polylines``: ``x, y, z``
    native voxels. Each threshold is traced with the infer settings, so the score is that of what
    ``predict`` would write. ``workers`` (Gate 8e.1): thresholds traced in parallel forked
    processes (``None`` = automatic, ``1`` = in this process); the scores are identical."""
    from tomocpt.metrics import _covered_length

    prob = np.asarray(prob, dtype=np.float32)
    apix = float(apix)
    tol = FILAMENT_TOLERANCE_FRAC * float(diameter_ang) / apix
    true = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in true_polylines if len(p)]
    step = 0.25
    near_t_empty, len_t = _covered_length(true, [], tol, step)
    _TRACE_STATE.clear()
    _TRACE_STATE.update(prob=prob, apix=apix, tol=tol, step=step, true=true, near_t_empty=near_t_empty,
                        len_t=len_t, diameter=float(diameter_ang),
                        min_length=float(infer_cfg.filament_min_length_ang),
                        spacing=float(infer_cfg.filament_spacing_ang),
                        smoothing=float(infer_cfg.filament_smoothing_ang))
    thresholds = [float(t) for t in grid]
    n_workers = _trace_workers(len(thresholds), workers, prob.size)
    try:
        if n_workers <= 1 or len(thresholds) <= 1:
            rows = [_filament_threshold_row(t) for t in thresholds]
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # fork: the workers share the map copy-on-write; they touch no CUDA state.
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork")) as pool:
                rows = list(pool.map(_filament_threshold_row, thresholds))
    finally:
        _TRACE_STATE.clear()
    out = np.zeros((len(thresholds), N_FILAMENT_COLUMNS), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i] = r
    return out


def calibrate_filament_thresholds(scores: Dict[Any, List[Optional[np.ndarray]]], grid: Sequence[float],
                                  beta: float = 1.0) -> Dict[Any, Dict[str, Any]]:
    """Chooses the tracing threshold of each filament class that maximises the length-weighted
    centreline F-beta, pooled over the tomograms.

    ``scores[c]``: per tomogram the ``[G, 5]`` of :func:`filament_threshold_scores`, or ``None``
    where unavailable (skipped); ``grid``: ascending. Ties pick the higher threshold. A class
    without true length, or whose best F-beta is 0, gets ``"threshold": None``. Returns ``{c:
    {"threshold", "fbeta", "n_true", "kind", "precision", "recall", "f1", "true_length_ang",
    "curve"}}``; ``curve`` = ``[[threshold, precision, recall, fbeta], ...]``, ``n_true`` counts
    true filaments."""
    from tomocpt.predict.filaments import FILAMENT

    grid = [float(t) for t in grid]
    out = {}
    for c, per_tomo in scores.items():
        rows = [np.asarray(s, dtype=np.float64) for s in per_tomo if s is not None]
        best = {"threshold": None, "fbeta": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        curve, n_true, true_len = [], 0, 0.0
        if rows:
            total = np.sum(rows, axis=0)
            n_true, true_len = int(total[0, 4]), float(total[0, 3])
            for i, thr in enumerate(grid):
                s = _length_scores(total[i], beta)
                curve.append([thr, s["precision"], s["recall"], s["fbeta"]])
                # The grid is ascending, so ">=" keeps the higher threshold on ties.
                if s["fbeta"] > 0.0 and s["fbeta"] >= best["fbeta"] - 1e-12:
                    best = {"threshold": thr, **s}
        out[c] = {**best, "n_true": n_true, "kind": FILAMENT, "true_length_ang": true_len, "curve": curve}
    return out


def _length_scores(row: np.ndarray, beta: float) -> Dict[str, float]:
    """Precision, recall, F1 and F-beta of one (pooled) score row."""
    near_p, len_p, near_t, len_t = (float(v) for v in row[:4])
    p = near_p / len_p if len_p > 0 else 0.0
    r = near_t / len_t if len_t > 0 else 0.0
    b2 = float(beta) ** 2
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r > 0 else 0.0,
            "fbeta": (1 + b2) * p * r / (b2 * p + r) if p + r > 0 else 0.0}


def filament_settings(infer_cfg) -> str:
    """JSON of the tracing settings the filament scores depend on. Not in :func:`picks_fingerprint`,
    so that particle picks saved earlier stay reusable."""
    return json.dumps({k: float(getattr(infer_cfg, k)) for k in
                       ("filament_min_length_ang", "filament_spacing_ang", "filament_smoothing_ang")}, sort_keys=True)


def pack_filament_scores(scores: Dict[Any, List[Optional[np.ndarray]]], settings: str) -> dict:
    """Arrays of saved picks for ``scores`` (``{}`` without filament classes): ``filament_numbers``,
    ``filament_grid``, ``filament_scores`` ``[F, T, G, 5]`` (NaN where unavailable),
    ``filament_settings``."""
    if not scores:
        return {}
    numbers = list(scores)
    n_tomos = max(len(v) for v in scores.values())
    arr = np.full((len(numbers), n_tomos, len(FILAMENT_GRID), N_FILAMENT_COLUMNS), np.nan, dtype=np.float64)
    for f, n in enumerate(numbers):
        for t, s in enumerate(scores[n]):
            if s is not None:
                arr[f, t] = s
    return {"filament_numbers": np.array(numbers, dtype=np.int64), "filament_grid": np.array(FILAMENT_GRID, dtype=np.float64),
            "filament_scores": arr, "filament_settings": np.array(str(settings))}


def unpack_filament_scores(z) -> dict:
    """``filament_scores`` (``{number: [array or None, ...]}``), ``filament_grid``,
    ``filament_settings`` of saved picks; empty for files without them."""
    if "filament_scores" not in z:
        return {"filament_scores": {}, "filament_grid": [], "filament_settings": ""}
    arr = np.asarray(z["filament_scores"], dtype=np.float64)
    scores = {int(n): [None if np.isnan(arr[f, t]).any() else arr[f, t] for t in range(arr.shape[1])]
              for f, n in enumerate(z["filament_numbers"])}
    return {"filament_scores": scores, "filament_grid": [float(g) for g in z["filament_grid"]],
            "filament_settings": str(z["filament_settings"])}


def _true_polylines(path, class_number: int) -> List[np.ndarray]:
    """True centrelines of one class from a filament CSV: ``x, y, z`` native voxels, in file order."""
    from tomocpt.labels.filaments import read_filament_csv

    df = read_filament_csv(path)
    df = df[df["class_number"] == class_number]
    return [g[["x", "y", "z"]].to_numpy(dtype=np.float64) for _, g in df.groupby("filament_id", sort=False)]


def read_calibration(ckpt_path) -> Optional[dict]:
    """The ``"tomocpt_calibration"`` record of a checkpoint (or of an ensemble file), or None."""
    from tomocpt.predict.ensemble import is_ensemble_file, read_ensemble

    if is_ensemble_file(ckpt_path):
        return read_ensemble(ckpt_path).get("calibration")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return ckpt.get(CALIBRATION_KEY)


def _true_coords(coords_path, class_number: int) -> np.ndarray:
    df = pd.read_csv(coords_path)
    return df.loc[df["class_number"] == class_number, ["x", "y", "z"]].to_numpy(dtype=np.float64)


def collect_validation_picks(model, val_tomograms: List[dict], infer_cfg: InferConfig, device: torch.device,
                             profiles: Optional[dict] = None, filament_scores: Optional[dict] = None):
    """Picks of a pipeline-v2 model on its validation tomograms at the lowest grid threshold
    (after cross-class suppression if ``infer_cfg.cross_class_nms``), with the true particles.
    Returns ``(picks, truths, radii, grid)`` in the format of :func:`calibrate_thresholds`.

    ``profiles``: if a dict, it is filled with ``{class number: accumulator}`` of
    :func:`peak_profile_accumulate` around the true particles of every particle class, pooled
    over the tomograms (summed).

    Filament rows have no points: empty ``picks``, ``truths`` None. ``filament_scores``: if a dict,
    it is filled with ``{class number: [per tomogram scores of filament_threshold_scores or None]}``
    (traced from the same maps, at every threshold of ``FILAMENT_GRID``)."""
    from tomocpt.predict.filaments import is_filament
    from tomocpt.predict.multiclass import (check_offset_head, peaks_from_class_maps, predict_class_maps,
                                            subvoxel_mode)

    warnings.filterwarnings("ignore", message=".*Using TorchIO images without a torchio.SubjectsLoader.*",
                            category=UserWarning)
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
        offsets = {} if subvoxel_mode(infer_cfg) == "offset" else None
        if offsets is not None:
            check_offset_head(model)
        maps, apix = predict_class_maps(model, tomo, tomo_cfg, device, classes=all_classes, offsets_out=offsets)
        peaks, _ = peaks_from_class_maps(maps, table, apix, infer_cfg, lowest, tomo.stem, device, offsets=offsets)
        del offsets
        available = entry.get("class_available") or [True] * len(table)
        for c, row in enumerate(table):
            # Radii in native voxels, as the coordinates (the same for every tomogram of
            # one voxel size; the last one wins otherwise).
            radii[row.number] = float(row.diameter_ang) / (2.0 * apix)
            p = peaks.get(c, np.zeros((0, 4)))
            picks[row.number].append((p[:, [2, 1, 0]], p[:, 3]))
            if is_filament(row):
                truths[row.number].append(None)
                if filament_scores is not None:
                    path = entry.get("filaments_path")
                    filament_scores.setdefault(row.number, []).append(
                        filament_threshold_scores(maps[c].detach().float().cpu().numpy(),
                                                  _true_polylines(path, row.number), row.diameter_ang, apix,
                                                  infer_cfg, FILAMENT_GRID)
                        if path and available[c] and c in maps else None)
                continue
            truths[row.number].append(_true_coords(entry["coords_path"], row.number) if available[c] else None)
            if profiles is not None and available[c] and getattr(row, "kind", PARTICLE) == PARTICLE and c in maps:
                acc = peak_profile_accumulate(maps[c], truths[row.number][-1], radii[row.number])
                profiles[row.number] = acc if row.number not in profiles else _add_profiles(profiles[row.number], acc)
        del maps
    return picks, truths, radii, grid


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
    from tomocpt.predict.filaments import filament_rows

    if infer_cfg.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    from tomocpt.predict.ensemble import isolated_checkpoint_load as _isolated_load

    with _isolated_load():
        model = InferenceModel.load_from_checkpoint(str(ckpt_path), map_location=device, strict=True).eval()
    if not model.is_pipeline_v2():
        raise ValueError(f"{ckpt_path} is not a pipeline-v2 checkpoint; nothing to calibrate.")
    table = model.class_table()
    profiles: dict = {}
    fil_scores: dict = {}
    # Particle-only tables keep the call of Gate 4c (its signature is a frozen interface).
    fil_kw = {"filament_scores": fil_scores} if filament_rows(table) else {}
    picks, truths, radii, grid = collect_validation_picks(model, val_tomograms, infer_cfg, device, profiles=profiles,
                                                          **fil_kw)

    classes = calibrate_thresholds(picks, truths, radii, beta=infer_cfg.calibration_beta,
                                   min_particles=infer_cfg.calibration_min_particles, grid=grid)
    classes.update(calibrate_filament_thresholds(fil_scores, FILAMENT_GRID, beta=infer_cfg.filament_calibration_beta))
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
        what = "true filament(s)" if "kind" in e else "true particle(s)"
        logger.info(f"Calibration, class {row.name}: threshold {thr}, F-beta {e['fbeta']:.3f}, {e['n_true']} {what}")

    del model
    try:
        write_val_picks(val_picks_path(ckpt_path), [str(e["tomogram_path"]) for e in val_tomograms], table,
                        picks, truths, radii, grid, profiles, picks_fingerprint(infer_cfg),
                        filaments=pack_filament_scores(fil_scores, filament_settings(infer_cfg)))
    except Exception as e:  # the calibration itself does not depend on the saved picks
        logger.error(f"Could not save the validation picks of {ckpt_path}: {e}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt[CALIBRATION_KEY] = record
    tmp = Path(str(ckpt_path) + ".calibration.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, ckpt_path)
    return record


# ------------------------------------------------------------------------------------------
# Peak profiles around true particles (Gate 4c, TODO 3a.3)
# ------------------------------------------------------------------------------------------

#: Radial bins of 0.05 R from 0 to 1.5 R; line samples from -1.5 R to 1.5 R in steps of 0.05 R.
PROFILE_STEP = 0.05
PROFILE_EXTENT = 1.5
N_RADIAL_BINS = int(round(PROFILE_EXTENT / PROFILE_STEP))
N_LINE_SAMPLES = 2 * N_RADIAL_BINS + 1
_PROFILE_KEYS = ("radial_sum", "radial_count", "line_sum", "line_count", "peak_sum", "n")


def empty_profile() -> dict:
    return {"radial_sum": np.zeros(N_RADIAL_BINS), "radial_count": np.zeros(N_RADIAL_BINS),
            "line_sum": np.zeros((3, N_LINE_SAMPLES)), "line_count": np.zeros((3, N_LINE_SAMPLES)),
            "peak_sum": 0.0, "n": 0}


def _add_profiles(a: dict, b: dict) -> dict:
    return {k: (a[k] + b[k]) for k in _PROFILE_KEYS}


def peak_profile_accumulate(class_map, centres_xyz, radius: float) -> dict:
    """Sums and counts of a probability map around true centres, so tomograms and folds pool
    by addition.

    ``class_map``: ``[D, H, W]`` (numpy or torch); ``centres_xyz``: ``[N, 3]`` in voxels of the
    map; ``radius``: class radius R in voxels. Returns numpy ``radial_sum, radial_count``
    (``[30]``, bins of 0.05 R by distance from the centre to the voxel centre, 0 to 1.5 R),
    ``line_sum, line_count`` (``[3, 61]``: lines along x, y, z through the centre, trilinear,
    t = -1.5 R .. 1.5 R in steps of 0.05 R; samples outside the map are skipped), ``peak_sum``
    (sum over centres of the map maximum within 0.5 R) and ``n`` (number of centres).
    """
    acc = empty_profile()
    if centres_xyz is None:
        return acc
    centres = np.asarray(centres_xyz, dtype=np.float64).reshape(-1, 3)
    if torch.is_tensor(class_map):
        vol_t = class_map.detach()
    else:
        vol_t = torch.as_tensor(np.asarray(class_map))
    shape = np.array(vol_t.shape)  # (D, H, W) = (z, y, x)
    R = float(radius)
    half = int(np.ceil(PROFILE_EXTENT * R)) + 1
    step = PROFILE_STEP * R
    t = (np.arange(N_LINE_SAMPLES) - N_RADIAL_BINS) * PROFILE_STEP * R
    for x, y, z in centres:
        c_zyx = np.array([z, y, x])
        lo = np.maximum(np.floor(c_zyx).astype(int) - half, 0)
        hi = np.minimum(np.floor(c_zyx).astype(int) + half + 2, shape)
        if np.any(hi <= lo):
            continue
        box = vol_t[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].float().cpu().numpy().astype(np.float64)
        gz, gy, gx = np.meshgrid(*[np.arange(a, b, dtype=np.float64) for a, b in zip(lo, hi)], indexing="ij")
        d = np.sqrt((gz - z) ** 2 + (gy - y) ** 2 + (gx - x) ** 2)
        b = np.floor(d / step).astype(np.int64)
        ok = b < N_RADIAL_BINS
        acc["radial_sum"] += np.bincount(b[ok], weights=box[ok], minlength=N_RADIAL_BINS)
        acc["radial_count"] += np.bincount(b[ok], minlength=N_RADIAL_BINS).astype(np.float64)
        core = d <= 0.5 * R
        acc["peak_sum"] += float(box[core].max()) if core.any() else 0.0
        acc["n"] += 1
        for ax_i, ax in enumerate((2, 1, 0)):  # x, y, z -> array axes W, H, D
            pts = np.repeat(c_zyx[None, :], N_LINE_SAMPLES, axis=0)
            pts[:, ax] += t
            vals, valid = _trilinear(box, pts - lo[None, :])
            acc["line_sum"][ax_i] += np.where(valid, vals, 0.0)
            acc["line_count"][ax_i] += valid.astype(np.float64)
    return acc


def _trilinear(vol: np.ndarray, pts_zyx: np.ndarray):
    """Trilinear samples of ``vol`` at ``pts_zyx`` [N, 3]; ``valid`` False where a corner is outside."""
    f = np.floor(pts_zyx).astype(np.int64)
    w = pts_zyx - f
    hi = np.array(vol.shape) - 1
    # a sample exactly on the last voxel needs no upper corner
    f = np.where((f == hi[None, :]) & (w == 0), f - 1, f)
    w = pts_zyx - f
    valid = np.all((f >= 0) & (f + 1 <= hi[None, :]), axis=1)
    out = np.zeros(len(pts_zyx))
    if valid.any():
        fz, fy, fx = f[valid].T
        wz, wy, wx = w[valid].T
        v = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    wt = (wz if dz else 1 - wz) * (wy if dy else 1 - wy) * (wx if dx else 1 - wx)
                    v = v + wt * vol[fz + dz, fy + dy, fx + dx]
        out[valid] = v
    return out, valid


# ------------------------------------------------------------------------------------------
# Saved validation picks (Gate 4c, TODO 3a.2)
# ------------------------------------------------------------------------------------------

#: Infer settings that change the picks of ``collect_validation_picks`` (with the values the
#: calibration forces: no contrast inversion, no masks, every class).
_FINGERPRINT_FIELDS = ("peak_spacing_fraction", "distance_threshold", "cross_class_nms", "use_tta", "length",
                       "accumulate_dtype", "PATCH_OVERLAP_FACTOR", "invert_contrast", "masks_dir", "classes",
                       "apix_override", "subvoxel_refine")


def picks_fingerprint(infer_cfg: InferConfig) -> str:
    """JSON of the infer settings that change validation picks."""
    vals = {}
    for k in _FINGERPRINT_FIELDS:
        v = getattr(infer_cfg, k, None)
        vals[k] = None if v is None else (v.value if hasattr(v, "value") else (str(v) if isinstance(v, Path) else v))
    return json.dumps(vals, sort_keys=True, default=str)


def val_picks_path(ckpt) -> Path:
    """``<ckpt stem>.val_picks.npz`` next to the checkpoint."""
    ckpt = Path(str(ckpt))
    return ckpt.with_name(ckpt.stem + ".val_picks.npz")


def pack_picks(table, picks, truths, radii, profiles, tomogram_fold=None) -> dict:
    """Flat numpy arrays of picks / truths / profiles (format of spec 04c D3). With
    ``tomogram_fold`` (one fold per tomogram), ``picks`` / ``truths`` get a leading fold column."""
    numbers = [int(r.number) for r in table]
    n_tomos = len(next(iter(picks.values()))) if picks else 0
    prow, trow = [], []
    available = np.ones((n_tomos, len(numbers)), dtype=bool)
    for ci, n in enumerate(numbers):
        for ti in range(n_tomos):
            fold = [] if tomogram_fold is None else [float(tomogram_fold[ti])]
            xyz, score = picks[n][ti]
            xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
            score = np.asarray(score, dtype=np.float64).reshape(-1)
            for p, s in zip(xyz, score):
                prow.append(fold + [ti, n, p[0], p[1], p[2], s])
            tr = truths[n][ti]
            if tr is None:
                available[ti, ci] = False
                continue
            for p in np.asarray(tr, dtype=np.float64).reshape(-1, 3):
                trow.append(fold + [ti, n, p[0], p[1], p[2]])
    extra = 0 if tomogram_fold is None else 1
    prof = {k: [] for k in _PROFILE_KEYS}
    for n in numbers:
        acc = (profiles or {}).get(n) or empty_profile()
        for k in _PROFILE_KEYS:
            prof[k].append(acc[k])
    return {
        "class_numbers": np.array(numbers, dtype=np.int64),
        "radii": np.array([float(radii.get(n, np.nan)) for n in numbers], dtype=np.float64),
        "picks": np.array(prow, dtype=np.float64).reshape(-1, 6 + extra),
        "truths": np.array(trow, dtype=np.float64).reshape(-1, 5 + extra),
        "available": available,
        "profile_radial_sum": np.array(prof["radial_sum"], dtype=np.float64).reshape(-1, N_RADIAL_BINS),
        "profile_radial_count": np.array(prof["radial_count"], dtype=np.float64).reshape(-1, N_RADIAL_BINS),
        "profile_line_sum": np.array(prof["line_sum"], dtype=np.float64).reshape(-1, 3, N_LINE_SAMPLES),
        "profile_line_count": np.array(prof["line_count"], dtype=np.float64).reshape(-1, 3, N_LINE_SAMPLES),
        "profile_peak_sum": np.array(prof["peak_sum"], dtype=np.float64),
        "profile_n": np.array(prof["n"], dtype=np.int64),
    }


def unpack_picks(z, fold_column: bool = False) -> dict:
    """Inverse of :func:`pack_picks`: ``picks``, ``truths`` (``calibrate_thresholds`` formats,
    tomograms in file order), ``radii``, ``available``, ``profiles``."""
    numbers = [int(n) for n in z["class_numbers"]]
    available = np.asarray(z["available"], dtype=bool)
    n_tomos = available.shape[0]
    off = 1 if fold_column else 0
    P, T = np.asarray(z["picks"], np.float64), np.asarray(z["truths"], np.float64)
    picks, truths = {}, {}
    for ci, n in enumerate(numbers):
        picks[n], truths[n] = [], []
        for ti in range(n_tomos):
            sel = P[(P[:, off] == ti) & (P[:, off + 1] == n)]
            picks[n].append((sel[:, off + 2:off + 5], sel[:, off + 5]))
            if not available[ti, ci]:
                truths[n].append(None)
            else:
                tsel = T[(T[:, off] == ti) & (T[:, off + 1] == n)]
                truths[n].append(tsel[:, off + 2:off + 5])
    profiles = {}
    for ci, n in enumerate(numbers):
        profiles[n] = {"radial_sum": np.asarray(z["profile_radial_sum"][ci], np.float64),
                       "radial_count": np.asarray(z["profile_radial_count"][ci], np.float64),
                       "line_sum": np.asarray(z["profile_line_sum"][ci], np.float64),
                       "line_count": np.asarray(z["profile_line_count"][ci], np.float64),
                       "peak_sum": float(z["profile_peak_sum"][ci]), "n": int(z["profile_n"][ci])}
    radii = {n: float(r) for n, r in zip(numbers, z["radii"])}
    return {"picks": picks, "truths": truths, "radii": radii, "available": available, "profiles": profiles}


def _savez_atomic(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_val_picks(path, tomograms: Sequence[str], table, picks, truths, radii, grid, profiles, fingerprint: str,
                    filaments: Optional[dict] = None) -> Path:
    """Saves the validation picks of one checkpoint (spec 04c D3). ``filaments``: the arrays of
    :func:`pack_filament_scores`."""
    arrays = pack_picks(table, picks, truths, radii, profiles)
    _savez_atomic(Path(path), tomograms=np.array([str(t) for t in tomograms], dtype=str),
                  grid=np.asarray(list(grid), dtype=np.float64), fingerprint=np.array(str(fingerprint)), **arrays,
                  **(filaments or {}))
    return Path(path)


def read_val_picks(path) -> dict:
    """The saved validation picks: ``picks``, ``truths``, ``radii``, ``grid`` (formats of
    :func:`calibrate_thresholds`), ``tomograms``, ``available``, ``fingerprint``, ``profiles``, and those of
    :func:`unpack_filament_scores`."""
    with np.load(str(path), allow_pickle=False) as z:
        out = unpack_picks(z)
        out["grid"] = [float(g) for g in z["grid"]]
        out["tomograms"] = [str(t) for t in z["tomograms"]]
        out["fingerprint"] = str(z["fingerprint"])
        out.update(unpack_filament_scores(z))
    return out
