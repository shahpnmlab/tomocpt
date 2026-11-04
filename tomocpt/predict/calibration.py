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
                             profiles: Optional[dict] = None):
    """Picks of a pipeline-v2 model on its validation tomograms at the lowest grid threshold
    (after cross-class suppression if ``infer_cfg.cross_class_nms``), with the true particles.
    Returns ``(picks, truths, radii, grid)`` in the format of :func:`calibrate_thresholds`.

    ``profiles``: if a dict, it is filled with ``{class number: accumulator}`` of
    :func:`peak_profile_accumulate` around the true particles of every particle class, pooled
    over the tomograms (summed)."""
    from tomocpt.predict.multiclass import peaks_from_class_maps, predict_class_maps

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
        maps, apix = predict_class_maps(model, tomo, tomo_cfg, device, classes=all_classes)
        peaks, _ = peaks_from_class_maps(maps, table, apix, infer_cfg, lowest, tomo.stem, device)
        available = entry.get("class_available") or [True] * len(table)
        for c, row in enumerate(table):
            # Radii in native voxels, as the coordinates (the same for every tomogram of
            # one voxel size; the last one wins otherwise).
            radii[row.number] = float(row.diameter_ang) / (2.0 * apix)
            p = peaks.get(c, np.zeros((0, 4)))
            picks[row.number].append((p[:, [2, 1, 0]], p[:, 3]))
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
    picks, truths, radii, grid = collect_validation_picks(model, val_tomograms, infer_cfg, device, profiles=profiles)

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
    try:
        write_val_picks(val_picks_path(ckpt_path), [str(e["tomogram_path"]) for e in val_tomograms], table,
                        picks, truths, radii, grid, profiles, picks_fingerprint(infer_cfg))
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
                       "apix_override")


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


def write_val_picks(path, tomograms: Sequence[str], table, picks, truths, radii, grid, profiles, fingerprint: str) -> Path:
    """Saves the validation picks of one checkpoint (spec 04c D3)."""
    arrays = pack_picks(table, picks, truths, radii, profiles)
    _savez_atomic(Path(path), tomograms=np.array([str(t) for t in tomograms], dtype=str),
                  grid=np.asarray(list(grid), dtype=np.float64), fingerprint=np.array(str(fingerprint)), **arrays)
    return Path(path)


def read_val_picks(path) -> dict:
    """The saved validation picks: ``picks``, ``truths``, ``radii``, ``grid`` (formats of
    :func:`calibrate_thresholds`), ``tomograms``, ``available``, ``fingerprint``, ``profiles``."""
    with np.load(str(path), allow_pickle=False) as z:
        out = unpack_picks(z)
        out["grid"] = [float(g) for g in z["grid"]]
        out["tomograms"] = [str(t) for t in z["tomograms"]]
        out["fingerprint"] = str(z["fingerprint"])
    return out
