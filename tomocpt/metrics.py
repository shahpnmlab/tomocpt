"""Picking metrics: point matching, precision / recall / F-beta, per-class scores.

Coordinates are arrays of shape ``[N, 3]`` in any consistent unit (voxels or Å);
the match radius must be in the same unit. β = 4 is the CZII challenge metric.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _as_points(xyz) -> np.ndarray:
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def match_points(pred_xyz, true_xyz, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy one-to-one matching of predictions to truths by increasing distance.

    Pairs closer than or at ``radius`` are taken shortest first; each prediction
    and each truth is used at most once.

    Parameters
    ----------
    pred_xyz : array-like, [P, 3]
        Predicted coordinates.
    true_xyz : array-like, [T, 3]
        Ground-truth coordinates.
    radius : float
        Maximum match distance.

    Returns
    -------
    tp_pairs : np.ndarray, int64 [K, 2]
        Matched ``(pred_index, true_index)`` rows.
    fp_idx : np.ndarray, int64 [P - K]
        Indices of unmatched predictions.
    fn_idx : np.ndarray, int64 [T - K]
        Indices of unmatched truths.
    """
    pred = _as_points(pred_xyz)
    true = _as_points(true_xyz)
    n_pred, n_true = len(pred), len(true)

    pairs = np.zeros((0, 2), dtype=np.int64)
    if n_pred and n_true:
        dist = cKDTree(pred).sparse_distance_matrix(
            cKDTree(true), max_distance=float(radius), output_type="ndarray"
        )
        if len(dist):
            # Sort by distance; ties broken by (pred, true) index for determinism.
            order = np.lexsort((dist["j"], dist["i"], dist["v"]))
            pred_used = np.zeros(n_pred, dtype=bool)
            true_used = np.zeros(n_true, dtype=bool)
            matched = []
            for k in order:
                i, j = int(dist["i"][k]), int(dist["j"][k])
                if pred_used[i] or true_used[j]:
                    continue
                pred_used[i] = True
                true_used[j] = True
                matched.append((i, j))
            if matched:
                pairs = np.asarray(matched, dtype=np.int64)

    fp_idx = np.setdiff1d(np.arange(n_pred, dtype=np.int64), pairs[:, 0])
    fn_idx = np.setdiff1d(np.arange(n_true, dtype=np.int64), pairs[:, 1])
    return pairs, fp_idx, fn_idx


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def prf_beta(tp: int, fp: int, fn: int, beta: float = 4.0) -> Dict[str, float]:
    """Precision, recall, F1 and F-beta from counts. Zero denominators give 0.0.

    Returns
    -------
    dict
        ``{"precision", "recall", "f1", "fbeta"}``.
    """
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    b2 = float(beta) ** 2
    fbeta = _safe_div((1 + b2) * precision * recall, b2 * precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "fbeta": fbeta}


def score_classes(
    pred: Dict[str, np.ndarray],
    true: Dict[str, np.ndarray],
    radii: Dict[str, float],
    match_radius_frac: float = 0.5,
    beta: float = 4.0,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Score each class and their (optionally weighted) macro mean.

    The match radius of class ``name`` is ``match_radius_frac * radii[name]``.
    A class with truth but no entry in ``pred`` counts as all-missed; a class with
    predictions but no entry in ``true`` counts as all false positives.

    Parameters
    ----------
    pred, true : dict[str, array-like [N, 3]]
        Coordinates per class name.
    radii : dict[str, float]
        Particle radius per class, in the unit of the coordinates.
    match_radius_frac : float
        Fraction of the radius within which a prediction matches.
    beta : float
        F-beta parameter.
    weights : dict[str, float], optional
        Per-class weights for the macro mean (missing classes weigh 1.0).
        ``None`` gives the plain mean.

    Returns
    -------
    dict
        ``{"per_class": {name: {precision, recall, f1, fbeta, tp, fp, fn}},
        "macro": {precision, recall, f1, fbeta}}``.
    """
    empty = np.zeros((0, 3))
    names = list(true) + [n for n in pred if n not in true]

    per_class = {}
    for name in names:
        p = _as_points(pred.get(name, empty))
        t = _as_points(true.get(name, empty))
        if len(p) and len(t):
            tp_pairs, fp_idx, fn_idx = match_points(p, t, match_radius_frac * float(radii[name]))
            tp, fp, fn = len(tp_pairs), len(fp_idx), len(fn_idx)
        else:
            tp, fp, fn = 0, len(p), len(t)
        scores = prf_beta(tp, fp, fn, beta=beta)
        scores.update(tp=tp, fp=fp, fn=fn)
        per_class[name] = scores

    keys = ("precision", "recall", "f1", "fbeta")
    w = np.array([1.0 if weights is None else float(weights.get(n, 1.0)) for n in names])
    if len(names) == 0 or w.sum() <= 0:
        macro = {k: 0.0 for k in keys}
    else:
        macro = {
            k: float(np.dot(w, [per_class[n][k] for n in names]) / w.sum()) for k in keys
        }
    return {"per_class": per_class, "macro": macro}


# ------------------------------------------------------------------------------------------
# Filament centrelines (Gate 6; moved here at Gate 6b so the calibration can use it)
# ------------------------------------------------------------------------------------------

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


def _within_tolerance(points: np.ndarray, polylines: Sequence[np.ndarray], tol: float, step: float) -> np.ndarray:
    """``distance(points, polylines) <= tol``. A KD-tree on nodes at most ``step`` apart decides
    all but the points whose node distance lies in ``(tol, sqrt(tol^2 + (step/2)^2)]``; only those
    need the exact distance."""
    nodes = np.concatenate([_densify(p, step) for p in polylines])
    d, _ = cKDTree(nodes).query(points)
    inside = d <= tol
    band = ~inside & (d <= np.sqrt(tol * tol + 0.25 * step * step) + 1e-9)
    if band.any():
        inside[band] = _point_to_polylines(points[band], polylines) <= tol
    return inside


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
            near += float(seg[_within_tolerance(mid, ref, tol, step)].sum())
    return near, total


def centerline_scores(pred_polylines: Sequence[np.ndarray], true_polylines: Sequence[np.ndarray],
                      tolerance_vox: float, step: float = 0.25) -> Dict[str, float]:
    """Length-weighted ``{"precision", "recall", "f1", "pred_length", "true_length"}``: precision
    is the fraction of predicted length within ``tolerance_vox`` of a true centreline, recall the
    fraction of true length within ``tolerance_vox`` of a predicted one. Polylines are ``[K, 3]``
    ``x, y, z`` voxels."""
    pred = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in pred_polylines if len(p)]
    true = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in true_polylines if len(p)]
    tp_p, len_p = _covered_length(pred, true, float(tolerance_vox), step)
    tp_t, len_t = _covered_length(true, pred, float(tolerance_vox), step)
    precision = tp_p / len_p if len_p > 0 else 0.0
    recall = tp_t / len_t if len_t > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "pred_length": float(len_p), "true_length": float(len_t)}
