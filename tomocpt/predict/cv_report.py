"""CV report of a k-fold ensemble (Gate 4c, TODO 3a.1 / 3a.3).

``write_cv_report(ensemble_path, out_dir)`` reads an ensemble file and its pooled out-of-fold
picks (``oof_picks.npz``) and writes ``cv_summary.json``, ``cv_summary.md`` and
``peak_profiles.png``: per class the out-of-fold F-beta at the calibrated threshold (also per
fold), precision, recall and counts, the localisation of matched picks, and the mean predicted
profile around true particles compared with the training target.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tomocpt.logger import get_logger
from tomocpt.metrics import match_points, prf_beta

logger = get_logger()

#: Match radius as a fraction of the class radius (Gate 4 calibration).
MATCH_RADIUS_FRAC = 0.5
#: Threshold used for a class without a calibrated threshold (``infer.confidence_threshold`` default).
FALLBACK_THRESHOLD = 0.3


def _target(r_in_R: np.ndarray) -> np.ndarray:
    """CLAUDE.md 1.3 target profile in units of R (0.5 narrow sigma 0.165 R + 0.5 wide sigma 0.33 R, cut at R)."""
    r = np.abs(np.asarray(r_in_R, dtype=np.float64))
    v = 0.5 * np.exp(-r * r / (2 * 0.165 ** 2)) + 0.5 * np.exp(-r * r / (2 * 0.33 ** 2))
    return np.where(r > 1.0, 0.0, v)


def _fwhm(r: np.ndarray, prof: np.ndarray) -> Optional[float]:
    """2 x the radius where ``prof`` falls to half its first finite value (linear interpolation)."""
    ok = np.isfinite(prof)
    if not ok.any():
        return None
    r, prof = r[ok], prof[ok]
    half = 0.5 * prof[0]
    if half <= 0:
        return None
    below = np.nonzero(prof <= half)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = below[0]
    r0, r1, p0, p1 = r[i - 1], r[i], prof[i - 1], prof[i]
    return float(2.0 * (r0 + (p0 - half) / (p0 - p1) * (r1 - r0))) if p0 != p1 else float(2.0 * r1)


def peak_profile_stats(acc: dict) -> dict:
    """Profiles and summary numbers from an accumulator of
    :func:`tomocpt.predict.calibration.peak_profile_accumulate`. Distances in units of R."""
    from tomocpt.predict.calibration import N_RADIAL_BINS, N_LINE_SAMPLES, PROFILE_STEP

    r = (np.arange(N_RADIAL_BINS) + 0.5) * PROFILE_STEP
    t = (np.arange(N_LINE_SAMPLES) - N_RADIAL_BINS) * PROFILE_STEP
    with np.errstate(invalid="ignore", divide="ignore"):
        radial = np.asarray(acc["radial_sum"], float) / np.asarray(acc["radial_count"], float)
        line = np.asarray(acc["line_sum"], float) / np.asarray(acc["line_count"], float)
    n = int(acc["n"])
    return {
        "r": r, "radial": radial, "t": t, "line": line,
        "peak_height": float(acc["peak_sum"]) / n if n else None,
        "centre": float(radial[0]) if np.isfinite(radial[0]) else None,
        "fwhm": _fwhm(r, radial),
        "target_radial": _target(r), "target_line": _target(t),
        "target_fwhm": 0.530, "n": n,
    }


def _parse_weights(text, names) -> Dict[str, float]:
    weights = {n: 1.0 for n in names}
    if text is None or not str(text).strip() or str(text).strip() in ("None", "???"):
        return weights
    for entry in str(text).split(","):
        if not entry.strip():
            continue
        name, sep, w = entry.rpartition(":")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"Malformed class weight '{entry}': expected '<class name>:<weight>'")
        if name not in weights:
            raise ValueError(f"cv_class_weights names an unknown class '{name}'; classes: {sorted(names)}")
        weights[name] = float(w)
    return weights


def _class_scores(oof: dict, number: int, threshold: float, beta: float) -> dict:
    radius = float(oof["radii"][number])
    match = MATCH_RADIUS_FRAC * radius
    folds = oof["tomogram_fold"]
    tot = np.zeros(3, dtype=np.int64)
    per_fold: Dict[int, np.ndarray] = {}
    dist, offs = [], []
    n_true = 0
    for ti, ((xyz, score), tr) in enumerate(zip(oof["picks"][number], oof["truths"][number])):
        if tr is None:
            continue
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        score = np.asarray(score, float).reshape(-1)
        tr = np.asarray(tr, float).reshape(-1, 3)
        n_true += len(tr)
        px = xyz[score >= threshold]
        pairs, fp, fn = match_points(px, tr, match)
        c = np.array([len(pairs), len(fp), len(fn)])
        tot += c
        f = int(folds[ti])
        per_fold[f] = per_fold.get(f, np.zeros(3, dtype=np.int64)) + c
        for i, j in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
            d = px[i] - tr[j]
            offs.append(d)
            dist.append(float(np.linalg.norm(d)))
    s = prf_beta(int(tot[0]), int(tot[1]), int(tot[2]), beta=beta)
    dist = np.array(dist)
    loc = {"n_matched": int(len(dist)),
           "mean_vox": float(dist.mean()) if len(dist) else None,
           "median_vox": float(np.median(dist)) if len(dist) else None,
           "mean_R": float(dist.mean() / radius) if len(dist) else None,
           "median_R": float(np.median(dist) / radius) if len(dist) else None,
           "mean_offset_xyz": [float(v) for v in np.mean(offs, axis=0)] if offs else None}
    return {"tp": int(tot[0]), "fp": int(tot[1]), "fn": int(tot[2]), "n_true": int(n_true),
            "precision": float(s["precision"]), "recall": float(s["recall"]), "fbeta": float(s["fbeta"]),
            "per_fold_fbeta": {str(k): float(prf_beta(*map(int, v), beta=beta)["fbeta"]) for k, v in sorted(per_fold.items())},
            "radius_vox": radius, "loc": loc}


def _filament_class_scores(oof: dict, number: int, threshold: Optional[float], beta: float) -> dict:
    """Length-weighted centreline scores of a filament class (same keys as :func:`_class_scores`;
    point counts and localisation are None). All 0 without a calibrated threshold."""
    from tomocpt.predict.calibration import _length_scores

    per_tomo = oof["filament_scores"][number]
    folds = oof["tomogram_fold"]
    grid = oof["filament_grid"]
    rows = [(int(folds[t]), s) for t, s in enumerate(per_tomo) if s is not None]
    s = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fbeta": 0.0}
    per_fold: Dict[int, np.ndarray] = {}
    if threshold is not None and rows:
        i = int(np.argmin(np.abs(np.asarray(grid) - float(threshold))))
        for f, row in rows:
            per_fold[f] = per_fold.get(f, np.zeros(row.shape[1])) + row[i]
        s = _length_scores(sum(per_fold.values()), beta)
    n_true = int(sum(row[0, 4] for _, row in rows))
    true_len = float(sum(row[0, 3] for _, row in rows))
    return {"kind": "filament", "tp": None, "fp": None, "fn": None, "n_true": n_true, "true_length_ang": true_len,
            **s, "per_fold_fbeta": {str(k): _length_scores(v, beta)["fbeta"] for k, v in sorted(per_fold.items())},
            "radius_vox": float(oof["radii"][number]),
            "loc": {"n_matched": 0, "mean_vox": None, "median_vox": None, "mean_R": None, "median_R": None,
                    "mean_offset_xyz": None}}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def _markdown(rep: dict) -> str:
    L = [f"## k-fold CV report ({rep['n_folds']} folds)", "",
         f"- ensemble: `{rep['ensemble']}`; out-of-fold tomograms: {rep['n_tomograms']}",
         f"- F{rep['beta']:g} of pooled out-of-fold picks at the thresholds calibrated on them (slightly optimistic)",
         "- localisation: distance of matched picks to the true centres of the coordinate files "
         "(whole-voxel truths until TODO 3b.1)", "",
         "| class | true | threshold | F-beta | precision | recall | TP | FP | FN | per fold | dist. mean / median (R) | mean offset x, y, z (vox) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in rep["classes"].items():
        thr = "None (" + _fmt(c["threshold_used"], ".2f") + ")" if c["threshold"] is None else _fmt(c["threshold"], ".2f")
        pf = " ".join(_fmt(v, ".2f") for v in c["per_fold_fbeta"].values())
        loc = c["loc"]
        off = "-" if loc["mean_offset_xyz"] is None else ", ".join(f"{v:+.2f}" for v in loc["mean_offset_xyz"])
        L.append(f"| {name} | {c['n_true']} | {thr} | {_fmt(c['fbeta'])} | {_fmt(c['precision'])} | "
                 f"{_fmt(c['recall'])} | {_fmt(c['tp'], 'd')} | {_fmt(c['fp'], 'd')} | {_fmt(c['fn'], 'd')} | {pf} | "
                 f"{_fmt(loc['mean_R'], '.2f')} / {_fmt(loc['median_R'], '.2f')} | {off} |")
    kind = "weighted" if rep["weighted"] else "unweighted"
    L += ["", f"Mean F-beta ({kind}): **{rep['mean_fbeta']:.3f}**"]
    if rep["weighted"]:
        L.append("Weights: " + ", ".join(f"{k} {v:g}" for k, v in rep["weights"].items()))
    fil = {n: c for n, c in rep["classes"].items() if c.get("kind") == "filament"}
    if fil:
        L += ["", "### Filament classes (length-weighted centreline, within 0.5 x diameter)", "",
              "| class | true filaments | true length (A) | threshold | precision | recall | F1 | F-beta |",
              "|---|---|---|---|---|---|---|---|"]
        for name, c in fil.items():
            L.append(f"| {name} | {c['n_true']} | {c['true_length_ang']:.0f} | {_fmt(c['threshold'], '.2f')} | "
                     f"{_fmt(c['precision'])} | {_fmt(c['recall'])} | {_fmt(c['f1'])} | {_fmt(c['fbeta'])} |")
    L += ["", "### Peak profiles around true centres (units of R; target: peak 1.0, FWHM 0.53 R)", "",
          "| class | particles | peak height | value at centre | FWHM (radial) |", "|---|---|---|---|---|"]
    for name, c in rep["classes"].items():
        p = c.get("profile")
        if p and p["n"]:
            L.append(f"| {name} | {p['n']} | {_fmt(p['peak_height'], '.2f')} | {_fmt(p['centre'], '.2f')} | "
                     f"{_fmt(p['fwhm'], '.2f')} |")
    return "\n".join(L) + "\n"


def _plot(rep: dict, stats: Dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, s in stats.items() if s["n"]]
    cols = max(1, len(names))
    fig, axes = plt.subplots(1, cols, figsize=(3.6 * cols, 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        s = stats[name]
        ax.plot(s["r"], s["target_radial"], color="0.6", lw=2, label="target")
        ax.plot(s["r"], s["radial"], color="k", lw=1.6, label="radial")
        for i, (lab, col) in enumerate(zip("xyz", ("tab:blue", "tab:orange", "tab:green"))):
            half = s["t"] >= 0
            ax.plot(s["t"][half], s["line"][i][half], color=col, lw=1, ls="--", label=f"{lab}+")
            ax.plot(-s["t"][~half], s["line"][i][~half], color=col, lw=1, ls=":", label=f"{lab}-")
        ax.set_title(f"{name} (n={s['n']})", fontsize=9)
        ax.set_xlabel("distance / R")
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.05)
    if not names:
        axes[0][0].text(0.5, 0.5, "no particle profiles", ha="center", va="center")
        axes[0][0].set_axis_off()
    else:
        axes[0][0].set_ylabel("mean probability")
        axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_cv_report(ensemble_path, out_dir, class_weights: Optional[str] = None) -> dict:
    """Writes ``cv_summary.json``, ``cv_summary.md`` and ``peak_profiles.png`` to ``out_dir``
    and returns the content of ``cv_summary.json``. ``class_weights``: ``"name:weight,..."``
    for the mean F-beta (classes not named weigh 1); an unknown name raises ValueError."""
    from tomocpt.predict.ensemble import read_ensemble, read_oof_picks

    ensemble_path, out_dir = Path(ensemble_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = read_ensemble(ensemble_path)
    cal = record["calibration"]
    beta = float(cal.get("beta", 4.0))
    oof = read_oof_picks(ensemble_path)
    names = oof["class_names"]
    weights = _parse_weights(class_weights, list(names.values()))
    classes, stats = {}, {}
    for number, name in names.items():
        entry = cal["classes"].get(str(number), cal["classes"].get(number, {})) or {}
        thr = entry.get("threshold")
        if number in oof["filament_scores"]:
            c = _filament_class_scores(oof, number, thr, beta)
            c.update(number=int(number), threshold=None if thr is None else float(thr),
                     threshold_used=None if thr is None else float(thr))
            classes[name] = c
            continue
        used = FALLBACK_THRESHOLD if thr is None else float(thr)
        c = _class_scores(oof, number, used, beta)
        c.update(number=int(number), threshold=None if thr is None else float(thr), threshold_used=used)
        acc = oof["profiles"].get(number)
        if acc is not None and int(acc["n"]) > 0:
            s = peak_profile_stats(acc)
            stats[name] = s
            c["profile"] = s
        classes[name] = c
    wsum = sum(weights[n] for n in classes)
    mean = sum(weights[n] * classes[n]["fbeta"] for n in classes) / wsum if wsum else 0.0
    weighted = class_weights is not None and bool(str(class_weights).strip())
    rep = {"ensemble": str(ensemble_path), "beta": beta, "n_folds": int(len(set(oof["tomogram_fold"].tolist()))),
           "n_tomograms": len(oof["tomograms"]), "tomograms": oof["tomograms"],
           "classes": classes, "weights": weights, "weighted": weighted, "mean_fbeta": float(mean)}
    rep = _jsonable(rep)
    (out_dir / "cv_summary.json").write_text(json.dumps(rep, indent=1))
    (out_dir / "cv_summary.md").write_text(_markdown(rep))
    _plot(rep, stats, out_dir / "peak_profiles.png")
    logger.info(f"CV report written to {out_dir / 'cv_summary.md'} (mean F-beta {mean:.3f})")
    return rep
