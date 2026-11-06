"""Pictures of filament runs (TODO 3c.3), for looking at what a model does.

* :func:`save_filament_view`: whole tomogram, one class: tomogram, target, prediction and traces, as a
  top (XY) and a side (XZ) view. Traces are green where they lie on a true centreline and red where
  they do not (false); true centrelines that no trace reaches are blue and dashed (missed).
* :func:`save_threshold_curve`: precision, recall and F-beta of the tracing threshold calibration.
* :func:`dump_training_images`: the chunk panels TensorBoard logged during training (central slices
  of input, target and prediction), first and last epoch, as PNG files.
"""

from __future__ import annotations

import re
from pathlib import Path
import sys as _sys
from typing import List, Sequence

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

MATCHED, FALSE, MISSED = "#009E73", "#D55E00", "#2E86C1"


def target_map(true_polylines: Sequence[np.ndarray], shape_zyx, diameter_ang: float, apix: float) -> np.ndarray:
    """The tube target of one class on the whole tomogram, ``[D, H, W]``."""
    from tomocpt.labels.filaments import render_tubes, segments_from_polyline

    d_vox = float(diameter_ang) / float(apix)
    segs = [segments_from_polyline(p, 0, d_vox) for p in true_polylines if len(p) >= 2]
    segs = np.concatenate(segs) if segs else np.zeros((0, 8), dtype=np.float32)
    return render_tubes(segs, shape_zyx, 1)[0]


def _split_runs(points: np.ndarray, good: np.ndarray) -> List[tuple]:
    """Consecutive runs of ``points`` with the same flag: ``[(points, flag), ...]``; runs share an end."""
    runs, start = [], 0
    for i in range(1, len(good) + 1):
        if i == len(good) or good[i] != good[start]:
            runs.append((points[start:min(i + 1, len(points))], bool(good[start])))
            start = i
    return runs


def classify_traces(pred: Sequence[np.ndarray], true: Sequence[np.ndarray], tol_vox: float, step: float = 1.0):
    """``(matched, false, missed)``: lists of ``[K, 3]`` ``x, y, z`` pieces at tolerance ``tol_vox``."""
    from tomocpt.metrics import _PolylineIndex, _densify

    matched, false, missed = [], [], []
    for src, ref, on, off in ((pred, true, matched, false), (true, pred, None, missed)):
        index = _PolylineIndex(list(ref), step) if len(ref) else None
        for p in src:
            d = _densify(p, step)
            if len(d) < 2:
                continue
            near = index.within(d, tol_vox) if index is not None else np.zeros(len(d), bool)
            for piece, flag in _split_runs(d, near):
                if len(piece) < 2:
                    continue
                if not flag:
                    off.append(piece)
                elif on is not None:
                    on.append(piece)
    return matched, false, missed


def _percentile_limits(a: np.ndarray):
    lo, hi = np.percentile(a, [1, 99])
    return float(lo), float(hi if hi > lo else lo + 1)


def _slab(tomo: np.ndarray, axis: int, frac: float = 0.2) -> np.ndarray:
    """Mean over the central ``frac`` of an axis: one slice is too noisy to see filaments in."""
    n = tomo.shape[axis]
    half = max(1, int(round(0.5 * frac * n)))
    idx = [slice(None)] * 3
    idx[axis] = slice(max(0, n // 2 - half), min(n, n // 2 + half))
    return np.asarray(tomo[tuple(idx)], dtype=np.float32).mean(axis=axis)


def save_filament_view(path, tomo, target, prob, pred, true, tol_vox: float, title: str) -> Path:
    """Writes the view (see the module docstring). ``tomo``, ``target``, ``prob``: ``[D, H, W]`` on one
    grid; ``pred``, ``true``: ``x, y, z`` polylines in voxels of that grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    D, H, W = np.shape(prob)
    matched, false, missed = classify_traces(pred, true, tol_vox)
    cols = ("tomogram (central slab)", "target (max projection)",
            f"prediction (max projection, peak {float(np.max(prob)):.2f})", "traces")
    fig_w = 15.0
    panel_w = fig_w / 4
    fig, axes = plt.subplots(2, 4, figsize=(fig_w, panel_w * (H + D) / W + 1.2), squeeze=False,
                             gridspec_kw={"height_ratios": [H, D]})
    for row, (axis, ix, iy, name) in enumerate(((0, 0, 1, "top, XY"), (1, 0, 2, "side, XZ"))):
        slab = _slab(tomo, axis)
        vmin, vmax = _percentile_limits(slab)
        panels = (slab, np.max(target, axis=axis), np.max(prob, axis=axis), slab)
        for col, image in enumerate(panels):
            ax = axes[row][col]
            is_map = col in (1, 2)
            ax.imshow(image, origin="lower", interpolation="nearest", cmap="Blues" if is_map else "gray",
                      vmin=0.0 if is_map else vmin, vmax=1.0 if is_map else vmax,
                      alpha=0.45 if col == 3 else 1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(cols[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
        ax = axes[row][3]
        for pieces, color, style, width in ((matched, MATCHED, "-", 1.4), (false, FALSE, "-", 1.8),
                                            (missed, MISSED, (0, (3, 2)), 1.8)):
            if pieces:
                ax.add_collection(LineCollection([p[:, [ix, iy]] for p in pieces], colors=color, linestyles=style,
                                                 linewidths=width))
    handles = [Line2D([], [], color=MATCHED, lw=1.8, label="traced, on a true centreline"),
               Line2D([], [], color=FALSE, lw=1.8, label="traced, false"),
               Line2D([], [], color=MISSED, lw=1.8, ls=(0, (3, 2)), label="true, missed")]
    fig.legend(handles=handles, loc="lower right", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_threshold_curve(path, entry: dict, title: str) -> Path:
    """Precision, recall and F-beta against the tracing threshold (``curve`` of a calibration entry)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = np.asarray(entry.get("curve") or np.zeros((0, 4)), dtype=np.float64).reshape(-1, 4)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for col, label, color, style in ((1, "precision", MATCHED, "-"), (2, "recall", MISSED, "--"), (3, "F-beta", "0.15", "-")):
        ax.plot(curve[:, 0], curve[:, col], color=color, ls=style, lw=2, label=label)
    if entry.get("threshold") is not None:
        ax.axvline(entry["threshold"], color="0.6", lw=1)
        ax.annotate(f"chosen {entry['threshold']:.1f}", (entry["threshold"], 0.03), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color="0.3")
    ax.set(xlabel="tracing threshold", ylabel="length-weighted score", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(color="0.92", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def dump_training_images(log_root, out_dir) -> List[Path]:
    """PNG files of the first and last epoch of every image tag in the TensorBoard logs under
    ``log_root`` (``<tag>_epoch<step>.png``), enlarged 3 times. Returns the paths written."""
    import io

    from PIL import Image
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out_dir = Path(out_dir)
    written: List[Path] = []
    for run in sorted({p.parent for p in Path(log_root).rglob("events.out.tfevents*")}):
        ea = EventAccumulator(str(run), size_guidance={"images": 0})
        ea.Reload()
        for tag in ea.Tags().get("images", []):
            events = ea.Images(tag)
            for ev in {id(e): e for e in (events[0], events[-1])}.values():
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', tag)}_epoch{int(ev.step):03d}.png"
                image = Image.open(io.BytesIO(ev.encoded_image_string))
                image.resize((3 * image.width, 3 * image.height), Image.NEAREST).save(path)
                written.append(path)
    return written


if __name__ == "__main__":  # panels of a run, also while it trains: <log root> <out dir>
    import sys

    for written in dump_training_images(sys.argv[1], sys.argv[2]):
        print(written)
