"""Helpers for the Gate 4b tests (no tests here). New public names are imported inside the
test functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from gate4 import g4_helpers as H4


def stem(p) -> str:
    return Path(str(p)).stem


def expected_folds(stems: Iterable[str], folds: int, seed: int = 42) -> Dict[str, int]:
    """Spec rule: distinct stems sorted, permuted with ``numpy.random.default_rng(seed)``,
    stem ``i`` of the permutation goes to fold ``i % folds``."""
    uniq = np.array(sorted(set(stems)))
    perm = np.random.default_rng(seed).permutation(uniq)
    return {str(s): i % folds for i, s in enumerate(perm)}


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------

def edit_ckpt_config(src, dst, edit) -> Path:
    """Copy checkpoint ``src`` to ``dst`` and apply ``edit(config)`` to its pickled MainConfig."""
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    edit(ck["hyper_parameters"]["config"])
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    return dst


def fold_ckpt(src, dst, chunks_dir) -> Path:
    """A copy of ``src`` whose ``train.chunks_dir`` (and ``shared.chunks_dir``) is ``chunks_dir``."""
    def _edit(cfg):
        cfg.train.chunks_dir = Path(chunks_dir)
        cfg.shared.chunks_dir = Path(chunks_dir)
    return edit_ckpt_config(src, dst, _edit)


# --------------------------------------------------------------------------------------
# validation tomograms and fold layouts
# --------------------------------------------------------------------------------------

def blob_val_tomogram(out_dir: Path, name: str, classes, shift: float = 0.0) -> dict:
    """Gate 4 style phantom (10 A/px, 80x192x192; four 100 A blobs of class 1 and one 500 A blob
    of class 2, positions shifted by ``shift`` voxels) written with a Gate 2 style
    ``val_tomograms`` entry. Returns ``dict(entry, coords)``."""
    apix = 10.0
    s = float(shift)
    parts = [(1, 100.0, (40.2, 30.3 + s, 30.6)), (1, 100.0, (39.7, 30.8, 160.4 - s)),
             (1, 100.0, (40.6, 160.1 - s, 30.2 + s)), (1, 100.0, (39.4, 160.6, 160.3)),
             (2, 500.0, (40.2, 96.3 + s, 95.8))]
    vol, coords = H4.blob_tomogram((80, 192, 192), apix, parts)
    out_dir = Path(out_dir)
    tomo = H4.write_mrc(out_dir / f"{name}.mrc", vol, apix)
    diam = {n: d for n, _, d in classes}
    rows = [dict(x=x, y=y, z=z, class_number=c, diameter_ang=diam[c]) for c, xyz in coords.items()
            for x, y, z in xyz]
    csv = out_dir / f"{name}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    entry = dict(tomogram_path=str(tomo), coords_path=str(csv), dataset_id=0, apix_native=apix,
                 class_available=[True] * len(classes))
    return dict(entry=entry, coords=coords, path=Path(tomo))


def write_fold_layout(chunks_dir: Path, classes, val_entries: List[dict], folds: int, fold: int) -> Path:
    """A pipeline-v2 ``chunk_layout.json`` with the class table, ``val_tomograms`` and ``cv``."""
    from tomocpt.labels.multiclass import build_class_table, class_table_to_json

    table = build_class_table({n: d for n, _, d in classes}, {n: name for n, name, _ in classes})
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    layout = {"layout": 2, "dir_names": {}, "split": "tomos", "pipeline_version": 2,
              "class_table": class_table_to_json(table), "val_tomograms": list(val_entries),
              "cv": {"folds": int(folds), "fold": int(fold)}}
    p = chunks_dir / "chunk_layout.json"
    p.write_text(json.dumps(layout, indent=1))
    return p


def write_ensemble(path, ckpts: Sequence, thresholds: Optional[Dict[int, Optional[float]]] = None) -> Path:
    """An ensemble file in the spec format (absolute checkpoint paths)."""
    classes = {str(c): {"threshold": t, "fbeta": 1.0, "n_true": 1} for c, t in (thresholds or {}).items()}
    obj = {"kind": "tomocpt_ensemble", "version": 1,
           "checkpoints": [str(Path(c).resolve()) for c in ckpts],
           "calibration": {"version": 1, "beta": 4.0, "grid": [round(0.05 * k, 2) for k in range(1, 20)],
                           "classes": classes, "oof_tomograms": []}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
    return path


def entry_of(classes_dict, c):
    return classes_dict[c] if c in classes_dict else classes_dict[str(c)]


def class_picks(pred_dir, name, coord_stem="tomocpt_coords") -> np.ndarray:
    """Rows ``[x, y, z, score]`` of ``<coord_stem>_<name>.star`` (empty if the file is absent)."""
    p = Path(pred_dir) / f"{coord_stem}_{name}.star"
    if not p.is_file():
        return np.zeros((0, 4))
    parts = list(H4.picks_by_class(p).values())
    return np.concatenate(parts) if parts else np.zeros((0, 4))


def sort_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
