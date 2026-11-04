"""Helpers for the Gate 4c tests (no tests here). New public names are imported inside the
test functions.

Interface fixed by the TA (spec ``04c_AGENT_CV_ROUTINE.md``, "Public names"):

* ``tomocpt.training.cv.run_cv(config)``: ``config`` is the ``train`` section (as ``train()``
  receives it); the other sections are read from the global ``mainConfig``.
* ``tomocpt.training.cv.prepare_cv_chunks(config) -> List[Path]``: builds the pool
  ``<chunks_dir>/pool`` and the fold chunk dirs ``<chunks_dir>/fold{k}``; returns the fold dirs
  in fold order.
* ``tomocpt.training.cv._run_fold_process(cmd: List[str], log_path: Path) -> int``: runs one
  fold process, returns its exit code (tests replace it).
* ``tomocpt.training.cv._build_ensemble_and_report(config, checkpoints: List[Path], cv_dir: Path)``
  (tests may replace it).
* Fold config: ``<cv dir>/folds/fold{k}/config.yaml`` (all sections); the fold command line is
  ``[sys.executable, "-m", "tomocpt.main", "train", "--config_file", <that path>]``.
* ``tomocpt.dataManager.chunking.chunk_pool_v2`` / ``link_fold_chunks``: called through
  ``prepare_cv_chunks`` in the tests.
* ``tomocpt.predict.calibration.collect_validation_picks(model, val, cfg, device, profiles=None)``:
  when ``profiles`` is a dict, it is filled with ``{class number: accumulator}``
  (``peak_profile_accumulate`` format, pooled over the tomograms).
* ``tomocpt.predict.calibration.peak_profile_accumulate(class_map, centres_xyz, radius) -> dict``
  with numpy ``radial_sum, radial_count`` ``[30]`` (bins of 0.05 R from 0 to 1.5 R),
  ``line_sum, line_count`` ``[3, 61]`` (x, y, z; t = -1.5 R .. 1.5 R in steps of 0.05 R),
  ``peak_sum`` (float), ``n`` (int). ``class_map`` is ``[D, H, W]`` (numpy or torch);
  ``centres_xyz`` ``[N, 3]`` in voxels of the map; ``radius`` in voxels.
* ``tomocpt.predict.calibration.val_picks_path(ckpt) -> Path`` (``<ckpt stem>.val_picks.npz``
  next to the checkpoint) and ``read_val_picks(path) -> dict`` with keys ``picks``, ``truths``,
  ``radii``, ``grid`` (the ``calibrate_thresholds`` formats), ``tomograms`` (list of str),
  ``available`` (bool ``[T, C]``), ``fingerprint`` (str), ``profiles`` (``{class number:
  accumulator}``).
* ``tomocpt.predict.ensemble.load_member_isolated(ckpt, device)``;
  ``read_oof_picks(ensemble_path) -> dict`` with ``tomograms`` (list), ``tomogram_fold`` (int
  ``[T]``), ``picks`` / ``truths`` / ``radii`` (``calibrate_thresholds`` formats, tomograms in
  ``tomograms`` order), ``class_names`` (``{number: name}``), ``profiles``.
* ``tomocpt.predict.cv_report.write_cv_report(ensemble_path, out_dir, class_weights=None) -> dict``
  (the content of ``cv_summary.json``); ``peak_profile_stats(acc) -> dict`` with ``r`` ``[30]``,
  ``radial`` ``[30]``, ``t`` ``[61]``, ``line`` ``[3, 61]``, ``peak_height``, ``fwhm`` (units of
  R), ``target_radial``, ``target_line``.
* ``oof_picks.npz`` keys (spec D3): ``tomograms`` str ``[T]``, ``tomogram_fold`` int ``[T]``,
  ``class_numbers`` int ``[C]``, ``class_names`` str ``[C]``, ``radii`` float ``[C]``,
  ``picks`` float ``[P, 7]`` (fold, tomogram index, class number, x, y, z, score),
  ``truths`` float ``[M, 6]`` (fold, tomogram index, class number, x, y, z), ``available`` bool
  ``[T, C]``, ``fingerprint`` str, ``profile_radial_sum`` / ``profile_radial_count`` ``[C, 30]``,
  ``profile_line_sum`` / ``profile_line_count`` ``[C, 3, 61]``, ``profile_peak_sum`` ``[C]``,
  ``profile_n`` ``[C]``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from gate3 import g3_helpers as G3

N_RADIAL = 30
N_LINE = 61


def build_ckpt(root: Path, classes, contrastive_weight: float = 0.0) -> dict:
    """Tiny pipeline-v2 checkpoint (as ``g3_helpers.build_v2_checkpoint``, unconditioned) with
    ``train.discriminative.contrastive_weight`` set; one CPU batch."""
    from tomocpt.networks.pickingModel import BasePickingModel
    from utils.tiny import applied_main_config

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = G3.v2_config(root, classes, cond=False)
    cfg.train.particles.class_head = "softmax"
    cfg.train.discriminative.contrastive_weight = float(contrastive_weight)
    d_vox = [d / 10.0 for _, _, d in classes]
    items = [G3.v2_item(200 + i, d_vox, apix=10.0) for i in range(2)]
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        trainer = G3.cpu_trainer(root / "tb")
        trainer.fit(model, datamodule=G3.datamodule(items))
        path = root / "v2.ckpt"
        trainer.save_checkpoint(path)
    return dict(path=path, classes=tuple(classes), contrastive_weight=float(contrastive_weight))


def target_profile(r_in_R: np.ndarray) -> np.ndarray:
    """CLAUDE.md 1.3 target in units of R (cut at R)."""
    r = np.asarray(r_in_R, float)
    v = 0.5 * np.exp(-r * r / (2 * 0.165 ** 2)) + 0.5 * np.exp(-r * r / (2 * 0.33 ** 2))
    return np.where(r > 1.0, 0.0, v)


def wide_profile(r_in_R: np.ndarray) -> np.ndarray:
    r = np.asarray(r_in_R, float)
    return np.exp(-r * r / (2 * 0.33 ** 2))


def profile_map(shape_zyx, centres_xyz, radius: float, fn=target_profile) -> np.ndarray:
    """A map that is ``fn(distance / radius)`` around each centre (maximum over centres)."""
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=float) for s in shape_zyx], indexing="ij")
    out = np.zeros(shape_zyx, dtype=np.float32)
    for x, y, z in np.asarray(centres_xyz, float):
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / radius
        out = np.maximum(out, fn(d).astype(np.float32))
    return out


def files_under(root: Path) -> Dict[str, Path]:
    """Relative path -> file, following symlinked directories."""
    root = Path(root)
    out = {}
    for d, _, files in os.walk(root, followlinks=True):
        for f in files:
            p = Path(d) / f
            out[str(p.relative_to(root))] = p
    return out


def same_chunk_file(a: Path, b: Path) -> bool:
    if str(a).endswith(".nii.gz"):
        import nibabel as nib

        ia, ib = nib.load(str(a)), nib.load(str(b))
        return np.array_equal(np.asarray(ia.dataobj), np.asarray(ib.dataobj)) and np.allclose(ia.affine, ib.affine)
    return Path(a).read_bytes() == Path(b).read_bytes()


def write_oof(path: Path, tomograms: List[str], tomogram_fold, classes, radii, picks, truths,
              available=None, profiles=None) -> Path:
    """An ``oof_picks.npz`` in the spec format. ``classes`` = ``((number, name), ...)``;
    ``picks`` rows ``(fold, tomo, class, x, y, z, score)``; ``truths`` rows ``(fold, tomo, class,
    x, y, z)``; ``profiles`` = ``{class number: accumulator}``."""
    C = len(classes)
    nums = [int(n) for n, _ in classes]
    prof = dict(profile_radial_sum=np.zeros((C, N_RADIAL)), profile_radial_count=np.zeros((C, N_RADIAL)),
                profile_line_sum=np.zeros((C, 3, N_LINE)), profile_line_count=np.zeros((C, 3, N_LINE)),
                profile_peak_sum=np.zeros(C), profile_n=np.zeros(C, dtype=np.int64))
    for n, acc in (profiles or {}).items():
        i = nums.index(int(n))
        prof["profile_radial_sum"][i] = acc["radial_sum"]
        prof["profile_radial_count"][i] = acc["radial_count"]
        prof["profile_line_sum"][i] = acc["line_sum"]
        prof["profile_line_count"][i] = acc["line_count"]
        prof["profile_peak_sum"][i] = acc["peak_sum"]
        prof["profile_n"][i] = acc["n"]
    T = len(tomograms)
    np.savez(path, tomograms=np.array(tomograms, dtype=str), tomogram_fold=np.asarray(tomogram_fold, np.int64),
             class_numbers=np.array(nums, np.int64), class_names=np.array([n for _, n in classes], dtype=str),
             radii=np.asarray(radii, float), picks=np.asarray(picks, float).reshape(-1, 7),
             truths=np.asarray(truths, float).reshape(-1, 6),
             available=np.ones((T, C), bool) if available is None else np.asarray(available, bool),
             fingerprint=np.array("{}"), **prof)
    return Path(path)


def write_record(path: Path, thresholds: Dict[int, float], beta: float = 4.0, oof: str = "oof_picks.npz") -> Path:
    obj = {"kind": "tomocpt_ensemble", "version": 1, "checkpoints": [],
           "oof_picks": oof,
           "calibration": {"version": 1, "beta": beta, "grid": [round(0.05 * k, 2) for k in range(1, 20)],
                           "classes": {str(c): {"threshold": t, "fbeta": 0.0, "n_true": 0}
                                       for c, t in thresholds.items()},
                           "oof_tomograms": []}}
    Path(path).write_text(json.dumps(obj, indent=1))
    return Path(path)


def state_of(cfg) -> bytes:
    import pickle

    return pickle.dumps(cfg)


def as_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
