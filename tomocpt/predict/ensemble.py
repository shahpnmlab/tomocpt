"""k-fold ensembles of pipeline-v2 pickers (Gate 4b).

``build_ensemble`` pools the out-of-fold (OOF) picks of fold checkpoints (each on the
validation tomograms of its own ``chunk_layout.json``), calibrates the per-class thresholds
once on the pool and writes an ensemble file (JSON). ``infer.weights`` may name that file;
``EnsembleModel`` then averages the fold models' per-class maps.

Usage::

    python -m tomocpt.predict.ensemble fold0.ckpt fold1.ckpt ... --out ensemble.json [--config_file config.yaml]
"""
import argparse
import contextlib
import copy
import dataclasses
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import yaml
from torch import nn

from tomocpt.configManager.configManager import safe_getattr
from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.logger import get_logger

logger = get_logger()

ENSEMBLE_KIND = "tomocpt_ensemble"
ENSEMBLE_VERSION = 1


def is_ensemble_file(path) -> bool:
    """True if ``path`` is an ensemble file (JSON with ``"kind": "tomocpt_ensemble"``)."""
    if path is None or Path(str(path)).suffix.lower() != ".json" or not Path(str(path)).is_file():
        return False
    try:
        with open(path) as fh:
            return json.load(fh).get("kind") == ENSEMBLE_KIND
    except (OSError, ValueError, AttributeError):
        return False


def read_ensemble(path) -> dict:
    with open(path) as fh:
        record = json.load(fh)
    if record.get("kind") != ENSEMBLE_KIND:
        raise ValueError(f"{path} is not a TomoCPT ensemble file")
    return record


def _signature(model) -> tuple:
    """Class table and scale settings, which every member must share."""
    from tomocpt.labels.multiclass import class_table_to_json
    from tomocpt.predict.multiclass import scale_config_of

    scale = scale_config_of(model)
    scale = dataclasses.asdict(scale) if dataclasses.is_dataclass(scale) else vars(scale)
    return json.dumps(class_table_to_json(model.class_table()), sort_keys=True), json.dumps(scale, sort_keys=True, default=str)


class EnsembleModel(nn.Module):
    """Mean of the members' ``predict_step``. Every other attribute (class table, patch size,
    checkpoint config) is the first member's."""

    def __init__(self, members: Sequence[nn.Module], calibration: Optional[dict] = None):
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = nn.ModuleList(members)
        self.calibration = calibration
        if len({_signature(m) for m in self.members}) > 1:
            raise ValueError("ensemble members differ in class table or scale settings")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["members"][0], name)

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0, apix=None, classes=None):
        total = None
        for m in self.members:
            p = m.predict_step(batch, batch_idx, dataloader_idx, apix=apix, classes=classes).float()
            total = p if total is None else total + p
        return total / len(self.members)


@contextlib.contextmanager
def isolated_checkpoint_load():
    """Checkpoint loads inside this block keep their own settings (Gate 4c, TODO 3a.4).

    ``BaseModel.__init__`` applies every ``train`` value that differs from the defaults in the CLI
    snapshot (or, without one, in the live ``mainConfig``) to the checkpoint config, and writes
    the result into ``mainConfig.train``; a second load then inherits the first one's settings
    (CLAUDE.md 1.10). Here the snapshot is a copy of the defaults (no changes to apply), and the
    previous snapshot and the content of ``mainConfig.train`` are restored afterwards."""
    from tomocpt.configManager.configManager import _restore_cli_snapshot, get_cli_snapshot, set_cli_snapshot
    from tomocpt.mainConfig import _mainConfigNoChanges, mainConfig

    previous = get_cli_snapshot()
    saved_train = copy.deepcopy(vars(mainConfig.train))
    set_cli_snapshot(_mainConfigNoChanges)
    try:
        yield
    finally:
        _restore_cli_snapshot(previous)
        state = vars(mainConfig.train)
        state.clear()
        state.update(saved_train)


def load_member_isolated(ckpt, device):
    """A pipeline-v2 checkpoint loaded for inference with its own settings, whatever was loaded
    before in this process (see :func:`isolated_checkpoint_load`)."""
    from tomocpt.networks.inference import InferenceModel

    with isolated_checkpoint_load():
        model = InferenceModel.load_from_checkpoint(str(ckpt), map_location=device, strict=True).eval()
    if not model.is_pipeline_v2():
        raise ValueError(f"{ckpt} is not a pipeline-v2 checkpoint; ensembles need pipeline v2")
    return model


_load_member = load_member_isolated


def load_ensemble(path, device=torch.device("cpu")) -> EnsembleModel:
    """The ``EnsembleModel`` of an ensemble file, members loaded on ``device``."""
    record = read_ensemble(path)
    members = [load_member_isolated(c, device) for c in record["checkpoints"]]
    return EnsembleModel(members, calibration=record.get("calibration")).eval()


def _val_tomograms(ckpt) -> List[dict]:
    """The validation tomograms of the chunks a checkpoint was trained on. The config is read
    from the raw checkpoint: a loaded model's ``train`` section can carry an earlier load's
    values when no CLI snapshot exists (CLAUDE.md Section 1.10)."""
    from tomocpt.dataManager.metadata import read_chunk_layout

    config = torch.load(str(ckpt), map_location="cpu", weights_only=False)["hyper_parameters"]["config"]
    chunks_dir = safe_getattr(config.train, "chunks_dir", None) or safe_getattr(config.shared, "chunks_dir", None)
    layout = read_chunk_layout(chunks_dir) if chunks_dir else None
    val = (layout or {}).get("val_tomograms") or []
    if not val:
        raise ValueError(f"{ckpt}: no val_tomograms in the chunk_layout.json of {chunks_dir}")
    return val


def build_ensemble(checkpoints: Sequence, out_path, infer_cfg: Optional[InferConfig] = None) -> dict:
    """Calibrates fold checkpoints on their pooled out-of-fold picks and writes the ensemble
    file. Returns the record."""
    from tomocpt.predict import calibration as C
    from tomocpt.predict.calibration import CALIBRATION_VERSION, calibrate_thresholds

    cfg = copy.deepcopy(infer_cfg) if infer_cfg is not None else InferConfig()
    cfg.use_cuda = bool(cfg.use_cuda) and torch.cuda.is_available()
    cfg.save_prediction_confidence_map = False
    cfg.invert_contrast = False  # validation tomograms are read as the chunking read them
    cfg.masks_dir = None
    cfg.classes = None
    device = torch.device("cuda" if cfg.use_cuda else "cpu")

    checkpoints = [str(Path(c).resolve()) for c in checkpoints]
    fingerprint = C.picks_fingerprint(cfg)
    picks, truths, radii, grid, seen, signature = {}, {}, {}, None, {}, None
    profiles, tomograms, tomogram_fold, table = {}, [], [], None
    for k, ckpt in enumerate(checkpoints):
        model = load_member_isolated(ckpt, device)
        sig = _signature(model)
        if signature is not None and sig != signature:
            raise ValueError(f"{ckpt} differs from {checkpoints[0]} in class table or scale settings")
        signature = sig
        table = model.class_table()
        val = _val_tomograms(ckpt)
        for entry in val:
            stem = Path(str(entry["tomogram_path"])).stem
            if stem in seen:
                raise ValueError(f"validation tomogram {stem} is in the folds of {seen[stem]} and {ckpt}; "
                                 "out-of-fold calibration needs disjoint folds")
            seen[stem] = ckpt
        saved = _reusable_val_picks(ckpt, val, fingerprint)
        if saved is not None:
            logger.info(f"Out-of-fold picks of {ckpt}: reusing {C.val_picks_path(ckpt)}")
            p, t, r, grid, prof = saved["picks"], saved["truths"], saved["radii"], saved["grid"], saved["profiles"]
        else:
            logger.info(f"Out-of-fold picks of {ckpt} on {len(val)} tomogram(s)...")
            prof = {}
            p, t, r, grid = C.collect_validation_picks(model, val, cfg, device, profiles=prof)
        for number in p:
            picks.setdefault(number, []).extend(p[number])
            truths.setdefault(number, []).extend(t[number])
        for number, acc in prof.items():
            profiles[number] = acc if number not in profiles else C._add_profiles(profiles[number], acc)
        radii.update(r)
        tomograms.extend(str(e["tomogram_path"]) for e in val)
        tomogram_fold.extend([k] * len(val))
        del model

    classes = calibrate_thresholds(picks, truths, radii, beta=cfg.calibration_beta,
                                   min_particles=cfg.calibration_min_particles, grid=grid)
    out_path = Path(out_path)
    oof_name = "oof_picks.npz"
    arrays = C.pack_picks(table, picks, truths, radii, profiles, tomogram_fold=tomogram_fold)
    C._savez_atomic(out_path.parent / oof_name, tomograms=np.array(tomograms, dtype=str),
                    tomogram_fold=np.array(tomogram_fold, dtype=np.int64),
                    class_names=np.array([str(r.name) for r in table], dtype=str),
                    grid=np.asarray(list(grid), dtype=np.float64), fingerprint=np.array(fingerprint), **arrays)
    record = {
        "kind": ENSEMBLE_KIND,
        "version": ENSEMBLE_VERSION,
        "checkpoints": checkpoints,
        "oof_picks": oof_name,
        "calibration": {
            "version": CALIBRATION_VERSION,
            "beta": float(cfg.calibration_beta),
            "grid": list(grid),
            "classes": {str(k): v for k, v in classes.items()},
            "oof_tomograms": sorted(seen),
        },
    }
    for number, e in classes.items():
        thr = "None (falls back)" if e["threshold"] is None else f"{e['threshold']:.2f}"
        logger.info(f"Ensemble calibration, class {number}: threshold {thr}, F-beta {e['fbeta']:.3f}, "
                    f"{e['n_true']} true particle(s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, out_path)
    logger.info(f"Ensemble of {len(checkpoints)} model(s) written to {out_path}")
    return record


def _reusable_val_picks(ckpt, val: List[dict], fingerprint: str) -> Optional[dict]:
    """The saved validation picks of ``ckpt`` if they were made on the same tomograms with the
    same pick settings, else None."""
    from tomocpt.predict import calibration as C

    path = C.val_picks_path(ckpt)
    if not path.is_file():
        return None
    try:
        saved = C.read_val_picks(path)
    except Exception as e:
        logger.warning(f"Cannot read {path} ({e}); predicting again.")
        return None
    same_tomos = [str(Path(t).resolve()) for t in saved["tomograms"]] == \
        [str(Path(str(e["tomogram_path"])).resolve()) for e in val]
    if not same_tomos or saved["fingerprint"] != fingerprint:
        logger.info(f"{path} was made on other tomograms or with other pick settings; predicting again.")
        return None
    return saved


def read_oof_picks(ensemble_path) -> dict:
    """The pooled out-of-fold picks of an ensemble file: ``tomograms``, ``tomogram_fold``,
    ``picks``, ``truths``, ``radii`` (``calibrate_thresholds`` formats, tomograms in
    ``tomograms`` order), ``available``, ``class_names`` (``{number: name}``), ``profiles``,
    ``grid``, ``fingerprint``."""
    from tomocpt.predict import calibration as C

    ensemble_path = Path(ensemble_path)
    record = read_ensemble(ensemble_path)
    name = record.get("oof_picks")
    if not name:
        raise ValueError(f"{ensemble_path} records no out-of-fold picks (built before Gate 4c)")
    with np.load(str(ensemble_path.parent / name), allow_pickle=False) as z:
        out = C.unpack_picks(z, fold_column=True)
        out["tomograms"] = [str(t) for t in z["tomograms"]]
        out["tomogram_fold"] = np.asarray(z["tomogram_fold"], dtype=np.int64)
        out["class_names"] = {int(n): str(s) for n, s in zip(z["class_numbers"], z["class_names"])}
        out["grid"] = [float(g) for g in z["grid"]] if "grid" in z else None
        out["fingerprint"] = str(z["fingerprint"])
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a k-fold ensemble with out-of-fold threshold calibration")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config_file", type=Path, default=None,
                        help="YAML whose `infer` section sets calibration_beta, calibration_min_particles, cross_class_nms, ...")
    args = parser.parse_args(argv)
    cfg = InferConfig()
    if args.config_file is not None:
        with open(args.config_file) as fh:
            section = (yaml.safe_load(fh) or {}).get("infer") or {}
        for key, value in section.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    build_ensemble(args.checkpoints, args.out, cfg)


if __name__ == "__main__":
    main()
