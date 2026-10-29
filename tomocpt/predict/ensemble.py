"""k-fold ensembles of pipeline-v2 pickers (Gate 4b).

``build_ensemble`` pools the out-of-fold (OOF) picks of fold checkpoints (each on the
validation tomograms of its own ``chunk_layout.json``), calibrates the per-class thresholds
once on the pool and writes an ensemble file (JSON). ``infer.weights`` may name that file;
``EnsembleModel`` then averages the fold models' per-class maps.

Usage::

    python -m tomocpt.predict.ensemble fold0.ckpt fold1.ckpt ... --out ensemble.json [--config_file config.yaml]
"""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

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


def _load_member(ckpt, device):
    from tomocpt.networks.inference import InferenceModel

    model = InferenceModel.load_from_checkpoint(str(ckpt), map_location=device, strict=True).eval()
    if not model.is_pipeline_v2():
        raise ValueError(f"{ckpt} is not a pipeline-v2 checkpoint; ensembles need pipeline v2")
    return model


def load_ensemble(path, device=torch.device("cpu")) -> EnsembleModel:
    """The ``EnsembleModel`` of an ensemble file, members loaded on ``device``."""
    record = read_ensemble(path)
    members = [_load_member(c, device) for c in record["checkpoints"]]
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
    from tomocpt.predict.calibration import CALIBRATION_VERSION, calibrate_thresholds, collect_validation_picks

    cfg = copy.deepcopy(infer_cfg) if infer_cfg is not None else InferConfig()
    cfg.use_cuda = bool(cfg.use_cuda) and torch.cuda.is_available()
    cfg.save_prediction_confidence_map = False
    cfg.invert_contrast = False  # validation tomograms are read as the chunking read them
    cfg.masks_dir = None
    cfg.classes = None
    device = torch.device("cuda" if cfg.use_cuda else "cpu")

    checkpoints = [str(Path(c).resolve()) for c in checkpoints]
    picks, truths, radii, grid, seen, signature = {}, {}, {}, None, {}, None
    for ckpt in checkpoints:
        model = _load_member(ckpt, device)
        sig = _signature(model)
        if signature is not None and sig != signature:
            raise ValueError(f"{ckpt} differs from {checkpoints[0]} in class table or scale settings")
        signature = sig
        val = _val_tomograms(ckpt)
        for entry in val:
            stem = Path(str(entry["tomogram_path"])).stem
            if stem in seen:
                raise ValueError(f"validation tomogram {stem} is in the folds of {seen[stem]} and {ckpt}; "
                                 "out-of-fold calibration needs disjoint folds")
            seen[stem] = ckpt
        logger.info(f"Out-of-fold picks of {ckpt} on {len(val)} tomogram(s)...")
        p, t, r, grid = collect_validation_picks(model, val, cfg, device)
        for number in p:
            picks.setdefault(number, []).extend(p[number])
            truths.setdefault(number, []).extend(t[number])
        radii.update(r)
        del model

    classes = calibrate_thresholds(picks, truths, radii, beta=cfg.calibration_beta,
                                   min_particles=cfg.calibration_min_particles, grid=grid)
    record = {
        "kind": ENSEMBLE_KIND,
        "version": ENSEMBLE_VERSION,
        "checkpoints": checkpoints,
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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, out_path)
    logger.info(f"Ensemble of {len(checkpoints)} model(s) written to {out_path}")
    return record


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
