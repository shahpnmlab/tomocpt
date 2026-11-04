"""k-fold cross-validation in one ``tomocpt train`` command (Gate 4c, TODO 3a.1).

``train.cv_all_folds = True`` with ``train.cv_folds = K``:

1. chunk every tomogram once into ``<chunks_dir>/pool`` and build ``<chunks_dir>/fold{k}`` from
   it (symlinks, :func:`tomocpt.dataManager.chunking.link_fold_chunks`);
2. train fold ``k`` in its own process (``python -m tomocpt.main train --config_file
   <cv dir>/folds/fold{k}/config.yaml``: the parent's settings with ``cv_fold=k``, the fold's
   chunks and model directories). Each fold is an ordinary single-fold run, including the
   post-fit calibration, which saves the validation picks;
3. build the fold ensemble with out-of-fold calibration (reusing those picks) and write the
   CV report, all in ``<cv dir>`` = ``<model_dir>/<experiment_name>``.

Running the same command again skips finished folds (``done`` marker and checkpoint) and
restarts unfinished ones from scratch; the ensemble and report are rebuilt every time.
"""

import copy
import dataclasses
import enum
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from tomocpt.configManager.configManager import safe_getattr
from tomocpt.logger import get_logger

logging = get_logger()

FOLD_CHECKPOINT = Path("checkpoints") / "weights.ckpt"


def _cv_dir(config) -> Path:
    return Path(config.model_dir) / str(config.experiment_name)


def _validate(config) -> None:
    """Errors of spec D6, raised before any work."""
    from tomocpt.defaultConfigs.train_config import TrainingModes

    K = int(safe_getattr(config, "cv_folds", 0) or 0)
    if K < 2:
        raise ValueError(f"train.cv_all_folds needs train.cv_folds >= 2, got {K}")
    if int(safe_getattr(config, "data_pipeline_version", 1) or 1) != 2:
        raise ValueError("train.cv_all_folds needs train.data_pipeline_version=2")
    if str(getattr(config.mode, "value", config.mode)) != TrainingModes.picking.value:
        raise ValueError("train.cv_all_folds supports mode=picking only")
    if config.resume_from or config.fine_tune_from:
        raise ValueError("train.cv_all_folds cannot be combined with resume_from or fine_tune_from; "
                         "rerun the same command to resume a CV")
    if not config.model_dir or not config.chunks_dir:
        raise ValueError("train.cv_all_folds needs model_dir and chunks_dir")


def _tomos_df(config):
    from tomocpt.mainConfig import mainConfig
    from tomocpt.utils import read_particles_csvs

    tdd = config.training_data_dir or mainConfig.prepData.training_data_dir
    if not tdd or not Path(tdd).is_dir():
        raise FileNotFoundError(f"Error, training_data_dir {tdd} not found")
    return read_particles_csvs(tdd)


def _pool_request(config) -> dict:
    """The settings a pool must have been built with (``chunking.POOL_SETTINGS``) and the
    keyword arguments of ``chunk_pool_v2``."""
    from tomocpt.labels.multiclass import class_table_from_config, class_table_from_json, class_table_to_json
    from tomocpt.mainConfig import mainConfig

    scale = config.scale
    table = class_table_from_config(mainConfig.prepData)
    policy = mainConfig.prepData.label_border_policy
    kwargs = dict(
        n_cpus=config.n_cpus_for_preprocessing, use_gpus=config.use_gpus, scale_cfg=scale,
        class_table=table, seed=int(getattr(config, "SEED_FOR_TRAIN_VAL_SPLIT", 42)),
        alpha_drop_empty=float(mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES),
        chunk_size=int(getattr(config, "CHUNK_SIZE", None) or mainConfig.train.CHUNK_SIZE),
        chunk_stride=int(getattr(config, "CHUNK_STRIDE", None) or mainConfig.train.CHUNK_STRIDE),
        label_border_policy=policy,
    )
    settings = {
        "band": [float(scale.band_min_vox), float(scale.band_max_vox)],
        "resample": str(getattr(scale.resample, "value", scale.resample)),
        "scales_per_level": int(scale.scales_per_level),
        "target_storage": str(getattr(scale, "target_storage", "points")),
        "label_border_policy": str(getattr(policy, "value", policy)),
        "chunk_size": kwargs["chunk_size"], "chunk_stride": kwargs["chunk_stride"], "seed": kwargs["seed"],
        "class_table": class_table_to_json(class_table_from_json(class_table_to_json(table))),
        "alpha_drop_empty": kwargs["alpha_drop_empty"],
    }
    return dict(kwargs=kwargs, settings=settings)


def prepare_cv_chunks(config) -> List[Path]:
    """Builds (or reuses) the chunk pool ``<chunks_dir>/pool`` and the fold chunk dirs
    ``<chunks_dir>/fold{k}``; returns the fold dirs in fold order. A pool built with other
    settings raises ValueError."""
    from tomocpt.dataManager.chunking import (POOL_SETTINGS, chunk_pool_v2, get_chunking_name_done,
                                              link_fold_chunks, split_rows_by_fold)
    from tomocpt.dataManager.metadata import read_chunk_layout

    _validate(config)
    K = int(config.cv_folds)
    tomos_df = _tomos_df(config)
    req = _pool_request(config)
    split_rows_by_fold(tomos_df, K, 0, seed=req["settings"]["seed"])  # K vs number of tomograms

    chunks_dir = Path(config.chunks_dir)
    pool_dir = chunks_dir / "pool"
    pool = read_chunk_layout(pool_dir) if Path(get_chunking_name_done(str(pool_dir), True)).is_file() else None
    if pool is not None:
        if pool.get("kind") != "pool":
            raise ValueError(f"{pool_dir} is not a chunk pool")
        diff = [k for k in POOL_SETTINGS
                if json.dumps(pool.get(k), sort_keys=True) != json.dumps(req["settings"][k], sort_keys=True)]
        if diff:
            raise ValueError(f"The chunk pool {pool_dir} was built with other settings ({', '.join(diff)}): "
                             f"pool {[pool.get(k) for k in diff]}, requested {[req['settings'][k] for k in diff]}. "
                             "Use another chunks_dir, or delete the pool to rebuild it.")
        logging.info(f"Reusing the chunk pool {pool_dir}")
    else:
        logging.info(f"Chunking every tomogram once into the pool {pool_dir}")
        chunk_pool_v2(tomos_df, pool_dir, **req["kwargs"])
    dirs = []
    for k in range(K):
        d = chunks_dir / f"fold{k}"
        link_fold_chunks(pool_dir, d, tomos_df, K, k)
        dirs.append(d)
    return dirs


# ------------------------------------------------------------------------------------------
# fold processes
# ------------------------------------------------------------------------------------------

def _plain(v):
    """A config value as plain YAML data."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def fold_config(config, k: int, fold_chunks: Path) -> dict:
    """The full configuration (all sections) of fold ``k`` as plain data."""
    from tomocpt.mainConfig import mainConfig

    cv_dir = _cv_dir(config)
    out = {}
    for section in ("shared", "prepData", "train", "infer"):
        src = config if section == "train" else getattr(mainConfig, section)
        out[section] = _plain(copy.deepcopy(src))
    overrides = dict(cv_all_folds=False, cv_fold=int(k), cv_folds=int(config.cv_folds),
                     chunks_dir=str(fold_chunks), model_dir=str(cv_dir / "folds"), experiment_name=f"fold{k}",
                     resume_from=None, fine_tune_from=None, launch_tensorboard=False)
    for key, v in overrides.items():
        if key in out["train"]:
            out["train"][key] = v
        if key in out["shared"]:
            out["shared"][key] = v
    return out


def _run_fold_process(cmd: List[str], log_path: Path) -> int:
    """Runs one fold process with stdout and stderr in ``log_path``; returns the exit code."""
    with open(log_path, "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def _build_ensemble_and_report(config, checkpoints: List[Path], cv_dir: Path) -> None:
    """Fold ensemble with out-of-fold calibration (infer section of the run), then the CV report."""
    import torch

    from tomocpt.mainConfig import mainConfig
    from tomocpt.predict.cv_report import write_cv_report
    from tomocpt.predict.ensemble import build_ensemble

    infer_cfg = copy.deepcopy(mainConfig.infer)
    infer_cfg.use_cuda = torch.cuda.is_available()
    ens = Path(cv_dir) / "ensemble.json"
    build_ensemble([str(c) for c in checkpoints], ens, infer_cfg)
    write_cv_report(ens, cv_dir, class_weights=safe_getattr(config, "cv_class_weights", None))


def _check_cv_config(cv_dir: Path, K: int, seed: int) -> None:
    path = cv_dir / "cv_config.json"
    if not path.is_file():
        return
    old = json.loads(path.read_text())
    if int(old.get("folds", K)) != K or int(old.get("seed", seed)) != seed:
        raise ValueError(f"{cv_dir} holds a CV with folds={old.get('folds')}, seed={old.get('seed')}; this run asks "
                         f"for folds={K}, seed={seed}. Use another experiment_name.")


def run_cv(config) -> Path:
    """Runs the whole k-fold CV (module docstring). ``config`` is the ``train`` section; the
    other sections come from the global ``mainConfig``. Returns the ensemble file."""
    import yaml

    from tomocpt.dataManager.metadata import read_chunk_layout

    _validate(config)
    K = int(config.cv_folds)
    seed = int(getattr(config, "SEED_FOR_TRAIN_VAL_SPLIT", 42))
    cv_dir = _cv_dir(config)
    _check_cv_config(cv_dir, K, seed)

    fold_chunks = prepare_cv_chunks(config)
    val_stems = {str(k): sorted(Path(str(e["tomogram_path"])).stem
                                for e in (read_chunk_layout(d) or {}).get("val_tomograms", []))
                 for k, d in enumerate(fold_chunks)}
    cv_dir.mkdir(parents=True, exist_ok=True)
    meta = {"folds": K, "seed": seed, "chunks_dir": str(config.chunks_dir), "val_stems": val_stems}
    (cv_dir / "cv_config.json").write_text(json.dumps(meta, indent=1))

    checkpoints = []
    for k in range(K):
        fold_dir = cv_dir / "folds" / f"fold{k}"
        ckpt = fold_dir / FOLD_CHECKPOINT
        if (fold_dir / "done").is_file() and ckpt.is_file():
            logging.info(f"CV fold {k}/{K}: already trained ({ckpt})")
            checkpoints.append(ckpt)
            continue
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True)
        cfg_path = fold_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(fold_config(config, k, fold_chunks[k]), sort_keys=False))
        log_path = fold_dir / "train.log"
        cmd = [sys.executable, "-m", "tomocpt.main", "train", "--config_file", str(cfg_path)]
        logging.info(f"CV fold {k}/{K}: training (log: {log_path})")
        rc = _run_fold_process(cmd, log_path)
        if rc != 0:
            raise RuntimeError(f"CV fold {k} failed (exit code {rc}); see {log_path}")
        if not ckpt.is_file():
            raise RuntimeError(f"CV fold {k} finished without a checkpoint at {ckpt}; see {log_path}")
        (fold_dir / "done").write_text("done")
        checkpoints.append(ckpt)

    logging.info("CV: building the fold ensemble and the report")
    _build_ensemble_and_report(config, checkpoints, cv_dir)
    return cv_dir / "ensemble.json"
