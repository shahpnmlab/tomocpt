from typing import List, Optional
import os
import os.path as osp
import shutil
import sys
import psutil

import pytorch_lightning as pl
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger

from tomocpt.dataManager.chunking import do_chunking, get_chunking_name_done, get_chunk_layout_path
from tomocpt.utils import read_particles_csvs, accelerator_selector
from tomocpt.defaultConfigs.train_config import TrainingModes
from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.networks.distillationModel import DistillationPickingModel
from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
from tomocpt.training.callbacks import LRVerificationCallback
from tomocpt.logger import get_logger
from tomocpt.mainConfig import mainConfig, _mainConfigNoChanges
from tomocpt.configManager.configManager import (
    get_cli_snapshot,
    backfill_missing_fields,
    safe_getattr,
    update_config_with_changed_values,
)

logging = get_logger()


def _prepare_data_if_needed(config: DictConfig):
    """Checks for chunked data and runs preprocessing if it doesn't exist.

    Existing chunks must match ``train.data_pipeline_version`` (the manifest's
    ``pipeline_version``, missing = 1), else ValueError("chunk format ..."). Pipeline v2
    chunks must also record the class table of ``prepData`` (``class_table_from_config``).
    """
    from tomocpt.dataManager.metadata import layout_pipeline_version, read_chunk_layout

    logging.info("Checking for prepared data...")
    resolve_sampling(config)
    resolve_fourier_augmentation(config)
    if config.mode == TrainingModes.selfSupervised:
        if _ssl_version(config) == 2:
            _prepare_ssl_v2_data_if_needed(config)
            return
        layout = read_chunk_layout(config.chunks_dir)
        if Path(get_chunking_name_done(config.chunks_dir, require_labels=False)).is_file() \
                and layout and layout.get("kind") == "ssl":
            raise ValueError(
                f"chunk format mismatch: {config.chunks_dir} holds SSL v2 source chunks but "
                "train.ssl.version=1. Use another chunks_dir, or set train.ssl.version=2."
            )
    require_labels = config.mode == TrainingModes.picking
    pipeline_version = int(getattr(config, "data_pipeline_version", 1) or 1)
    cv_folds = int(safe_getattr(config, "cv_folds", 0) or 0)
    cv = (cv_folds, int(safe_getattr(config, "cv_fold", 0) or 0)) if cv_folds > 0 else None
    if cv is not None and pipeline_version != 2:
        raise ValueError("k-fold training (train.cv_folds > 0) needs train.data_pipeline_version=2")
    val_stems = _requested_val_stems(config)
    chunking_name_done = get_chunking_name_done(
        config.chunks_dir, require_labels=require_labels
    )

    if Path(chunking_name_done).is_file():
        logging.info("Found existing chunked data. Skipping data preparation.")
        layout = read_chunk_layout(config.chunks_dir)
        found = layout_pipeline_version(layout)
        if found != pipeline_version:
            raise ValueError(
                f"chunk format mismatch: {config.chunks_dir} holds pipeline v{found} chunks but "
                f"train.data_pipeline_version={pipeline_version}. Delete {chunking_name_done} to rebuild "
                "the chunks, or set train.data_pipeline_version to match."
            )
        if pipeline_version == 2:
            # The class table at training start must be the one the chunks were built with.
            from tomocpt.labels.multiclass import (class_table_from_config, class_table_from_json,
                                                   class_table_to_json, same_class_table)

            expected = class_table_from_config(mainConfig.prepData)
            recorded = class_table_from_json(layout.get("class_table", []))
            if not same_class_table(expected, recorded):
                raise ValueError(
                    "The class table of prepData differs from the one the chunks were built with: "
                    f"prepData gives {class_table_to_json(expected)}, chunk_layout.json records "
                    f"{class_table_to_json(recorded)}. Fix prepData.class_diameters / class_names, or "
                    f"delete {chunking_name_done} to rebuild the chunks."
                )
            found_storage = str(layout.get("storage", "chunks"))
            if found_storage != _requested_storage(config):
                raise ValueError(
                    f"storage mismatch: {config.chunks_dir} holds "
                    f"{'a tomogram store' if found_storage == 'tomogram' else 'chunk files'} but train.sampling="
                    f"{safe_getattr(config, 'sampling', 'chunks')}. Delete {chunking_name_done} to rebuild, or "
                    "change train.sampling.")
            _check_layout_val_stems(layout, val_stems, chunking_name_done)
            requested = {"folds": cv[0], "fold": cv[1]} if cv is not None else None
            if layout.get("cv") != requested:
                raise RuntimeError(
                    f"k-fold mismatch: {config.chunks_dir} was chunked with cv={layout.get('cv')} but "
                    f"train.cv_folds/cv_fold request cv={requested}. Use one chunks_dir per fold, or "
                    f"delete {chunking_name_done} to rebuild the chunks."
                )
            return
        training_data_dir = (
            config.training_data_dir or mainConfig.prepData.training_data_dir
        )
        if Path(training_data_dir).is_dir() and not Path(get_chunk_layout_path(config.chunks_dir)).is_file():
            tomos_df = read_particles_csvs(training_data_dir)
            stems = tomos_df["tomogram_path"].map(lambda p: Path(str(p)).stem)
            if not stems.is_unique:
                raise RuntimeError(
                    f"chunks_dir was built by a version with the chunk-collision bug (several "
                    f"datasets share tomograms). Delete {chunking_name_done} to rebuild the chunks."
                )
        return

    logging.info("Chunked data not found or incomplete. Starting data preparation...")
    training_data_dir = (
        config.training_data_dir or mainConfig.prepData.training_data_dir
    )

    if not Path(training_data_dir).is_dir():
        raise FileNotFoundError(
            f"Error, training_data_dir {training_data_dir} not found"
        )

    tomos_df = read_particles_csvs(training_data_dir)
    # Every value is passed explicitly (T2.6): chunking workers do not read mainConfig.
    chunk_kwargs = dict(
        alpha_drop_empty=mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES,
        chunk_size=getattr(config, "CHUNK_SIZE", None),
        chunk_stride=getattr(config, "CHUNK_STRIDE", None),
    )
    if pipeline_version == 2:
        from tomocpt.labels.multiclass import class_table_from_config

        if not require_labels:
            raise ValueError("train.data_pipeline_version=2 supports mode=picking only")
        chunk_kwargs.update(
            pipeline_version=2,
            scale_cfg=config.scale,
            class_table=class_table_from_config(mainConfig.prepData),
            seed=getattr(config, "SEED_FOR_TRAIN_VAL_SPLIT", 42),
            label_border_policy=mainConfig.prepData.label_border_policy,
            cv=cv,
            val_stems=val_stems,
            storage=_requested_storage(config),
            volume_dtype=str(safe_getattr(safe_getattr(config, "otf", None), "volume_dtype", "float16")),
        )
    elif _requested_storage(config) != "chunks":
        raise ValueError("train.sampling=on_the_fly needs train.data_pipeline_version=2")
    do_chunking(
        tomos_df,
        chunkedDataDir=config.chunks_dir,
        desired_particle_pixel_size=mainConfig.prepData.desired_particle_pixel_size,
        n_cpus=config.n_cpus_for_preprocessing,
        require_labels=require_labels,
        train_val_level=config.train_on,
        use_gpus=config.use_gpus,
        **chunk_kwargs,
    )
    logging.info("Data preparation complete.")


def _sampling_value(config) -> str:
    v = safe_getattr(config, "sampling", "chunks")
    sampling = str(getattr(v, "value", v) or "chunks")
    if sampling not in ("auto", "chunks", "on_the_fly"):
        raise ValueError(f"train.sampling must be 'auto', 'chunks' or 'on_the_fly', got '{sampling}'")
    return sampling


def resolve_sampling(config) -> str:
    """Resolve ``train.sampling = auto`` (Gate 8c default, human decision 14 Nov 2026) in place.

    ``auto`` becomes ``on_the_fly`` for pipeline v2 picking, unless ``chunks_dir`` already holds
    prepared chunk files (then ``chunks``, so existing chunks are reused, not refused). Pipeline v1
    and self-supervised training always resolve to ``chunks`` (invariant 1). Explicit values are
    returned unchanged. Returns the resolved value.
    """
    sampling = _sampling_value(config)
    if sampling != "auto":
        return sampling
    resolved = "chunks"
    if (config.mode == TrainingModes.picking and int(getattr(config, "data_pipeline_version", 1) or 1) == 2):
        resolved = "on_the_fly"
        chunks_dir = safe_getattr(config, "chunks_dir", None)
        # an existing prepared dir, or the chunk pool of a k-fold CV (training.cv), decides
        for d in ([Path(chunks_dir), Path(chunks_dir) / "pool"] if chunks_dir else []):
            if Path(get_chunking_name_done(str(d), require_labels=True)).is_file():
                from tomocpt.dataManager.metadata import read_chunk_layout

                if str((read_chunk_layout(str(d)) or {}).get("storage") or "chunks") == "chunks":
                    resolved = "chunks"
                break
    config.sampling = resolved
    logging.info(f"train.sampling=auto -> {resolved}")
    return resolved


def resolve_fourier_augmentation(config) -> bool:
    """Turn the Fourier degradation on for pipeline v2 picking (human decision 16 Nov 2026), in place.

    With ``train.augmentation.fourier_auto`` (default True), pipeline v2 picking sets
    ``augmentation.use_mw_aug`` and ``augmentation.use_fourier_aug`` to True, so the checkpoint
    records the switches that ran. Pipeline v1 and self-supervised training are never changed
    (invariant 1), nor is a config with ``fourier_auto=False``. Returns whether the Fourier
    degradation is on after resolution.
    """
    aug = safe_getattr(config, "augmentation", None)
    if aug is None:
        return False
    if (bool(safe_getattr(aug, "fourier_auto", True)) and config.mode == TrainingModes.picking
            and int(getattr(config, "data_pipeline_version", 1) or 1) == 2):
        if not (aug.use_mw_aug and aug.use_fourier_aug):
            logging.info("augmentation.fourier_auto: pipeline v2 picking -> use_mw_aug=True, use_fourier_aug=True")
        aug.use_mw_aug = True
        aug.use_fourier_aug = True
    return bool(aug.use_mw_aug or aug.use_fourier_aug)


def _requested_storage(config) -> str:
    """Storage the train config asks for: ``chunks`` (chunk files) or ``tomogram`` (Gate 8c store)."""
    sampling = _sampling_value(config)
    if sampling == "auto":
        sampling = resolve_sampling(config)
    return "tomogram" if sampling == "on_the_fly" else "chunks"


def _requested_val_stems(config) -> Optional[List[str]]:
    """Sorted stems of ``train.val_stems`` (Gate 6c), or None for the random split or k-fold."""
    text = safe_getattr(config, "val_stems", None)
    if text is None or not str(text).strip():
        return None
    if int(safe_getattr(config, "cv_folds", 0) or 0) > 0:
        raise ValueError("train.val_stems cannot be combined with train.cv_folds")
    if int(getattr(config, "data_pipeline_version", 1) or 1) != 2:
        raise ValueError("train.val_stems needs train.data_pipeline_version=2")
    return sorted({s.strip() for s in str(text).split(",") if s.strip()})


def _check_layout_val_stems(layout: dict, requested: Optional[List[str]], done_file) -> None:
    """Existing chunks must have the requested fixed validation split."""
    found = layout.get("val_stems")
    if found != requested:
        raise RuntimeError(f"Validation split mismatch: the chunks were built with val_stems={found} but "
                           f"train.val_stems requests {requested}. Delete {done_file} to rebuild the chunks.")


def _ssl_version(config) -> int:
    """``train.ssl.version`` of a train config (1 when the field is absent)."""
    ssl = safe_getattr(config, "ssl", None)
    return int(getattr(ssl, "version", 1) or 1) if ssl is not None else 1


def _prepare_ssl_v2_data_if_needed(config) -> None:
    """SSL v2 source chunks (``dataManager.ssl_views``): built if the done file is absent;
    existing chunks must be SSL v2 chunks."""
    from tomocpt.dataManager.metadata import read_chunk_layout
    from tomocpt.dataManager.ssl_views import is_ssl_layout, list_ssl_tomograms, make_ssl_source_chunks

    done = get_chunking_name_done(config.chunks_dir, require_labels=False)
    if Path(done).is_file():
        if not is_ssl_layout(read_chunk_layout(config.chunks_dir)):
            raise ValueError(
                f"chunk format mismatch: {config.chunks_dir} holds no SSL v2 source chunks but "
                f"train.ssl.version=2. Delete {done} to rebuild the chunks, or use another chunks_dir."
            )
        logging.info("Found existing SSL v2 source chunks. Skipping data preparation.")
        return
    ssl = config.ssl
    training_data_dir = config.training_data_dir or mainConfig.prepData.training_data_dir
    if not Path(training_data_dir).is_dir():
        raise FileNotFoundError(f"Error, training_data_dir {training_data_dir} not found")
    logging.info("SSL v2 source chunks not found. Starting data preparation...")
    make_ssl_source_chunks(
        list_ssl_tomograms(training_data_dir), config.chunks_dir,
        apix_ref_min=ssl.apix_ref_min, apix_ref_max=ssl.apix_ref_max, source_chunk=ssl.source_chunk,
        apix_override=safe_getattr(mainConfig.prepData, "apix_override", None),
        seed=getattr(config, "SEED_FOR_TRAIN_VAL_SPLIT", 42), use_gpus=config.use_gpus,
    )
    logging.info("Data preparation complete.")


def _is_ssl_v2_checkpoint(checkpoint_config) -> bool:
    """True for a checkpoint written by SSL v2 (self-supervised mode, ``train.ssl.version = 2``)."""
    return (checkpoint_config.train.mode == TrainingModes.selfSupervised
            and _ssl_version(checkpoint_config.train) == 2)


def _train_continue_kwargs(checkpoint_path: str, is_resuming: bool) -> dict:
    """train_continue controls the differential LR. Fine-tuning sets it to the checkpoint.
    Resuming leaves the value saved in the checkpoint, so a resumed fine-tuning run keeps
    the parameter groups its saved optimizer state was built with."""
    return {} if is_resuming else {"train_continue": checkpoint_path}


# --------------------------------------------------------------------------------------
# Pipeline v2 model creation (Gate 3): architecture checks and warm starts
# --------------------------------------------------------------------------------------

def _model_architecture(cfg) -> tuple:
    """``(scale_conditioning, class table rows or None)`` of a full ``MainConfig``. The class
    table is part of the architecture only for pipeline v2 (a v1 model has one channel)."""
    from tomocpt.labels.multiclass import class_table_from_config, class_table_to_json

    train_cfg = cfg.train
    scale_conditioning = bool(safe_getattr(train_cfg.network, "scale_conditioning", False))
    is_v2 = int(safe_getattr(train_cfg, "data_pipeline_version", 1) or 1) == 2
    table = class_table_to_json(class_table_from_config(cfg.prepData)) if is_v2 else None
    return scale_conditioning, table


def _is_v2_architecture(arch: tuple) -> bool:
    return bool(arch[0]) or arch[1] is not None


def _load_checkpoint_config(checkpoint_path):
    """Raw read of a checkpoint: ``(state_dict, constants_dict, config)``. The config is
    backfilled with current defaults for fields the checkpoint predates."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    config = backfill_missing_fields(hparams["config"], _mainConfigNoChanges)
    return checkpoint["state_dict"], hparams.get("constants_dict"), config


def _effective_config(checkpoint_config):
    """The config a model loaded from this checkpoint gets: the checkpoint's config with the
    values changed on the command line applied (as ``BaseModel.__init__`` does)."""
    return update_config_with_changed_values(
        target=checkpoint_config, originaConfig=_mainConfigNoChanges,
        configAfterCli=get_cli_snapshot() or mainConfig,
    )


def _check_resume_architecture(checkpoint_config) -> None:
    """``resume_from`` never changes the architecture (class table, scale conditioning)."""
    saved = _model_architecture(checkpoint_config)
    requested = _model_architecture(_effective_config(checkpoint_config))
    if saved != requested:
        raise ValueError(
            "resume_from cannot change the model architecture; use fine_tune_from instead. "
            f"Checkpoint (scale_conditioning, class table) = {saved}, requested = {requested}"
        )


def _map_output_channels(state_dict: dict, old_table, new_table) -> dict:
    """Re-index the ``out`` layer of a v2 state dict from ``old_table`` to ``new_table``.

    Classes present in both tables (matched by class number) keep their channel; new
    classes start from the mean of the old channels."""
    import torch

    old_index = {row.number: i for i, row in enumerate(old_table)}
    mapping = {row.name: (old_table[old_index[row.number]].name if row.number in old_index else "mean")
               for row in new_table}
    logging.info(f"Fine-tune class mapping (new class <- old channel): {mapping}")
    state_dict = dict(state_dict)
    for key in [k for k in state_dict if k.startswith("out.")]:
        old = state_dict[key]
        rows = [old[old_index[row.number]] if row.number in old_index else old.mean(0)
                for row in new_table]
        state_dict[key] = torch.stack(rows)
    return state_dict


def _fine_tune_into_v2(checkpoint_path, lr, state_dict, constants_dict, checkpoint_config):
    """Fine-tune into a pipeline-v2 picking model (multi-class and/or scale-conditioned).

    Source: a legacy single-class picker, a v2 picker (possibly with another class table) or
    a self-supervised checkpoint. Builds the new model and loads the weights; the run starts
    fresh (no optimizer state)."""
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.networks.swinunetr import load_legacy_state_dict, load_single_class_into_multiclass

    from_picking = checkpoint_config.train.mode == TrainingModes.picking
    if from_picking:
        # Like load_from_checkpoint: the checkpoint's config with the command-line changes.
        model = BasePickingModel(train_continue=checkpoint_path, lr=lr,
                                 config=checkpoint_config, constants_dict=constants_dict)
    else:
        # Self-supervised source: the picking config comes from the command line, as for v1.
        model = BasePickingModel(train_continue=checkpoint_path, lr=lr)
    network = model.model
    weights = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    if getattr(network, "cond", None) is None and any(k.startswith("cond.") for k in weights):
        # SSL v2 source into an unconditioned picker: its FiLM weights have no place there.
        weights = {k: v for k, v in weights.items() if not k.startswith("cond.")}
        logging.info("Source conditioning weights (cond.*) dropped: the picker is not scale-conditioned.")

    source_is_v2 = from_picking and _model_architecture(checkpoint_config)[1] is not None
    if source_is_v2 and model.is_pipeline_v2():
        weights = _map_output_channels(weights, class_table_from_config(checkpoint_config.prepData),
                                       model.class_table())
        load_legacy_state_dict(network, weights)
    elif network.out.conv.conv.out_channels == 1:
        load_legacy_state_dict(network, weights)
    else:
        # One-channel source (legacy picker or SSL): every class starts as the old output.
        load_single_class_into_multiclass(network, weights)
    n_new = sum(1 for k in network.state_dict() if k.startswith("cond.") and "model." + k not in state_dict)
    logging.info(
        f"Fine-tuning {model.model_name} from {checkpoint_path}: new run, no optimizer state; "
        f"{n_new} new conditioning tensors start at zero."
    )
    return model


def _init_contrastive_head_from_ssl(model, state_dict) -> bool:
    """Gate 7: copy an SSL v2 checkpoint's dense head (``dense_head.*``) into the picker's
    contrastive head when ``train.discriminative.init_from_ssl`` and the shapes match."""
    head = getattr(model, "contrastive_head", None)
    if head is None or not bool(model._disc("init_from_ssl", True)):
        return False
    src = {k[len("dense_head."):]: v for k, v in state_dict.items() if k.startswith("dense_head.")}
    own = head.state_dict()
    if not src or set(src) != set(own) or any(tuple(src[k].shape) != tuple(own[k].shape) for k in own):
        logging.info("Contrastive head starts fresh: the source checkpoint has no matching SSL dense head.")
        return False
    head.load_state_dict(src)
    logging.info("Contrastive head initialised from the SSL v2 dense head.")
    return True


def _create_model(config: DictConfig):
    """Factory function to create the appropriate model based on the config."""
    from tomocpt.training.hparam_scaling import config_lr

    lr = config_lr(config)  # Gate 8g: = optimizer.lr unless lr_scaling is on
    model = None
    checkpointer = ModelCheckpoint(monitor="val_loss", filename="weights", verbose=True)

    checkpoint_path = config.resume_from or config.fine_tune_from
    is_resuming = config.resume_from is not None

    # This is the path that will be passed to trainer.fit(), telling it to resume
    # from a checkpoint. It should only be set when explicitly resuming.
    resume_from_ckpt = config.resume_from

    if config.mode == TrainingModes.picking:
        # Logic for picking mode: distillation, fine-tuning, or from scratch
        if config.use_distillation:
            if not config.fine_tune_from:
                logging.warning(
                    "use_distillation=True but no --fine-tune-from path provided. "
                    "Distillation requires a teacher model. Falling back to standard training."
                )
            else:
                logging.info(
                    f"Initializing Distillation model for fine-tuning from teacher: {config.fine_tune_from}"
                )
                distill_kwargs = {
                    "teacher_checkpoint_path": config.fine_tune_from,
                    "distill_weight": config.distill_weight,
                    "feature_distill_weight": config.feature_distill_weight,
                    "temperature": config.temperature,
                    "lr": lr,
                }
                model = DistillationPickingModel(**distill_kwargs)
                # Start a new training run, don't resume optimizer state etc. from teacher
                return model, None, checkpointer

        use_v2_fine_tune = False
        if checkpoint_path:
            state_dict, constants_dict, checkpoint_config = _load_checkpoint_config(checkpoint_path)
            from_picking = checkpoint_config.train.mode == TrainingModes.picking
            if is_resuming and from_picking:
                _check_resume_architecture(checkpoint_config)
            # The picking model a fine-tune builds: from the checkpoint's config with the
            # command-line changes, or from the command line for a self-supervised source.
            target_config = _effective_config(checkpoint_config) if from_picking else mainConfig
            use_v2_fine_tune = not is_resuming and (
                _is_v2_architecture(_model_architecture(target_config))
                or (from_picking and _is_v2_architecture(_model_architecture(checkpoint_config)))
                # Self-supervised branch: an SSL v2 network is scale-conditioned, so it always
                # goes through the explicit weight transfer (heads dropped, cond kept or dropped).
                or _is_ssl_v2_checkpoint(checkpoint_config)
            )

        if checkpoint_path and use_v2_fine_tune:
            model = _fine_tune_into_v2(checkpoint_path, lr, state_dict, constants_dict, checkpoint_config)
            resume_from_ckpt = None
            if checkpoint_config.train.mode != TrainingModes.picking:
                _init_contrastive_head_from_ssl(model, state_dict)
        elif checkpoint_path:
            logging.info(f"Loading model from checkpoint: {checkpoint_path}")
            try:
                model = BasePickingModel.load_from_checkpoint(
                    checkpoint_path, lr=lr, **_train_continue_kwargs(checkpoint_path, is_resuming)
                )
            except RuntimeError:
                logging.info(
                    "Could not load as BasePickingModel, trying as SelfSupervisedModel for transfer learning."
                )
                # When transfer learning, it's always a fine-tune, not a resume.
                pretrained_model = SelfSupervisedModel.load_from_checkpoint(
                    checkpoint_path
                )
                model = BasePickingModel(
                    train_continue=checkpoint_path, lr=lr, model=pretrained_model
                )
                resume_from_ckpt = (
                    None  # Can't resume state from a different model type
                )
        else:
            logging.info("Initializing new BasePickingModel from scratch.")
            model = BasePickingModel(train_continue=None, lr=lr)

    elif config.mode == TrainingModes.selfSupervised:
        # Logic for self-supervised mode
        checkpointer = ModelCheckpoint(
            monitor="val_loss",
            filename=f"weights_{config.mode}_{config.network.model_type}",
            verbose=True,
        )
        if checkpoint_path:
            logging.info(f"Loading self-supervised model from: {checkpoint_path}")
            model = SelfSupervisedModel.load_from_checkpoint(
                checkpoint_path, lr=lr, **_train_continue_kwargs(checkpoint_path, is_resuming)
            )
        else:
            logging.info("Initializing new SelfSupervisedModel from scratch.")
            model = SelfSupervisedModel(train_continue=None, lr=lr)
    else:
        raise ValueError(f"Error, training mode '{config.mode}' is not valid")

    return model, resume_from_ckpt, checkpointer


def _setup_callbacks(
    config: DictConfig, model: pl.LightningModule, checkpointer: ModelCheckpoint
):
    """Creates and returns a list of PyTorch Lightning callbacks.

    Early stopping uses ``train.early_stop_min_delta`` / ``early_stop_patience`` (legacy 0.005 /
    10), or the relative ``train.early_stop_rel_delta`` when set (Gate 4h). With ``train.schedule_free`` there is no SWA: the optimiser averages the weights itself
    and SWA would also reset the learning rate (Gate 4e)."""
    rel_delta = safe_getattr(config, "early_stop_rel_delta", None)
    es_patience = int(safe_getattr(config, "early_stop_patience", 10))
    if rel_delta is None:
        early_stopping = EarlyStopping(
            monitor="val_loss_smooth",
            patience=es_patience,
            min_delta=float(safe_getattr(config, "early_stop_min_delta", 0.005)),
            verbose=True,
        )
    else:
        # Gate 4h: relative improvement threshold
        from tomocpt.training.callbacks import RelativeEarlyStopping

        early_stopping = RelativeEarlyStopping(
            monitor="val_loss_smooth", rel_delta=float(rel_delta), patience=es_patience, verbose=True,
        )
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        early_stopping,
        checkpointer,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    schedule_free = bool(safe_getattr(config, "schedule_free", False))
    averaging = str(safe_getattr(config, "weight_averaging", "swa"))
    if averaging not in ("swa", "ema", "swa_plateau", "none"):
        raise ValueError(f"train.weight_averaging must be swa, ema, swa_plateau or none, got {averaging!r}")
    if schedule_free and averaging in ("ema", "swa_plateau"):
        raise ValueError(f"train.weight_averaging={averaging} cannot be combined with train.schedule_free "
                         "(the schedule-free optimiser already averages the weights)")
    if averaging == "swa" and not schedule_free:
        callbacks.append(StochasticWeightAveraging(
            annealing_epochs=config.COSINE_LR_SCHEDULE_N_EPOCHS,
            swa_lrs=0.1 * model.lr,
        ))
    elif averaging in ("ema", "swa_plateau"):
        # Gate 4f: average validated and saved, no fixed start epoch
        from tomocpt.training.averaging import WeightAveraging
        from tomocpt.training.hparam_scaling import config_ema_decay

        callbacks.append(WeightAveraging(
            averaging, ema_decay=config_ema_decay(config, float(safe_getattr(config, "ema_decay", 0.999))),
            plateau_patience=int(safe_getattr(config, "swa_plateau_patience", 5)),
            plateau_rel_delta=float(safe_getattr(config, "swa_plateau_rel_delta", 1e-3))))
    callbacks.append(LRVerificationCallback())
    if config.mode == TrainingModes.picking and config.use_gpus:
        # Gate 8 (human decision 14 Nov 2026): Swin checkpointing off when memory allows
        from tomocpt.training.swin_memory import SwinCheckpointAuto

        prec = config.network.TORCH_FLOAT_PRECISION
        callbacks.append(SwinCheckpointAuto(
            feature_size=int(safe_getattr(config.network.SwinUNETR, "feature_size", 36)),
            chunk_size=int(safe_getattr(config, "CHUNK_SIZE", 64)),
            precision=str(getattr(prec, "value", prec)), batch_size=int(config.batch_size)))
    n_snap = int(safe_getattr(config, "snapshot_chunks", 0))
    if n_snap > 0 and int(safe_getattr(config, "data_pipeline_version", 1)) == 2:
        from tomocpt.training.snapshots import VolumeSnapshots

        callbacks.append(VolumeSnapshots(n_snap))
    return callbacks


def _setup_trainer(config: DictConfig, callbacks: list, logger: TensorBoardLogger):
    """Configures and returns the PyTorch Lightning Trainer."""
    accel, dev_count = accelerator_selector(
        use_cuda=config.use_gpus, n_cpus=config.N_CPUS_IF_NO_GPU
    )
    return pl.Trainer(
        default_root_dir=config.model_dir,
        devices=f"{dev_count}",
        accelerator="auto" if config.use_gpus else accel,
        max_epochs=config.n_epochs,
        callbacks=callbacks,
        logger=logger,
        overfit_batches=config.OVERFIT_N_BATCHES or 0,
        strategy="ddp_find_unused_parameters_false" if accel == "gpu" else "auto",
        precision=config.network.TORCH_FLOAT_PRECISION.value,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_model_summary=False,
    )


def _copy_code_for_reproducibility(logdir: str):
    """Copies the project source code to the log directory for reproducibility."""
    copy_code_dir = osp.join(logdir, "code", "tomocpt")
    os.makedirs(copy_code_dir, exist_ok=True)

    module_path = osp.abspath(sys.modules[__name__].__file__)
    root_path = osp.dirname(osp.dirname(osp.dirname(module_path)))

    for root, _, files in os.walk(osp.join(root_path, "tomocpt")):
        for file in files:
            if file.endswith((".py", ".yaml")):
                source_file = osp.join(root, file)
                relative_path = osp.relpath(source_file, root_path)
                target_file = osp.join(logdir, "code", relative_path)
                os.makedirs(osp.dirname(target_file), exist_ok=True)
                shutil.copy2(source_file, target_file)

    command_history = f"'''.join(sys.argv)'''"
    try:
        current_process = psutil.Process()
        parent_process = current_process.parent()
        parent_command = " ".join(
            [
                "'" + x + "'" if x.startswith("{") else x
                for x in parent_process.cmdline()
            ]
        )
        command_history += f"Parent command: {parent_command}"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    with open(osp.join(logdir, "command.txt"), "w") as f:
        f.write(command_history)
