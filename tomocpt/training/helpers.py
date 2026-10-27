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
from tomocpt.mainConfig import mainConfig

logging = get_logger()


def _prepare_data_if_needed(config: DictConfig):
    """Checks for chunked data and runs preprocessing if it doesn't exist.

    Existing chunks must match ``train.data_pipeline_version`` (the manifest's
    ``pipeline_version``, missing = 1), else ValueError("chunk format ..."). Pipeline v2
    chunks must also record the class table of ``prepData`` (``class_table_from_config``).
    """
    from tomocpt.dataManager.metadata import layout_pipeline_version, read_chunk_layout

    logging.info("Checking for prepared data...")
    require_labels = config.mode == TrainingModes.picking
    pipeline_version = int(getattr(config, "data_pipeline_version", 1) or 1)
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
        )
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


def _train_continue_kwargs(checkpoint_path: str, is_resuming: bool) -> dict:
    """train_continue controls the differential LR. Fine-tuning sets it to the checkpoint.
    Resuming leaves the value saved in the checkpoint, so a resumed fine-tuning run keeps
    the parameter groups its saved optimizer state was built with."""
    return {} if is_resuming else {"train_continue": checkpoint_path}


def _create_model(config: DictConfig):
    """Factory function to create the appropriate model based on the config."""
    lr = config.optimizer.lr
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

        if checkpoint_path:
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
    """Creates and returns a list of PyTorch Lightning callbacks."""
    return [
        TQDMProgressBar(refresh_rate=10),
        EarlyStopping(
            monitor="val_loss_smooth", patience=10, min_delta=0.005, verbose=True
        ),
        checkpointer,
        LearningRateMonitor(logging_interval="epoch"),
        StochasticWeightAveraging(
            annealing_epochs=config.COSINE_LR_SCHEDULE_N_EPOCHS,
            swa_lrs=0.1 * model.lr,
        ),
        LRVerificationCallback(),
    ]


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
