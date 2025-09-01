import os
import os.path as osp
import shutil
import sys
import psutil
from typing import Optional

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

from tomocpt.dataManager.chunking import do_chunking, get_chunking_name_done
from tomocpt.utils import read_particles_csvs, accelerator_selector
from tomocpt.defaultConfigs.train_config import TrainingModes
from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.networks.distillationModel import DistillationPickingModel
from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
from tomocpt.training.callbacks import LRVerificationCallback
from tomocpt.logger import get_logger

logging = get_logger()


def _prepare_data_if_needed(config: DictConfig):
    """Checks for chunked data and runs preprocessing if it doesn't exist."""
    logging.info("Checking for prepared data...")
    require_labels = config.train.mode == TrainingModes.picking
    chunking_name_done = get_chunking_name_done(
        config.train.chunks_dir, require_labels=require_labels
    )

    if Path(chunking_name_done).is_file():
        logging.info("Found existing chunked data. Skipping data preparation.")
        return

    logging.info("Chunked data not found or incomplete. Starting data preparation...")
    training_data_dir = config.train.training_data_dir or config.prepData.training_data_dir

    if not Path(training_data_dir).is_dir():
        raise FileNotFoundError(f"Error, training_data_dir {training_data_dir} not found")

    tomos_df = read_particles_csvs(training_data_dir)
    do_chunking(
        tomos_df,
        chunkedDataDir=config.train.chunks_dir,
        desired_particle_pixel_size=config.prepData.desired_particle_pixel_size,
        n_cpus=config.train.n_cpus_for_preprocessing,
        require_labels=require_labels,
        train_val_level=config.train.train_on,
        use_gpus=config.train.use_gpus,
    )
    logging.info("Data preparation complete.")


def _create_model(config: DictConfig, checkpoint_path: Optional[Path]):
    """Factory function to create the appropriate model based on the config."""
    lr = config.train.optimizer.lr
    model = None
    resume_from_checkpoint = None
    checkpointer = ModelCheckpoint(monitor="val_loss", filename="weights", verbose=True)

    if config.train.mode == TrainingModes.picking:
        # Logic for picking mode: distillation, fine-tuning, or from scratch
        if config.train.get('fine_tune_with_distillation') and checkpoint_path:
            logging.info(f"Initializing Distillation model for fine-tuning.")
            logging.info(f"Teacher checkpoint: {checkpoint_path}")
            distill_kwargs = {
                "teacher_checkpoint_path": checkpoint_path,
                "feature_distill_weight": config.train.get('feature_distill_weight', 0.5),
                "lr": lr,
            }
            model = DistillationPickingModel(**distill_kwargs)
            # Start a new training run, don't resume optimizer state etc. from teacher
            resume_from_checkpoint = None

        elif checkpoint_path:
            logging.info(f"Loading model from checkpoint for fine-tuning/resuming: {checkpoint_path}")
            resume_from_checkpoint = checkpoint_path if config.train.restore_full_state else None
            try:
                model = BasePickingModel.load_from_checkpoint(
                    checkpoint_path, train_continue=checkpoint_path, lr=lr
                )
            except RuntimeError:
                logging.info("Could not load as BasePickingModel, trying as SelfSupervisedModel for transfer learning.")
                pretrained_model = SelfSupervisedModel.load_from_checkpoint(checkpoint_path)
                model = BasePickingModel(train_continue=checkpoint_path, lr=lr, model=pretrained_model)
                resume_from_checkpoint = None  # Can't resume state from a different model type
        else:
            logging.info("Initializing new BasePickingModel from scratch.")
            model = BasePickingModel(train_continue=None, lr=lr)

    elif config.train.mode == TrainingModes.selfSupervised:
        # Logic for self-supervised mode
        checkpointer = ModelCheckpoint(
            monitor="val_loss",
            filename=f"weights_{config.train.mode}_{config.train.network.model_type}",
            verbose=True,
        )
        if checkpoint_path:
            logging.info(f"Resuming self-supervised training from: {checkpoint_path}")
            resume_from_checkpoint = checkpoint_path if config.train.restore_full_state else None
            model = SelfSupervisedModel.load_from_checkpoint(
                checkpoint_path, train_continue=checkpoint_path, lr=lr
            )
        else:
            logging.info("Initializing new SelfSupervisedModel from scratch.")
            model = SelfSupervisedModel(train_continue=None, lr=lr)
    else:
        raise ValueError(f"Error, training mode '{config.train.mode}' is not valid")

    return model, resume_from_checkpoint, checkpointer


def _setup_callbacks(config: DictConfig, model: pl.LightningModule, checkpointer: ModelCheckpoint):
    """Creates and returns a list of PyTorch Lightning callbacks."""
    return [
        TQDMProgressBar(refresh_rate=10),
        EarlyStopping(monitor="val_loss_smooth", patience=10, min_delta=0.005, verbose=True),
        checkpointer,
        LearningRateMonitor(logging_interval="epoch"),
        StochasticWeightAveraging(
            annealing_epochs=config.train.COSINE_LR_SCHEDULE_N_EPOCHS,
            swa_lrs=0.1 * model.lr,
        ),
        LRVerificationCallback(),
    ]


def _setup_trainer(config: DictConfig, callbacks: list, logger: TensorBoardLogger):
    """Configures and returns the PyTorch Lightning Trainer."""
    accel, dev_count = accelerator_selector(
        use_cuda=config.train.use_gpus, n_cpus=config.train.N_CPUS_IF_NO_GPU
    )
    return pl.Trainer(
        default_root_dir=config.train.model_dir,
        devices=f"{dev_count}",
        accelerator="auto" if config.train.use_gpus else accel,
        max_epochs=config.train.n_epochs,
        callbacks=callbacks,
        logger=logger,
        overfit_batches=config.train.OVERFIT_N_BATCHES or 0,
        strategy="ddp_find_unused_parameters_false" if accel == "gpu" else "auto",
        precision=config.train.network.TORCH_FLOAT_PRECISION.value,
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

    command_history = f"'''.join(sys.argv)"
    try:
        current_process = psutil.Process()
        parent_process = current_process.parent()
        parent_command = " ".join(
            ["'" + x + "'" if x.startswith('{') else x for x in parent_process.cmdline()]
        )
        command_history += f"Parent command: {parent_command}"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    with open(osp.join(logdir, "command.txt"), "w") as f:
        f.write(command_history)
