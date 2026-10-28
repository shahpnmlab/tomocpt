import warnings
import subprocess
import typer
import torch

from omegaconf import DictConfig
from pathlib import Path
from typing import Annotated, Optional
from pytorch_lightning.loggers import TensorBoardLogger

from tomocpt.dataManager.dataloading import Data
from tomocpt.networks.baseModel import verify_different_lrs
from tomocpt.utils import is_main_process
from tomocpt.logger import get_logger
from tomocpt.training.helpers import (
    _prepare_data_if_needed,
    _create_model,
    _setup_callbacks,
    _setup_trainer,
    _copy_code_for_reproducibility
)

warnings.filterwarnings("ignore", ".*SwinUNETR.*", FutureWarning)
logging = get_logger()


def train(config: DictConfig = None):
    """
    Trains a deep learning model for particle picking or self-supervised learning.
    """
    # Validate that resume_from and fine_tune_from are not used at the same time
    if config.resume_from and config.fine_tune_from:
        raise ValueError("Cannot use --resume-from and --fine-tune-from at the same time.")

    # Use the config object passed in by the decorator, not the global mainConfig
    torch.set_float32_matmul_precision(config.network.TORCH_MATMUL_PRECISION)
    assert config.model_dir, "Error, you need to provide a model_dir"
    assert config.chunks_dir, "Error, you need to provide a chunks_dir"

    # 1. Prepare data if needed
    _prepare_data_if_needed(config)

    # 2. Create the model based on configuration
    model, resume_from_ckpt, checkpointer = _create_model(config)
    if model is None:
        logging.error("Model initialization failed. Exiting.")
        return

    # 3. Set up data loaders
    ssl_cfg = getattr(config, "ssl", None)
    if config.mode == "selfSupervised" and ssl_cfg is not None and int(ssl_cfg.version) == 2:
        from tomocpt.dataManager.ssl_views import SSLPairData

        data = SSLPairData(chunks_dir=config.chunks_dir, batch_size=config.batch_size,
                           workers_for_data=config.n_cpus_for_dataloading, ssl_cfg=ssl_cfg,
                           persistent_workers=config.use_gpus)
    else:
        data = Data(
            data_dir=config.chunks_dir,
            return_labels=(config.mode == "picking"),
            batch_size=config.batch_size,
            workers_for_data=config.n_cpus_for_dataloading,
            augmentation=config.augmentation,
            pipeline_version=config.data_pipeline_version,
            scale_cfg=config.scale,
            chunk_sampling=config.chunk_sampling,
            repeat_factor_threshold=config.repeat_factor_threshold,
        )
    data.setup()

    # 4. Set up callbacks
    callbacks = _setup_callbacks(config, model, checkpointer)

    if is_main_process():
        logging.info(f"Size of the training dataset {len(data.train_dataloader())}")
        verify_different_lrs(model)

    # 5. Set up logger and trainer
    tb_logger = TensorBoardLogger(
        save_dir=f"{config.model_dir}/{config.experiment_name}",
        name="",
        version="",
    )
    trainer = _setup_trainer(config, callbacks, tb_logger)

    # 6. Save code for reproducibility and launch TensorBoard
    if trainer.is_global_zero:
        _copy_code_for_reproducibility(trainer.log_dir)
        if config.launch_tensorboard:
            subprocess.Popen(
                ["tensorboard", "--logdir", trainer.logger.log_dir],
            )
            logging.info(f"Launched TensorBoard: tensorboard --logdir {trainer.logger.log_dir}")

    # 7. Start training
    logging.info("Starting trainer.fit()...")
    trainer.fit(model, datamodule=data, ckpt_path=resume_from_ckpt)
    logging.info("Training finished.")

    # 8. Per-class threshold calibration (pipeline-v2 picking only, global rank 0)
    if config.mode == "picking" and int(config.data_pipeline_version) == 2 and trainer.is_global_zero:
        _calibrate_best_checkpoint(config, trainer)


def _calibrate_best_checkpoint(config, trainer) -> None:
    """Calibrates the class thresholds of the best checkpoint on the validation tomograms of
    ``chunk_layout.json``. A failure is logged and leaves the checkpoint uncalibrated."""
    try:
        import copy

        from tomocpt.dataManager.metadata import read_chunk_layout
        from tomocpt.mainConfig import mainConfig
        from tomocpt.predict.calibration import calibrate_checkpoint

        callback = trainer.checkpoint_callback
        ckpt_path = getattr(callback, "best_model_path", "") or getattr(callback, "last_model_path", "")
        if not ckpt_path or not Path(ckpt_path).is_file():
            logging.error("Threshold calibration skipped: no saved checkpoint found.")
            return
        layout = read_chunk_layout(config.chunks_dir) or {}
        val_tomograms = layout.get("val_tomograms") or []
        if not val_tomograms:
            logging.error("Threshold calibration skipped: chunk_layout.json lists no val_tomograms.")
            return
        # The calibration settings (beta, minimum particles, cross-class NMS) of the infer section.
        infer_cfg = copy.deepcopy(mainConfig.infer)
        infer_cfg.use_cuda = torch.cuda.is_available()
        infer_cfg.save_prediction_confidence_map = False
        # Validation tomograms are read as the chunking read them.
        infer_cfg.invert_contrast = False
        infer_cfg.masks_dir = None
        infer_cfg.classes = None
        logging.info(f"Calibrating per-class thresholds of {ckpt_path} on {len(val_tomograms)} tomogram(s)...")
        calibrate_checkpoint(ckpt_path, val_tomograms, infer_cfg)
        logging.info("Threshold calibration stored in the checkpoint.")
    except Exception as e:
        logging.error(f"Threshold calibration failed; the checkpoint stays uncalibrated: {e}", exc_info=True)