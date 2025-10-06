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


def train(
        checkpoint_path: Annotated[
            Optional[Path],
            typer.Option(
                help="Path to a checkpoint to resume from, fine-tune, or use as a teacher."
            ),
        ] = None,
        config: DictConfig = None,
):
    """
    Trains a deep learning model for particle picking or self-supervised learning.
    """
    # Use the config object passed in by the decorator, not the global mainConfig
    torch.set_float32_matmul_precision(config.network.TORCH_MATMUL_PRECISION)
    assert config.model_dir, "Error, you need to provide a model_dir"
    assert config.chunks_dir, "Error, you need to provide a chunks_dir"

    # 1. Prepare data if needed
    _prepare_data_if_needed(config)

    # 2. Create the model based on configuration
    model, resume_from_ckpt, checkpointer = _create_model(config, checkpoint_path)
    if model is None:
        logging.error("Model initialization failed. Exiting.")
        return

    # 3. Set up data loaders
    data = Data(
        data_dir=config.chunks_dir,
        return_labels=(config.mode == "picking"),
        batch_size=config.batch_size,
        workers_for_data=config.n_cpus_for_dataloading,
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