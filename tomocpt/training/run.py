import os
import os.path as osp
import shutil
import warnings
import subprocess
import sys
import psutil
import typer
import torch

from omegaconf import DictConfig
from pathlib import Path
from typing import Annotated, Optional
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from tomocpt.dataManager.chunking import do_chunking, get_chunking_name_done
from tomocpt.dataManager.dataloading import Data
from tomocpt.networks.baseModel import verify_different_lrs
from tomocpt.training.callbacks import LRVerificationCallback
from tomocpt.utils import read_particles_csvs, is_main_process
from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.networks.distillationModel import DistillationPickingModel
from tomocpt.networks.continuousLearningModel import ContinuousLearningModel

from tomocpt.logger import get_logger

logging = get_logger()
warnings.filterwarnings("ignore", ".*SwinUNETR.*", FutureWarning)


def train(
        train_continue: Annotated[
            Optional[Path],
            typer.Option(
                help="Path to pre-existing checkpoint file for fine-tuning with new data"
            ),
        ] = None,
        config: DictConfig = None,
):
    """
    Trains a deep learning model for particle picking or self-supervised learning using PyTorch Lightning.
    """

    from tomocpt.mainConfig import mainConfig

    torch.set_float32_matmul_precision(mainConfig.train.network.TORCH_MATMUL_PRECISION)
    from tomocpt.defaultConfigs.train_config import TrainingModes
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
    from tomocpt.utils import accelerator_selector
    from pytorch_lightning.callbacks import (
        TQDMProgressBar,
        EarlyStopping,
        ModelCheckpoint,
        LearningRateMonitor,
        StochasticWeightAveraging,
    )

    kwargs = dict(lr=mainConfig.train.optimizer.lr)

    train__config = mainConfig.train
    network__config = mainConfig.train.network
    assert mainConfig.train.model_dir, "Error, you need to provide a model_dir"
    assert mainConfig.train.chunks_dir, "Error, you need to provide a chunks_dir"

    chunks_dir = mainConfig.train.chunks_dir

    require_labels = mainConfig.train.mode == TrainingModes.picking
    chunking_name_done = get_chunking_name_done(
        chunks_dir, require_labels=require_labels
    )
    if not Path(chunking_name_done).is_file():
        logging.info("Chunked data not found or incomplete. Starting data preparation...")
        if mainConfig.train.training_data_dir is None:
            training_data_dir = mainConfig.prepData.training_data_dir
        else:
            training_data_dir = mainConfig.train.training_data_dir

        if not Path(training_data_dir).is_dir():
            raise RuntimeError(
                f"Error, prepared_data_dir {training_data_dir} not found"
            )

        tomosDf = read_particles_csvs(training_data_dir)

        do_chunking(
            tomosDf,
            chunkedDataDir=mainConfig.train.chunks_dir,
            desired_particle_pixel_size=mainConfig.prepData.desired_particle_pixel_size,
            n_cpus=train__config.n_cpus_for_preprocessing,
            require_labels=require_labels,
            train_val_level=mainConfig.train.train_on,
            use_gpus=mainConfig.train.use_gpus,
        )
        logging.info("Data preparation complete.")
    else:
        logging.info("Found existing chunked data. Skipping data preparation.")

    # Model selection and initialization
    if mainConfig.train.mode == TrainingModes.picking:
        # Check for Continual Learning mode first. Use .get() for safety if key is not in config.
        if mainConfig.train.continuous_learning:
            logging.info("Initializing model in Continual Learning mode.")
            # Gather CL-specific hyperparameters from config
            cl_kwargs = {
                "config": mainConfig,  # Pass the whole config to be stored in hparams
                "memory_size": mainConfig.train.memory_size,
                "replay_batch_size": mainConfig.train.replay_batch_size,
                "distill_weight": mainConfig.train.distill_weight,
                "temperature": mainConfig.train.temperature,
                **kwargs  # Pass base kwargs like lr
            }

            checkpointer = ModelCheckpoint(monitor="val_loss", filename="weights_cl", verbose=True)

            if train_continue:
                # This is a new task in the CL sequence. Load the previous model and memory.
                logging.info(f"Continual learning: starting new task from checkpoint {train_continue}")
                # Signal to the model to create a teacher on the first epoch of this new task
                mainConfig.train.train_new_task = True

                pl_model = ContinuousLearningModel.load_from_checkpoint(
                    train_continue,
                    train_continue=train_continue,  # For base class optimizer logic
                    **cl_kwargs
                )
                resume_from_checkpoint = train_continue if mainConfig.train.restore_full_state else None
            else:
                # This is the very first task in the CL sequence. No teacher yet.
                logging.info("Continual learning: starting the first task from scratch.")
                mainConfig.train.train_new_task = False  # No teacher needed

                pl_model = ContinuousLearningModel(train_continue=None, **cl_kwargs)
                resume_from_checkpoint = None

        # --- The rest of the logic is for non-continual learning modes ---
        elif train_continue and mainConfig.train.use_distillation:
            # Using standard distillation with a fixed teacher model from train_continue
            checkpointer = ModelCheckpoint(monitor="val_loss", filename="weights", verbose=True)
            distill_kwargs = {
                "teacher_checkpoint_path": train_continue,
                "distill_weight": mainConfig.train.distill_weight,
                "temperature": mainConfig.train.temperature
            }
            if mainConfig.train.restore_full_state:
                resume_from_checkpoint = None
                pl_model = DistillationPickingModel(train_continue=None, **distill_kwargs, **kwargs)
                checkpoint = torch.load(train_continue)
                student_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('teacher')}
                pl_model.load_state_dict(student_state_dict, strict=False)
            else:
                resume_from_checkpoint = None
                pl_model = DistillationPickingModel(train_continue=None, **distill_kwargs, **kwargs)
        else:
            # Normal training without distillation or continual learning
            checkpointer = ModelCheckpoint(monitor="val_loss", filename="weights", verbose=True)
            if train_continue:
                resume_from_checkpoint = train_continue if mainConfig.train.restore_full_state else None
                try:
                    pl_model = BasePickingModel.load_from_checkpoint(
                        train_continue, train_continue=train_continue, **kwargs
                    )
                except RuntimeError:
                    pretrained_model = SelfSupervisedModel.load_from_checkpoint(train_continue)
                    pl_model = BasePickingModel(train_continue=train_continue, **kwargs, model=pretrained_model)
                    del pretrained_model
                    resume_from_checkpoint = None
            else:
                pl_model = BasePickingModel(train_continue=None, **kwargs)
                resume_from_checkpoint = None
    # ======================================================================

    elif mainConfig.train.mode == TrainingModes.selfSupervised:
        # This section remains unchanged
        Model = SelfSupervisedModel
        checkpointer = ModelCheckpoint(
            monitor="val_loss",
            filename=f"weights_{mainConfig.train.mode}_{network__config.model_type}",
            verbose=True,
        )

        if train_continue:
            resume_from_checkpoint = train_continue if mainConfig.train.restore_full_state else None
            pl_model = Model.load_from_checkpoint(train_continue, train_continue=train_continue, **kwargs)
        else:
            pl_model = Model(train_continue=None, **kwargs)
            resume_from_checkpoint = None
    else:
        raise ValueError("Error, mode (%s) not valid" % mainConfig.train.mode)

    # Set up data
    data = Data(
        data_dir=mainConfig.train.chunks_dir,
        return_labels=(mainConfig.train.mode == TrainingModes.picking),
        batch_size=mainConfig.train.batch_size,
        workers_for_data=mainConfig.train.n_cpus_for_dataloading,
    )

    data.setup()

    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        EarlyStopping(
            monitor="val_loss_smooth",
            patience=10,
            min_delta=0.005,
            verbose=True,
        ),
        checkpointer,
        LearningRateMonitor(logging_interval="epoch"),
        StochasticWeightAveraging(
            annealing_epochs=train__config.COSINE_LR_SCHEDULE_N_EPOCHS,
            swa_lrs=0.1 * pl_model.lr,
        ),
        LRVerificationCallback(),
    ]

    if is_main_process():
        logging.info(f"Size of the training dataset {len(data.train_dataloader())}")
        verify_different_lrs(pl_model)

    tb_logger = TensorBoardLogger(
        save_dir=f"{mainConfig.train.model_dir}/{mainConfig.train.experiment_name}",
        name="",
        version="",
    )

    accel, dev_count = accelerator_selector(
        use_cuda=mainConfig.train.use_gpus, n_cpus=mainConfig.train.N_CPUS_IF_NO_GPU
    )

    trainer = pl.Trainer(
        default_root_dir=mainConfig.train.model_dir,
        devices=f"{dev_count}",
        accelerator="auto" if mainConfig.train.use_gpus else accel,
        max_epochs=mainConfig.train.n_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        overfit_batches=(
            mainConfig.train.OVERFIT_N_BATCHES
            if mainConfig.train.OVERFIT_N_BATCHES
            else 0
        ),
        strategy="ddp_find_unused_parameters_false" if accel == "gpu" else "auto",
        precision=network__config.TORCH_FLOAT_PRECISION.value,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_model_summary=False,
    )

    if trainer.is_global_zero:
        _copyCodeForReproducibility(trainer.log_dir)

    if mainConfig.train.launch_tensorboard and trainer.is_global_zero:
        subprocess.Popen(
            [
                "tensorboard",
                "--logdir",
                f"{mainConfig.train.model_dir}/{mainConfig.train.experiment_name}",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        logging.info(
            f"tensorboard --logdir {mainConfig.train.model_dir}/{mainConfig.train.experiment_name}"
        )

    trainer.fit(pl_model, datamodule=data, ckpt_path=resume_from_checkpoint)


def _copyCodeForReproducibility(logdir):
    """
    Copy the code to the logdir so that reproducibility is ensured

    Args:
        logdir: The directory were the code will be saved

    Returns:

    """
    copycodedir = osp.join(logdir, "code")
    os.makedirs(copycodedir, exist_ok=True)
    copycodedir = osp.join(copycodedir, "tomocpt")

    modulePath = osp.abspath(sys.modules[__name__].__file__)
    rootPath = osp.dirname(osp.dirname(modulePath))

    for root, dirs, files in os.walk(rootPath):
        # Iterate through all folders
        for directory in dirs:
            # Create the corresponding directory in the target path
            source_folder = osp.join(root, directory)
            target_folder = source_folder.replace(rootPath, copycodedir)
            os.makedirs(target_folder, exist_ok=True)
        # Iterate through all Python files
        for file in files:
            if file.endswith(".py") or file.endswith(".yaml"):
                # Copy the Python file to the corresponding directory in the target path
                source_file = os.path.join(root, file)
                target_file = source_file.replace(rootPath, copycodedir)
                shutil.copy2(source_file, target_file)

    fname = osp.join(logdir, "command.txt")
    with open(fname, "w") as f:
        f.write(" ".join(sys.argv))

    current_process = psutil.Process()
    # Get the parent process
    parent_process = current_process.parent()
    # Get the command line of the parent process
    parent_command = " ".join(
        ["'" + x + "'" if x.startswith('{"') else x for x in parent_process.cmdline()]
    )
    fname = osp.join(logdir, "parent_command.txt")
    with open(fname, "w") as f:
        f.write(f"{parent_command}\n")