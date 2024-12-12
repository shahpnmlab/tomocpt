from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Annotated, Tuple

from omegaconf import MISSING
import typer

import torch

from tomocpt.defaultConfigs.network_config import NetworkConfig


class TrainingModes(str, Enum):
    selfSupervised = "selfSupervised"
    picking = "picking"

@dataclass
class OptimizerConfig:
    _target_: Annotated[str, typer.Option(help="Choose your optimiser e.g. torch.optim.Adam")] = "torch.optim.Adam"
    lr: Annotated[float, typer.Option(help="Learning rate")] = 4e-4
    weight_decay: Annotated[float, typer.Option(help="weight_decay")] = 1e-8
    betas: Annotated[Tuple[float, float], typer.Option(help="betas")] = (0.9, 0.999)
    #decoupled_weight_decay=True

@dataclass
class TrainConfig:
    optimizer: Annotated[OptimizerConfig, typer.Option(help="The optimizer")] = field(default_factory=OptimizerConfig) #TODO: Please try if you can actually change the optimizer in the config and/or command line

    network: Annotated[NetworkConfig, typer.Option(help="The network config")] = field(default_factory=NetworkConfig)
    prepared_data_dir: Annotated[
        Path, typer.Option(help="Path to where the volume label pairs are stored")] = None

    chunks_dir: Annotated[Optional[Path], typer.Option(help="The directory with chunks")] = None #"/tmp/refactor/chunks/" #TODO: We probably want to set it to MISSING, or automatically use /tmp/ within train. Up to pran
    model_dir: Annotated[Path, typer.Option(help="The directory where the model will be saved chunks")] = MISSING #"/tmp/model"
    experiment_name: Annotated[Optional[str], typer.Option(help="The name of the experiment")] = "unnamed"
    n_epochs: Annotated[int, typer.Option(help="Number of epochs to train")] = 10
    mode: Annotated[TrainingModes, typer.Option(help="The training mode")] = TrainingModes.picking #"picking" #TrainingModes.picking does not work. Why?
    restore_full_state: Annotated[bool, typer.Option(help="If true, load the optimizier and other info on top of the weights")] = True
    batch_size: Annotated[int, typer.Option(help="batch size")] = 2
    use_cuda: Annotated[bool, typer.Option(help="use cuda for training")] = True

    OVERFIT_N_BATCHES: Optional[int] = None
    WORKERS_FOR_DATA: Annotated[int, typer.Option(help="Number of CPU workers per GPU to pre-process data")] = 0
    N_GPUS: int = 1
    N_CPUS_IF_NO_GPU: int = 2
    USE_CUDA_FOR_DATA: bool = False

    FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS: float = 0.5  # TODO: Move to train_config
    COSINE_LR_SCHEDULE_N_EPOCHS: int = 6  # TODO: Move to train_config
    PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS: int = 6  # TODO: Move to train_config

    CHUNK_SIZE: Annotated[int, typer.Option(help="The patch size of the cubes")] = 64
    CHUNK_STRIDE: int = 32
    RANDOM_FRACTION_TO_SAMPLE_TRAIN: float = -1.  # Train on all the chunks
    SEED_FOR_TRAIN_VAL_SPLIT: int = 113
