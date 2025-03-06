from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Annotated, Tuple

import typer
from omegaconf import MISSING

from tomocpt.defaultConfigs.network_config import NetworkConfig


class TrainingModes(str, Enum):
    selfSupervised = "selfSupervised"
    picking = "picking"


@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.RAdam"
    lr: Annotated[float, typer.Option(help="Learning rate")] = 4e-4
    weight_decay: float = 1e-8
    betas: Tuple[float, float] = (0.9, 0.999)
    # decoupled_weight_decay=True


class CrossValidationLevelSplit(str, Enum):
    tomos = "tomos"
    cubes = "chunks"


@dataclass
class TrainConfig:
    training_data_dir: Annotated[
        Path, typer.Option(help="Path to where the volume label pairs are stored")
    ] = None
    chunks_dir: Annotated[
        Optional[Path], typer.Option(help="Path to directory containing chunked data")
    ] = MISSING
    model_dir: Annotated[
        Path,
        typer.Option(help="The directory where the training weights will be saved"),
    ] = MISSING
    n_epochs: Annotated[int, typer.Option(help="Number of epochs to train")] = 10
    batch_size: Annotated[int, typer.Option(help="batch size")] = 16
    use_gpus: Annotated[bool, typer.Option(help="use cuda for training")] = True
    n_cpus_for_train: Annotated[
        int, typer.Option(help="Number of CPU workers per GPU to pre-process data")
    ] = 8
    experiment_name: Annotated[
        Optional[str], typer.Option(help="The name of the experiment")
    ] = "tomocpt"
    mode: Annotated[TrainingModes, typer.Option(help="The training mode")] = (
        TrainingModes.picking
    )
    train_on: Annotated[
        CrossValidationLevelSplit,
        typer.Option(help="Whether to split train-val on chunks or tomograms"),
    ] = CrossValidationLevelSplit.tomos
    restore_full_state: Annotated[
        bool,
        typer.Option(
            help="If true, load the optimizer and other info on top of the weights"
        ),
    ] = True
    optimizer: Annotated[OptimizerConfig, typer.Option(help="The optimizer")] = field(
        default_factory=OptimizerConfig
    )
    network: Annotated[NetworkConfig, typer.Option(help="The network config")] = field(
        default_factory=NetworkConfig
    )
    launch_tensorboard: Annotated[
        bool, typer.Option(help="Launch tensorboard for evaluating training")
    ] = False

    OVERFIT_N_BATCHES: Optional[int] = None
    N_GPUS: int = 4
    N_CPUS_IF_NO_GPU: int = 32
    USE_CUDA_FOR_DATA: bool = True

    FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS: float = 0.5
    COSINE_LR_SCHEDULE_N_EPOCHS: int = 6
    PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS: int = 6

    CHUNK_SIZE: int = 64
    CHUNK_STRIDE: int = 32
    RANDOM_FRACTION_TO_SAMPLE_TRAIN: float = -1.0  # Train on all the chunks
    SEED_FOR_TRAIN_VAL_SPLIT: int = 42
