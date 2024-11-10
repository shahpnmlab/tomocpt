from dataclasses import dataclass
from enum import Enum
from typing import Optional

class TrainingModes(Enum):
    selfSupervised = "selfSupervised"
    picking = "picking"


@dataclass
class TrainConfig:
    chunks_dir: str = "/tmp/refactor/chunks/"
    model_dir: str = "/tmp/model"
    experiment_name: str = "unnamed"
    learning_rate:float = 4e-4
    n_epochs: int = 10
    mode: TrainingModes = TrainingModes.picking
    restore_full_state: bool = True

    OVERFIT_N_BATCHES: Optional[int] = 10
    batch_size: int = 2
    WORKERS_FOR_DATA: int = 1
    N_GPUS: int = 1
    N_CPUS_IF_NO_GPU: int = 4
    USE_CUDA_FOR_DATA: bool = False

    WEIGHT_DECAY:float = 1e-8


