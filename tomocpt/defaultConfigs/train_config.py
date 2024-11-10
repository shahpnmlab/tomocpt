from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    chunks_dir: str = "/tmp/refactor/chunks/"
    model_dir: str = "/tmp/model"
    EXPERIMENT_NAME: str = "unnamed"
    N_EPOCHS: int = 10
    OVERFIT_N_BATCHES: Optional[int] = 10

    BATCH_SIZE: int = 2
    WORKERS_FOR_DATA: int = 1
    N_GPUS: int = 1
    USE_CUDA_FOR_DATA: bool = False

    restoreFullStateWhenContinue: bool = True
