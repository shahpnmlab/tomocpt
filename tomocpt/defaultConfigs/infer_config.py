from dataclasses import dataclass
from enum import Enum
from typing import Optional



@dataclass
class InferConfig:


    batch_size: int = 2
    WORKERS_FOR_DATA: int = 1
    use_cuda: bool = True
    N_GPUS: int = 1
    N_CPUS_IF_NO_GPU: int = 4
    USE_CUDA_FOR_DATA: bool = False



