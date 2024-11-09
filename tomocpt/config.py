import os
import tempfile
from dataclasses import dataclass

DATA_CHUNKS_DIR: str = "/tmp/refactor/chunks/"
BATCH_SIZE: int = 8
WORKERS_FOR_DATA: int = 1
N_GPUS: int = 1
USE_CUDA_FOR_DATA: bool = False