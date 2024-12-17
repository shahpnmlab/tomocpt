import typer
from pathlib import Path
from dataclasses import dataclass
from typing import Annotated, Optional


@dataclass
class GlobalPropertyConfig:
    my_global_prop: Annotated[Optional[int], typer.Option(help="A number")] = 5

    # chunks_dir: Annotated[Optional[Path], typer.Option(help="The directory with chunks")] = Path("/tmp/chunks/")
    #
    # prepared_data_dir: Annotated[
    #     Path, typer.Option(help="Path to where the volume label pairs should be stored")
    # ] = Path("/tmp/inputs")
    #
    # WORKERS_FOR_DATA: Annotated[int, typer.Option(help="Number of CPU workers per GPU to pre-process data")] = 1
    # USE_CUDA: Annotated[bool, typer.Option(help="use cuda for training")] = True
    # N_GPUS: int = 1
    # N_CPUS_IF_NO_GPU: int = 4
    # USE_CUDA_FOR_DATA: bool = False
    # CHUNK_SIZE: Annotated[int, typer.Option(help="The patch size of the cubes")] = 64
    # CHUNK_STRIDE: int = 32
    # RANDOM_FRACTION_TO_SAMPLE_TRAIN: float = -1.  # Train on all the chunks
    # SEED_FOR_TRAIN_VAL_SPLIT: int = 113
