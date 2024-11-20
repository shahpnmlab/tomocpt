from dataclasses import dataclass
from pathlib import Path

import typer
from typing import Annotated, Optional
from omegaconf import MISSING


@dataclass
class InferConfig:
    tomosDir: Annotated[Path, typer.Option(help="The directory that contains the tomograms with the same particle size")] = MISSING
    predsDir: Annotated[Path, typer.Option(help="The directory where predictions will be saved")] = MISSING
    modelFname: Annotated[Path, typer.Option(help="The model fname")] = MISSING
    particleLengthAng: Annotated[float, typer.Option(help="Particle diameter in Angstroms")] = MISSING # TODO: Pranav, check if it is lenght, radius or whatever
    batch_size: Annotated[int, typer.Option(help="batch size")] = 2
    oversubscribeFactor: Annotated[int, typer.Option(help="How many tomograms should be run in each gpu in parallel")] = 1
    savePredMasks: Annotated[bool, typer.Option(help="#TODO")] = False
    extractCoords: Annotated[bool, typer.Option(help="#TODO")] = True
    nearest_neigs_angs: Annotated[Optional[float], typer.Option(help="#TODO")] = None #TODO: Is it not better to have it as fraction of particle size
    deep_threshold: Annotated[float, typer.Option(help="#TODO")] = 0.3
    outCoordFname: Annotated[str, typer.Option(help="#TODO")] = "tomopicker_coords.star"
    masksDir: Annotated[Optional[Path], typer.Option(help="The directory with masks")] = None #TODO: Pranav, how does this work?

    WORKERS_FOR_DATA: Annotated[int, typer.Option(help="Number of CPU workers per GPU to pre-process data")] = 1
    use_cuda: Annotated[bool, typer.Option(help="use cuda for training")] = True
    N_GPUS: int = 1
    N_CPUS_IF_NO_GPU: int = 4
    USE_CUDA_FOR_DATA: bool = False



