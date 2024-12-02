from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import typer
from typing import Annotated, Optional
from omegaconf import MISSING

class OutputFormat(str, Enum):
    warp = "warp"
    relion = "relion"

@dataclass
class InferConfig:
    tomosDir: Annotated[Path, typer.Option(help="The directory that contains the tomograms with the same particle size")] = MISSING
    predsDir: Annotated[Path, typer.Option(help="The directory where predictions will be saved")] = MISSING
    modelFname: Annotated[Path, typer.Option(help="The model fname")] = MISSING
    particleLengthAng: Annotated[float, typer.Option(help="Particle diameter in Angstroms if spherical, otherwise the longest axis")] = MISSING
    batch_size: Annotated[int, typer.Option(help="batch size")] = 2
    oversubscribeFactor: Annotated[int, typer.Option(help="How many tomograms should be run in each gpu in parallel")] = 1
    savePredMasks: Annotated[bool, typer.Option(help="If true, predicted labels tomograms are saved")] = False
    extractCoords: Annotated[bool, typer.Option(help="If true, a starfile with coords wil be saved")] = True
    nearest_neigs_angs: Annotated[Optional[float], typer.Option(help="Filter out particle closer than this distance in Angstroms")] = None
    deep_threshold: Annotated[float, typer.Option(help="Threshold to be applied to the predicted labels to select the centroids")] = 0.3
    outCoordFname: Annotated[str, typer.Option(help="The starfile name to be saved in the predsDir")] = "tomopicker_coords.star"
    outCoordFormat:  Annotated[OutputFormat, typer.Option(help="The output coordinate format")] = OutputFormat.relion
    masksDir: Annotated[Optional[Path], typer.Option(help="The directory with masks.")] = None #TODO: Pranav, how does this work. Explain in help?

    patch_overlap_factor: int = 4

    WORKERS_FOR_DATA: Annotated[int, typer.Option(help="Number of CPU workers per GPU to pre-process data")] = 1
    use_cuda: Annotated[bool, typer.Option(help="use cuda for training")] = True
    N_GPUS: int = 1
    N_CPUS_IF_NO_GPU: int = 4
    USE_CUDA_FOR_DATA: bool = False



