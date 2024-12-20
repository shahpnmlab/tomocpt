from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import typer
from typing import Annotated, Optional
from omegaconf import MISSING

class OutputFormat(str, Enum):
    warp = "warp"
    relion = "relion_31" # for backward compatibility
    relion31 = "relion_31"
    relion50 = "relion_50"

@dataclass
class InferConfig:
    tomogram_dir: Annotated[Path, typer.Option(help="The directory that contains the tomograms")] = MISSING
    predictions_dir: Annotated[Path, typer.Option(help="The directory where predictions will be saved")] = MISSING
    weights: Annotated[Path, typer.Option(help="The model fname")] = MISSING
    masks_dir: Annotated[Optional[Path], typer.Option(help="The directory with masks for targeted picking.")] = None
    prediction_particle_length_ang: Annotated[float, typer.Option(help="Particle diameter in Angstroms if spherical, otherwise the longest axis")] = MISSING
    predictions_coord_filename: Annotated[str, typer.Option(help="The starfile name to be saved in the predsDir")] = "tomopicker_coords.star"
    predictions_coord_format:  Annotated[OutputFormat, typer.Option(help="The output coordinate format")] = OutputFormat.relion31
    save_prediction_confidence_map: Annotated[bool, typer.Option(help="If true, predicted labels tomograms are saved")] = False
    save_predicted_coords: Annotated[bool, typer.Option(help="If true, a starfile with coords wil be saved")] = True
    confidence_threshold: Annotated[float, typer.Option(help="Threshold to be applied to the predicted labels to select the centroids")] = 0.3
    nearest_neighbour_dist_angs: Annotated[Optional[float], typer.Option(help="Filter out particle closer than this distance in Angstroms")] = None
    predictions_batch_size: Annotated[int, typer.Option(help="batch size")] = 2
    oversubscribe_factor: Annotated[int, typer.Option(help="How many tomograms should be run in each gpu in parallel")] = 1
    use_cuda: Annotated[bool, typer.Option(help="use cuda for training")] = True
    n_cpus_for_per_gpu: Annotated[int, typer.Option(help="Number of CPU workers per GPU to pre-process data")] = 1

    PATCH_OVERLAP_FACTOR: int = 4
    N_GPUS: int = 1
    N_CPUS_IF_NO_GPU: int = 4
    USE_CUDA_FOR_DATA: bool = True



