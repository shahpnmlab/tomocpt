from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import typer
from typing import Annotated, Optional
from omegaconf import MISSING

class OutputFormat(str, Enum):
    warp, relion, relion31, relion50 = "warp", "relion_31", "relion_31", "relion_50"

@dataclass
class InferConfig:
    tomogram_dir: Annotated[Optional[Path], typer.Option(help="Directory of tomograms")] = None
    tomogram_file: Annotated[Optional[Path], typer.Option(help="A single tomogram file")] = None
    predictions_dir: Annotated[Path, typer.Option(help="Prediction output directory")] = MISSING
    weights: Annotated[Path, typer.Option(help="Model weights file")] = MISSING
    masks_dir: Annotated[Optional[Path], typer.Option(help="Masks for targeted picking")] = None
    length: Annotated[float, typer.Option(help="Particle diameter in Angstroms")] = MISSING
    distance_threshold: Annotated[Optional[float], typer.Option(help="NN distance threshold (Angstroms)")] = None
    predictions_coord_filename: str = "tomopicker_coords.star"
    predictions_coord_format: OutputFormat = OutputFormat.relion31
    save_prediction_confidence_map: bool = False
    save_predicted_coords: bool = True
    confidence_threshold: float = 0.3
    predictions_batch_size: int = 2
    oversubscribe_factor: int = 1
    use_cuda: bool = True
    n_cpus_per_gpu: int = 1
    PATCH_OVERLAP_FACTOR: int = 4
    N_GPUS: int = 1
    N_CPUS_IF_NO_GPU: int = 32
