from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import typer
from typing import Annotated, Optional
from omegaconf import MISSING

class OutputFormat(str, Enum):
    warp = "warp"
    relion_31 = "relion_31"
    relion_50 = "relion_50"
    imod = "imod"

@dataclass
class InferConfig:
    tomogram_dir: Annotated[Optional[Path], typer.Option(help="Directory containing tomograms to process.")] = None
    tomogram_file: Annotated[Optional[Path], typer.Option(help="Path to a single tomogram file.")] = None
    predictions_dir: Annotated[Path, typer.Option(help="Directory to save prediction outputs.")] = MISSING
    weights: Annotated[Path, typer.Option(help="Path to the trained model weights file (.ckpt).")] = MISSING
    masks_dir: Annotated[Optional[Path], typer.Option(help="Optional directory with masks for targeted picking. Mask filenames must match tomogram filenames.")] = None
    length: Annotated[float, typer.Option(help="Particle diameter in Angstroms.")] = MISSING
    distance_threshold: Annotated[Optional[float], typer.Option(help="Minimum distance between picked particles in Angstroms. Defaults to particle diameter if not set.")] = None
    confidence_threshold: Annotated[float, typer.Option(help="Minimum confidence score for a voxel to be considered as a potential particle center.")] = 0.3
    predictions_coord_filename: Annotated[str, typer.Option(help="Filename for the output coordinate star file. (Not used for imod format).")] = "tomocpt_coords.star"
    predictions_coord_format: Annotated[OutputFormat, typer.Option(help="Format for the output coordinate file.")] = OutputFormat.relion_31
    predictions_batch_size: Annotated[int, typer.Option(help="Internal batch size for model inference on tomogram patches.")] = 2
    save_predicted_coords: Annotated[bool, typer.Option(help="Enable/disable saving of predicted particle coordinates.")] = True
    save_prediction_confidence_map: Annotated[bool, typer.Option(help="Save the full 3D prediction confidence map as an MRC file.")] = False
    use_cuda: Annotated[bool, typer.Option(help="Enable/disable CUDA for GPU acceleration.")] = True
    use_tta: Annotated[bool, typer.Option(help="Use test-time augmentation for prediction. This might improve picking, but at the cost of time anf GPU memory.")] = False
    invert_contrast: Annotated[bool, typer.Option(help="Invert tomogram contrast (signal becomes black).")] = False
    cpus_per_worker: Annotated[int, typer.Option(help="Number of CPU threads to support each Dask worker (for I/O, etc.).")] = 2
    PATCH_OVERLAP_FACTOR: int = 4
    N_CPUS_IF_NO_GPU: int = 8
