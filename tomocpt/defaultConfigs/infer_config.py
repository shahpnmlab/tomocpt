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
    weights: Annotated[Optional[Path], typer.Option(help="Path to the trained model weights file (.ckpt). Required unless --skip_inference is used.")] = None
    masks_dir: Annotated[Optional[Path], typer.Option(help="Optional directory with masks for targeted picking. Mask filenames must match tomogram filenames.")] = None
    length: Annotated[Optional[float], typer.Option(help="Particle diameter in Angstroms. Required for pipeline-v1 checkpoints; not used for pipeline-v2 checkpoints (the class table gives the diameters).")] = None
    distance_threshold: Annotated[Optional[float], typer.Option(help="Minimum distance between picked particles in Angstroms. Defaults to particle diameter if not set.")] = None
    confidence_threshold: Annotated[float, typer.Option(help="Minimum confidence score for a voxel to be considered as a potential particle center.")] = 0.3
    predictions_coord_filename: Annotated[str, typer.Option(help="Filename for the output coordinate star file. (Not used for imod format).")] = "tomocpt_coords.star"
    predictions_coord_format: Annotated[OutputFormat, typer.Option(help="Format for the output coordinate file.")] = OutputFormat.relion_31
    predictions_batch_size: Annotated[int, typer.Option(help="Internal batch size for model inference on tomogram patches.")] = 2
    save_predicted_coords: Annotated[bool, typer.Option(help="Enable/disable saving of predicted particle coordinates.")] = True
    save_prediction_confidence_map: Annotated[bool, typer.Option(help="Save the full 3D prediction confidence map as an MRC file.")] = False
    skip_inference: Annotated[bool, typer.Option(help="Skip the inference step and extract peaks from confidence maps already present in `predictions_dir`.")] = False
    use_cuda: Annotated[bool, typer.Option(help="Enable/disable CUDA for GPU acceleration.")] = True
    use_tta: Annotated[bool, typer.Option(help="Use test-time augmentation for prediction. This might improve picking, but at the cost of time anf GPU memory.")] = False
    invert_contrast: Annotated[bool, typer.Option(help="Invert tomogram contrast (signal becomes black).")] = False
    cpus_per_worker: Annotated[int, typer.Option(help="Number of CPU threads to support each Dask worker (for I/O, etc.).")] = 2
    apix_override: Annotated[Optional[float], typer.Option(help="Voxel size in Angstroms to use instead of the MRC header value (pipeline v2).")] = None
    classes: Annotated[Optional[str], typer.Option(help="Comma-separated class names to pick (pipeline v2). Default: every class of the model.")] = None
    cross_class_nms: Annotated[bool, typer.Option(help="Suppress peaks of different classes closer than half the smaller radius, keeping the higher score (pipeline v2).")] = True
    accumulate_dtype: Annotated[str, typer.Option(help="dtype of the per-class map accumulators: float32 or float16 (pipeline v2).")] = "float32"
    accumulate_device: Annotated[str, typer.Option(help="Device of the per-class map accumulators: auto (GPU if the maps fit in free memory), cuda or cpu (pipeline v2).")] = "auto"
    class_thresholds: Annotated[Optional[str], typer.Option(help="Per-class confidence thresholds by class name, e.g. 'ribosome:0.4,proteasome:0.35'. Overrides the checkpoint calibration (pipeline v2).")] = None
    calibration_beta: Annotated[float, typer.Option(help="Beta of the F-beta score maximised by the per-class threshold calibration (pipeline v2).")] = 4.0
    calibration_min_particles: Annotated[int, typer.Option(help="Minimum number of true particles of a class for its threshold to be calibrated (pipeline v2).")] = 20
    PATCH_OVERLAP_FACTOR: int = 4
    N_CPUS_IF_NO_GPU: int = 8
