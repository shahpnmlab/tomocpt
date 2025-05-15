import torch
import typer
from click.core import batch

from tomocpt.logger import get_logger

try:
    from itertools import batched
except:
    from more_itertools import batched

from pathlib import Path
from omegaconf import DictConfig
from typing import Annotated
from joblib import Parallel, delayed
from tomocpt.infer.helpers import process_extracted_coordinates, infer_tomos

logger = get_logger()

def predict(
        plot: Annotated[bool, typer.Option(help="Plot the cubes")] = False,
        config: DictConfig = None,
):
    """
    Enhanced parallel inference on tomogram data with improved device management.

    This function processes multiple tomogram files (.mrc or .rec) in parallel, applying
    a trained model to detect particles and optionally extract their coordinates.
    """
    from tomocpt.mainConfig import mainConfig
    from tomocpt.utils import accelerator_selector
    from tqdm import tqdm

    infer_config = mainConfig.infer

    # Setup paths
    tomosDirPath = Path(infer_config.tomogram_dir).resolve()
    Path(infer_config.predictions_dir).mkdir(parents=True, exist_ok=True)

    # Find all tomogram files
    data_fnames = []
    patterns = ("*.mrc", "*.rec")
    for pattern in patterns:
        data_fnames.extend(tomosDirPath.glob(pattern))
    data_fnames = sorted(data_fnames)

    if not data_fnames:
        logger.warning(f"No tomogram files found in {tomosDirPath}")
        return

    # Determine available compute resources
    accel, dev_count = accelerator_selector(
        use_cuda=infer_config.use_cuda, n_cpus=infer_config.N_CPUS_IF_NO_GPU
    )

    # Configure GPU/CPU usage
    if accel.startswith("cpu"):
        n_gpus = None
        logger.warning(f"Using CPU for processing {len(data_fnames)} tomograms")
    else:
        n_gpus = dev_count
        logger.info(f"Using {n_gpus} GPUs for processing {len(data_fnames)} tomograms")

        # Pre-initialize CUDA to prevent race conditions
        for gpu_id in range(n_gpus):
            with torch.cuda.device(gpu_id):
                torch.tensor([1.0], device=f"cuda:{gpu_id}")
                torch.cuda.empty_cache()

    # Calculate batch size for distribution
    batch_size = max(1, len(data_fnames) // (infer_config.oversubscribe_factor * max(1, dev_count)))

    # Setup progress tracking (instead of "Working with" prints)
    total_batches = (len(data_fnames) + batch_size - 1) // batch_size
    logger.info(f"Processing in {total_batches} batch{'es' if total_batches > 1 else ''}")

    # Run parallel inference with the requested GPU distribution formula
    results = Parallel(
        n_jobs=infer_config.oversubscribe_factor * max(1, dev_count),
        batch_size=1,
        verbose=10  # Show progress bar instead of print statements
    )(
        delayed(infer_tomos)(
            batch_fnames,
            infer_config.predictions_dir,
            infer_config.weights,
            particleLengthAng=infer_config.length,
            gpu_id=(
                (i // infer_config.oversubscribe_factor) % n_gpus
                if n_gpus is not None
                else None
            ),
            batch_size=infer_config.predictions_batch_size,
            plot=plot,
            save_pred_mask=infer_config.save_prediction_confidence_map,
            extract_coords=infer_config.save_predicted_coords,
            nearest_neigs_angs=infer_config.distance_threshold,
            threshold=infer_config.confidence_threshold,
            masksDir=infer_config.masks_dir,
        )
        for i, batch_fnames in enumerate(batched(data_fnames, n=batch_size))
    )

    # Process and save coordinates if requested
    if infer_config.save_predicted_coords:
        logger.info("Processing extracted coordinates...")
        total_coords = process_extracted_coordinates(
            results=results,
            output_dir=mainConfig.infer.predictions_dir,
            output_format=mainConfig.infer.predictions_coord_format,
            output_filename=mainConfig.infer.predictions_coord_filename,
        )
        logger.info(f"Processing complete. Coordinates saved to {mainConfig.infer.predictions_dir}")
    else:
        logger.info("Processing complete.")