import typer

try:
    from itertools import batched
except:
    from more_itertools import batched

from pathlib import Path
from omegaconf import DictConfig
from typing import Annotated
from joblib import Parallel, delayed
from tomocpt.infer.helpers import process_extracted_coordinates, infer_tomos


def predict(
    plot: Annotated[bool, typer.Option(help="Plot the cubes")] = False,
    config: DictConfig = None,
):
    """
    Performs parallel inference on tomogram data for particle detection and coordinate extraction.

    This function processes multiple tomogram files (.mrc or .rec) in parallel, applying a trained model
    to detect particles and optionally extract their coordinates. It supports both CPU and GPU inference
    with automatic device selection and load balancing.
    """

    from tomocpt.mainConfig import mainConfig
    from tomocpt.utils import accelerator_selector

    infer_config = mainConfig.infer

    tomosDirPath = Path(infer_config.tomogram_dir).resolve()
    Path(infer_config.predictions_dir).mkdir(parents=True, exist_ok=True)

    data_fnames = []
    patterns = ("*.mrc", "*.rec")
    for pattern in patterns:
        data_fnames.extend(tomosDirPath.glob(pattern))
    data_fnames = sorted(data_fnames)

    accel, dev_count = accelerator_selector(
        use_cuda=infer_config.use_cuda, n_cpus=infer_config.N_CPUS_IF_NO_GPU
    )
    if accel.startswith("cpu"):
        n_gpus = None
    else:
        n_gpus = dev_count
    # Run parallel inference
    results = Parallel(
        n_jobs=infer_config.oversubscribe_factor * dev_count, batch_size=1
    )(
        delayed(infer_tomos)(
            batch_fnames,
            infer_config.predictions_dir,
            infer_config.weights,
            particleLengthAng=infer_config.length,
            gpu_id=(
                (i % infer_config.oversubscribe_factor) % n_gpus
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
        for i, batch_fnames in enumerate(
            batched(data_fnames, n=infer_config.oversubscribe_factor * dev_count)
        )
    )

    if infer_config.save_predicted_coords:
        process_extracted_coordinates(
            results=results,
            output_dir=mainConfig.infer.predictions_dir,
            output_format=mainConfig.infer.predictions_coord_format,
            output_filename=mainConfig.infer.predictions_coord_filename,
        )
