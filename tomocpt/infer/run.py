import typer
from click.core import batch

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
    """Enhanced prediction with improved GPU distribution"""
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

    # Determine available devices
    accel, dev_count = accelerator_selector(
        use_cuda=infer_config.use_cuda, n_cpus=infer_config.N_CPUS_IF_NO_GPU
    )

    # Proper GPU assignment
    if accel.startswith("cpu"):
        n_gpus = None
        use_cuda = False
    else:
        n_gpus = dev_count
        use_cuda = True
        print(f"Using {n_gpus} GPUs for processing")

    # Optimize batch size for parallelization
    batch_size = max(1, len(data_fnames) // (infer_config.oversubscribe_factor * (dev_count or 1)))

    # Create tasks with appropriate GPU assignments
    tasks = []
    for i, batch_start in enumerate(range(0, len(data_fnames), batch_size)):
        batch_fnames = data_fnames[batch_start:batch_start + batch_size]

        # Apply the requested GPU distribution formula with proper exception handling
        if use_cuda and n_gpus:
            if infer_config.oversubscribe_factor > 0:
                gpu_id = (i // infer_config.oversubscribe_factor) % n_gpus
            else:
                gpu_id = (i % n_gpus)
        else:
            gpu_id = None

        tasks.append((batch_fnames, gpu_id))

    # Run parallel inference with proper device handling
    results = Parallel(n_jobs=len(tasks), batch_size=1)(
        delayed(infer_tomos)(
            batch_fnames,
            infer_config.predictions_dir,
            infer_config.weights,
            particleLengthAng=infer_config.length,
            gpu_id=task_gpu_id,  # Pass the GPU ID to each task
            batch_size=infer_config.predictions_batch_size,
            plot=plot,
            save_pred_mask=infer_config.save_prediction_confidence_map,
            extract_coords=infer_config.save_predicted_coords,
            nearest_neigs_angs=infer_config.distance_threshold,
            threshold=infer_config.confidence_threshold,
            masksDir=infer_config.masks_dir,
        )
        for batch_fnames, task_gpu_id in tasks
    )

    # Process extracted coordinates
    if infer_config.save_predicted_coords:
        process_extracted_coordinates(
            results=results,
            output_dir=mainConfig.infer.predictions_dir,
            output_format=mainConfig.infer.predictions_coord_format,
            output_filename=mainConfig.infer.predictions_coord_filename,
        )