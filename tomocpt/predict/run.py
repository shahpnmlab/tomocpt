import os
from pathlib import Path
from typing import Annotated, Dict, Any, List

import torch
import typer
from dask.distributed import Client, as_completed
from dask_cuda import LocalCUDACluster
from omegaconf import DictConfig
from tqdm import tqdm

from tomocpt.predict.helpers import (
    extract_peaks_from_confidence_maps,
    infer_tomos,
    process_extracted_coordinates,
)
from tomocpt.predict.multiclass import (
    parse_class_thresholds,
    peaks_from_saved_maps,
    read_checkpoint_settings,
    selected_classes,
    write_class_outputs,
)
from tomocpt.logger import get_logger
from tomocpt.mainConfig import mainConfig

logger = get_logger()


def _collect_tomograms(infer_config) -> List[Path]:
    """Resolves the list of tomograms to run inference on."""
    data_fnames: List[Path] = []
    if infer_config.tomogram_file:
        file_path = Path(infer_config.tomogram_file).resolve()
        if not file_path.is_file(): raise FileNotFoundError(f"File not found: {file_path}")
        data_fnames.append(file_path)
    elif infer_config.tomogram_dir:
        dir_path = Path(infer_config.tomogram_dir).resolve()
        if not dir_path.is_dir(): raise NotADirectoryError(f"Directory not found: {dir_path}")
        data_fnames.extend(sorted(dir_path.glob("*.mrc")))
        data_fnames.extend(sorted(dir_path.glob("*.rec")))
    else:
        raise ValueError("Must specify either `tomogram_file` or `tomogram_dir` for inference.")
    return data_fnames


def _collect_confidence_maps(infer_config) -> List[Path]:
    """
    Resolves the confidence maps to extract peaks from when inference is skipped.

    If a tomogram file/directory is given, only the maps matching those tomograms
    are used; otherwise every MRC file in `predictions_dir` is taken as a map.
    """
    predictions_dir = Path(infer_config.predictions_dir).resolve()
    if not predictions_dir.is_dir():
        raise NotADirectoryError(
            f"Predictions directory not found: {predictions_dir}. "
            "It must contain the confidence maps from a previous run."
        )

    if infer_config.tomogram_file or infer_config.tomogram_dir:
        map_fnames = []
        for tomo_fname in _collect_tomograms(infer_config):
            map_path = predictions_dir / f"{tomo_fname.stem}.mrc"
            if map_path.is_file():
                map_fnames.append(map_path)
            else:
                logger.warning(
                    f"No confidence map found for {tomo_fname.name} at {map_path}, skipping."
                )
        return map_fnames

    return sorted(predictions_dir.glob("*.mrc"))


def _v2_map_stems(infer_config, settings) -> List[Path]:
    """``skip_inference`` without tomograms (pipeline v2): the tomogram stems of the
    ``<stem>_<class name>.mrc`` maps in ``predictions_dir``."""
    predictions_dir = Path(infer_config.predictions_dir).resolve()
    if not predictions_dir.is_dir():
        raise NotADirectoryError(
            f"Predictions directory not found: {predictions_dir}. "
            "It must contain the confidence maps from a previous run."
        )
    stems = set()
    for row in settings.class_table:
        suffix = f"_{row.name}"
        for path in predictions_dir.glob(f"*{suffix}.mrc"):
            stems.add(path.stem[: -len(suffix)])
    return [predictions_dir / f"{stem}.mrc" for stem in sorted(stems)]


def predict(
    plot: Annotated[bool, typer.Option(help="Plotting not implemented")] = False,
    config: DictConfig = None,
):
    """
    Performs parallel inference on tomogram data using a trained model.
    """
    infer_config = mainConfig.infer
    skip_inference = getattr(infer_config, "skip_inference", False)

    # Pipeline v2 is decided by the checkpoint (its training settings, never mainConfig.train).
    settings = read_checkpoint_settings(infer_config.weights) if infer_config.weights else None
    v2 = settings is not None and settings.pipeline_version == 2
    if v2:
        # Configuration errors surface here, before any worker starts.
        selected_classes(settings.class_table, infer_config.classes)
        parse_class_thresholds(infer_config.class_thresholds, settings.class_table)
        logger.info(
            "Pipeline-v2 checkpoint with classes "
            f"{[row.name for row in settings.class_table]}; `infer.length` is not used."
        )
    elif infer_config.length is None:
        raise ValueError("`infer.length` (particle diameter in Angstroms) is required for this checkpoint.")

    if skip_inference and v2:
        if not infer_config.save_predicted_coords:
            raise ValueError(
                "`skip_inference` requires `save_predicted_coords` to be enabled, "
                "otherwise there is nothing left to do."
            )
        tomo_fnames = _collect_tomograms(infer_config) if (
            infer_config.tomogram_file or infer_config.tomogram_dir) else _v2_map_stems(infer_config, settings)
        results = peaks_from_saved_maps(tomo_fnames, settings, infer_config)
        if results:
            write_class_outputs(results, settings.class_table, infer_config)
        else:
            logger.warning(f"No per-class confidence maps found in {infer_config.predictions_dir}.")
        return

    if skip_inference:
        if not infer_config.save_predicted_coords:
            raise ValueError(
                "`skip_inference` requires `save_predicted_coords` to be enabled, "
                "otherwise there is nothing left to do."
            )
        data_fnames = _collect_confidence_maps(infer_config)
        if not data_fnames:
            logger.warning(
                f"No confidence maps found in {infer_config.predictions_dir}. "
                "They are only written when a previous run had "
                "`save_prediction_confidence_map` enabled."
            )
            return
        logger.info(
            f"Skipping inference. Extracting peaks from {len(data_fnames)} existing "
            f"confidence map(s) in {infer_config.predictions_dir}."
        )
    else:
        if not infer_config.weights:
            raise ValueError(
                "Must specify `weights` for inference (or use `skip_inference` to "
                "extract peaks from existing confidence maps)."
            )
        data_fnames = _collect_tomograms(infer_config)
        if not data_fnames:
            logger.warning("No tomogram files to process.")
            return

    Path(infer_config.predictions_dir).mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count() if infer_config.use_cuda and torch.cuda.is_available() else 0

    if n_gpus > 0:
        num_workers = n_gpus
        threads_per_worker = max(1, (infer_config.cpus_per_worker or 1))
        cluster = LocalCUDACluster(n_workers=num_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
        logger.info(f"Starting Dask CUDA cluster for inference with {num_workers} GPU workers.")
    else:
        from dask.distributed import LocalCluster
        num_workers = min((os.cpu_count() or 1), len(data_fnames), (infer_config.cpus_per_worker or os.cpu_count()))
        cluster = LocalCluster(n_workers=num_workers)
        client = Client(cluster)
        logger.info(f"Starting Dask cluster for inference with {num_workers} CPU workers.")

    # Distribute the list of tomogram files among the workers
    tasks_per_worker = [[] for _ in range(num_workers)]
    for i, fname in enumerate(data_fnames):
        tasks_per_worker[i % num_workers].append(fname)
        
    futures = []
    for i in range(num_workers):
        if tasks_per_worker[i]:
            # Each future represents one worker processing its batch of files
            gpu_id = i if n_gpus > 0 else None
            # Pass the unique worker_id (i) to the worker function for positioning the progress bar
            if skip_inference:
                future = client.submit(extract_peaks_from_confidence_maps, tasks_per_worker[i], gpu_id, i, infer_config)
            else:
                future = client.submit(infer_tomos, tasks_per_worker[i], gpu_id, i, infer_config.weights, infer_config)
            futures.append(future)

    # The main progress bar is removed to reduce clutter.
    # Progress is now shown by the individual worker progress bars.
    results = []
    logger.info(f"Waiting for {len(futures)} worker batches to complete...")
    try:
        for future in as_completed(futures):
            worker_result_list = future.result()
            if worker_result_list:
                results.extend(worker_result_list)
        logger.info("All worker batches completed.")
    finally:
        client.close()
        cluster.close()

    if v2:
        if infer_config.save_predicted_coords and results:
            write_class_outputs(results, settings.class_table, infer_config)
        return

    if infer_config.save_predicted_coords and results:
        logger.info("Aggregating and saving extracted coordinates...")
        all_tomo_names, all_centroids, all_voxel_sizes = [], [], []
        
        # The 'results' list contains dictionaries, each for a single tomogram's output.
        for res_dict in results:
            tomo_name = res_dict["name"]
            voxel_size = res_dict["voxel_size"]
            # A single tomogram can have multiple coordinates (centroids).
            for centroid in res_dict["coords"]:
                all_tomo_names.append(tomo_name)
                all_centroids.append(centroid)
                all_voxel_sizes.append(voxel_size)
            
        if all_tomo_names:
            process_extracted_coordinates(
                output_dir=infer_config.predictions_dir,
                tomo_names=all_tomo_names,
                predicted_centroids_with_scores=all_centroids,
                voxel_sizes=all_voxel_sizes,
                output_format=infer_config.predictions_coord_format,
                output_filename=infer_config.predictions_coord_filename,
            )
