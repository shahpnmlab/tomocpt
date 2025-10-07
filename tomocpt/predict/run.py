import os
from pathlib import Path
from typing import Annotated, Dict, Any, List

import torch
import typer
from dask.distributed import Client, as_completed
from dask_cuda import LocalCUDACluster
from omegaconf import DictConfig
from tqdm import tqdm

from tomocpt.predict.helpers import infer_tomos, process_extracted_coordinates
from tomocpt.logger import get_logger
from tomocpt.mainConfig import mainConfig

logger = get_logger()


def predict(
    plot: Annotated[bool, typer.Option(help="Plotting not implemented")] = False,
    config: DictConfig = None,
):
    """
    Performs parallel inference on tomogram data using a trained model.
    """
    infer_config = mainConfig.infer

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
            # Pass the unique worker_id (i) to the infer_tomos function for positioning the progress bar
            future = client.submit(infer_tomos, tasks_per_worker[i], gpu_id, i, infer_config.weights, infer_config)
            futures.append(future)

    # The main progress bar is removed to reduce clutter.
    # Progress is now shown by the individual worker progress bars.
    results = []
    logger.info(f"Waiting for {len(futures)} worker batches to complete...")
    for future in as_completed(futures):
        worker_result_list = future.result()
        if worker_result_list:
            results.extend(worker_result_list)
    logger.info("All worker batches completed.")

    client.close()
    cluster.close()

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
