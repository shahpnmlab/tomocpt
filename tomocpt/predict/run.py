import warnings
import os
from pathlib import Path
from typing import Annotated

import torch
import typer
from dask.distributed import Client, as_completed, LocalCluster
from dask_cuda import LocalCUDACluster
from omegaconf import DictConfig
from tqdm import tqdm

from tomocpt.predict.helpers import infer_tomos, process_extracted_coordinates
from tomocpt.logger import get_logger
from tomocpt.mainConfig import mainConfig

logger = get_logger()

warnings.filterwarnings(
    "ignore",
    message=".*SwinUNETR.*: Argument `img_size` has been deprecated.*",
    category=FutureWarning
)
# Filter the specific TorchIO UserWarning about SubjectsLoader
warnings.filterwarnings(
    "ignore",
    message=".*Using TorchIO images without a torchio.SubjectsLoader.*",
    category=UserWarning
)

def predict(plot: Annotated[bool, typer.Option(help="Plotting not implemented")] = False, config: DictConfig = None):
    """
    Performs parallel inference on tomogram data using a trained model.
    Can process a single tomogram file or all tomograms in a directory.
    """
    infer_config = mainConfig.infer
    data_fnames = []
    if infer_config.tomogram_file:
        file_path = Path(infer_config.tomogram_file).resolve()
        if not file_path.is_file(): raise FileNotFoundError(f"File not found: {file_path}")
        data_fnames.append(file_path)
    elif infer_config.tomogram_dir:
        dir_path = Path(infer_config.tomogram_dir).resolve()
        if not dir_path.is_dir(): raise NotADirectoryError(f"Directory not found: {dir_path}")
        data_fnames.extend(sorted(p for p in dir_path.iterdir() if p.suffix in ['.mrc', '.rec']))
    else:
        raise ValueError("Must specify `tomogram_file` or `tomogram_dir` for inference.")

    if not data_fnames:
        logger.warning("No tomogram files found to process.")
        return

    Path(infer_config.predictions_dir).mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count() if infer_config.use_cuda and torch.cuda.is_available() else 0
    if n_gpus > 0:
        num_workers = n_gpus
        threads_per_worker = infer_config.cpus_per_worker
        cluster = LocalCUDACluster(n_workers=num_workers, threads_per_worker=threads_per_worker)
        logger.info(
            f"Starting Dask CUDA cluster with {num_workers} GPU workers, each supported by {threads_per_worker} CPU threads.")
    else:
        num_workers = min(infer_config.N_CPUS_IF_NO_GPU, os.cpu_count() or 1, len(data_fnames))
        num_workers = max(1, num_workers)
        threads_per_worker = infer_config.cpus_per_worker
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=threads_per_worker)
        logger.info(f"Starting Dask CPU cluster with {num_workers} workers, each with {threads_per_worker} threads.")

    client = Client(cluster)

    per_worker_batch_size = max(1, (len(data_fnames) + num_workers - 1) // num_workers)
    batched_fnames = [data_fnames[i:i + per_worker_batch_size] for i in range(0, len(data_fnames), per_worker_batch_size)]

    futures = []
    for i, fnames_batch in enumerate(batched_fnames):
        gpu_id = i % n_gpus if n_gpus > 0 else None
        future = client.submit(
            infer_tomos,
            tomo_fnames=list(fnames_batch),
            gpu_id=gpu_id,
            model_fname=infer_config.weights,
            infer_config=infer_config
        )
        futures.append(future)

    results = []
    with tqdm(total=len(futures), desc="Processing Tomogram Batches") as pbar:
        for future, result in as_completed(futures, with_results=True):
            if result and result[0]:
                results.append(result)
            else:
                logger.warning("A worker completed with no results. Check logs for errors.")
            pbar.update()

    client.close()
    cluster.close()

    if infer_config.save_predicted_coords and results:
        logger.info("Aggregating and saving all coordinates...")
        all_names, all_coords, all_vxs = [], [], []
        for names, coords, vxs in results:
            all_names.extend(names)
            all_coords.extend(coords)
            all_vxs.extend(vxs)

        if all_names:
            process_extracted_coordinates(
                output_dir=infer_config.predictions_dir,
                tomo_names=all_names,
                predicted_centroids_with_scores=all_coords,
                voxel_sizes=all_vxs,
                output_format=infer_config.predictions_coord_format,
                output_filename=infer_config.predictions_coord_filename
            )
    elif infer_config.save_predicted_coords:
        logger.info("Inference finished, but no coordinates were extracted.")