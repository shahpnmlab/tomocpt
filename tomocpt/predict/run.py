import os
from itertools import repeat
from pathlib import Path
from typing import Annotated

import torch
import torch.multiprocessing as mp
import typer
from more_itertools import batched
from omegaconf import DictConfig
from tqdm import tqdm

from tomocpt.infer.helpers import process_extracted_coordinates, infer_tomos
from tomocpt.logger import get_logger
from tomocpt.mainConfig import mainConfig

logger = get_logger()


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
        data_fnames.extend(sorted(dir_path.glob("*.mrc")))
        data_fnames.extend(sorted(dir_path.glob("*.rec")))
    else:
        raise ValueError("Must specify `tomogram_file` or `tomogram_dir` for inference.")

    if not data_fnames: logger.warning("No tomogram files to process."); return
    Path(infer_config.predictions_dir).mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count() if infer_config.use_cuda and torch.cuda.is_available() else 0
    num_workers = min(n_gpus, len(data_fnames)) if n_gpus > 0 else min(infer_config.N_CPUS_IF_NO_GPU, os.cpu_count(), len(data_fnames))
    if num_workers == 0: num_workers = 1

    per_worker_batch_size = max(1, (len(data_fnames) + num_workers - 1) // num_workers)
    batched_fnames = list(batched(data_fnames, n=per_worker_batch_size))
    
    args_for_pool = list(zip(
        batched_fnames,
        [i % n_gpus if n_gpus > 0 else None for i in range(len(batched_fnames))],
        repeat(infer_config.weights), repeat(infer_config.length), repeat(infer_config.predictions_dir),
        repeat(infer_config.save_prediction_confidence_map), repeat(infer_config.save_predicted_coords),
        repeat(infer_config.confidence_threshold), repeat(infer_config.distance_threshold),
        repeat(infer_config.masks_dir)))

    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(processes=num_workers) as pool, tqdm(total=len(batched_fnames), desc="Processing Batches") as pbar:
        for result in pool.starmap(infer_tomos, args_for_pool):
            results.append(result); pbar.update()

    if infer_config.save_predicted_coords:
        logger.info("Aggregating and saving coordinates...")
        all_names, all_coords, all_vxs = [], [], []
        for names, coords, vxs in results:
            all_names.extend(names); all_coords.extend(coords); all_vxs.extend(vxs)
        if all_names:
            process_extracted_coordinates(
                output_dir=infer_config.predictions_dir, tomo_names=all_names,
                predicted_centroids_with_scores=all_coords, voxel_sizes=all_vxs,
                output_format=infer_config.predictions_coord_format,
                output_filename=infer_config.predictions_coord_filename)
