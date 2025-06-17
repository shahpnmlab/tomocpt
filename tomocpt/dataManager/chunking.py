import gc
import os
import random
import shutil
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch
import torchio as tio
from dask.distributed import Client, as_completed
from dask_cuda import LocalCUDACluster
from monai.data import PersistentDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tomocpt import constants
from tomocpt.dataManager.transforms import PreprocessTomogramd
from tomocpt.defaultConfigs.train_config import CrossValidationLevelSplit
from tomocpt.mainConfig import mainConfig
from tomocpt.utils import makedir
from tomocpt.logger import get_logger

logging = get_logger()

warnings.filterwarnings("ignore", ".*Using TorchIO images without a torchio.SubjectsLoader.*", UserWarning)
warnings.filterwarnings("ignore", ".*You are using `torch.load` with `weights_only=False`.*", FutureWarning)


def get_chunking_name_done(chunkedDataDir: str, require_labels: bool) -> str:
    """Returns the conventional path for the 'done' file."""
    return f"{chunkedDataDir}/done_labels_{require_labels}.txt"


def _save_chunk(tensor: torch.Tensor, path: str):
    """Saves a 4D tensor to disk as a NIfTI file."""
    img = tio.ScalarImage(tensor=tensor.cpu().float())
    img.save(path)


def process_single_tomo_task(task: Dict[str, Any]) -> tuple:
    """
    This is the core task function. It includes a try-except block to fall
    back to CPU processing for operations that cause CUDA OOM errors.
    """
    try:
        tomo_info, outpath, cache_dir, require_labels = (
            task["tomo_info"], task["outpath"], task["cache_dir"], task["require_labels"]
        )
        tomo_path_str = str(tomo_info["tomogram_path"])
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_list = [{"tomo": tomo_path_str, 
                      "label": tomo_info["label_path"] if require_labels else tomo_path_str, 
                      "diameter": tomo_info["particle_diameter_angst"]}]
                      
        transform_pipeline = PreprocessTomogramd(tomo_key="tomo", label_key="label", particle_diameter_key="diameter", device=device)
        
        p_dataset = PersistentDataset(data=data_list, transform=transform_pipeline, cache_dir=cache_dir)
        preprocessed_data = p_dataset[0]
        
        vol_tensor_cpu = preprocessed_data["tomo"]
        label_tensor_cpu = preprocessed_data["label"]
        del preprocessed_data

        bn = Path(tomo_path_str).stem
        tomo_out_dir = Path(outpath) / bn
        vol_dir = tomo_out_dir / constants.VOLUMES_DIR_NAME_PREFIX
        labels_dir_name = constants.LABELS_DIR_NAME_PREFIX % ('_supervised' if require_labels else '_selfSup')
        labels_dir = tomo_out_dir / labels_dir_name
        makedir(vol_dir); makedir(labels_dir)

        try:
            # First, attempt all memory-intensive work on the GPU for speed.
            vol_tensor_gpu = vol_tensor_cpu.to(device)
            label_tensor_gpu = label_tensor_cpu.to(device)

            drop_probability = 0.0
            if require_labels and not (tomo_path_str == tomo_info["label_path"]):
                f0 = (label_tensor_gpu == 0).sum()
                f1 = label_tensor_gpu.numel() - f0
                if f0 > 0:
                    drop_probability = torch.clip(1 - mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES * (f1 / f0), min=0.0, max=1.0).item()
            
            subject = tio.Subject(volume=tio.ScalarImage(tensor=vol_tensor_gpu.unsqueeze(0)), 
                                  label=tio.LabelMap(tensor=label_tensor_gpu.unsqueeze(0)))
            del vol_tensor_gpu, label_tensor_gpu

        except torch.cuda.OutOfMemoryError:
            logging.warning(
                f"CUDA out of memory for {tomo_path_str}. "
                "Falling back to CPU for chunking this tomogram. This will be slower."
            )
            gc.collect()
            torch.cuda.empty_cache()

            drop_probability = 0.0
            if require_labels and not (tomo_path_str == tomo_info["label_path"]):
                f0 = (label_tensor_cpu == 0).sum()
                f1 = label_tensor_cpu.numel() - f0
                if f0 > 0:
                    drop_probability = torch.clip(1 - mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES * (f1 / f0), min=0.0, max=1.0).item()
            
            subject = tio.Subject(volume=tio.ScalarImage(tensor=vol_tensor_cpu.unsqueeze(0)), 
                                  label=tio.LabelMap(tensor=label_tensor_cpu.unsqueeze(0)))
            del vol_tensor_cpu, label_tensor_cpu

        grid_sampler = tio.GridSampler(subject, patch_size=mainConfig.train.CHUNK_SIZE,
                                       patch_overlap=mainConfig.train.CHUNK_SIZE - mainConfig.train.CHUNK_STRIDE)

        dataloader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        
        n_threads = os.cpu_count() or 1
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i, patch in enumerate(dataloader):
                chunk_data, chunk_target = patch['volume'][tio.DATA].squeeze(0), patch['label'][tio.DATA].squeeze(0)
                if require_labels and chunk_target.sum() == 0 and random.random() < drop_probability:
                    continue
                
                coords = patch[tio.LOCATION][0, :3].tolist()
                chunk_data_clone = chunk_data.clone().detach()
                chunk_target_clone = chunk_target.clone().detach()
                
                vol_path = str(vol_dir / f"{constants.VOLUMES_DIR_NAME_PREFIX}_{i}_{coords[0]}_{coords[1]}_{coords[2]}.nii.gz")
                label_path = str(labels_dir / f"{labels_dir_name}_{i}_{coords[0]}_{coords[1]}_{coords[2]}.nii.gz")
                
                futures.append(executor.submit(_save_chunk, chunk_data_clone, vol_path))
                futures.append(executor.submit(_save_chunk, chunk_target_clone, label_path))

            n_cubes_written = len(futures) // 2
        
        return tomo_path_str, n_cubes_written
    
    except Exception:
        logging.error(f"Dask worker processing {task.get('tomo_info', {}).get('tomogram_path', 'unknown file')} FAILED!")
        logging.error(f"Full Traceback:\n{traceback.format_exc()}")
        return None, None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def do_chunking(tomosDf: pd.DataFrame, chunkedDataDir: str,
                n_cpus: int, require_labels: bool = True,
                train_val_level: CrossValidationLevelSplit = CrossValidationLevelSplit.tomos, use_gpus: bool = True):
    try:
        if train_val_level == CrossValidationLevelSplit.tomos:
            df_train, df_val = train_test_split(tomosDf, test_size=constants.PERCENT_TO_VALIDATE, random_state=42)
        else:
            df_train, df_val = tomosDf, None

        train_outDir, val_outDir = Path(chunkedDataDir) / "train", Path(chunkedDataDir) / "val"
        if train_outDir.exists(): shutil.rmtree(train_outDir)
        if val_outDir.exists(): shutil.rmtree(val_outDir)
        makedir(train_outDir); makedir(val_outDir)

        cache_dir = Path(chunkedDataDir) / ".monai_cache"
        
        all_tasks: List[Dict[str, Any]] = []
        datasets = {'train': df_train}
        if df_val is not None: datasets['val'] = df_val

        for split, df in datasets.items():
            if df is None: continue
            outpath = train_outDir if split == 'train' else val_outDir
            for _, row in df.iterrows():
                all_tasks.append({"tomo_info": row, "outpath": str(outpath), "cache_dir": cache_dir,
                                  "require_labels": require_labels})
        
        if not all_tasks:
            logging.warning("No tomograms found to process. Stopping data preparation.")
            return

        n_gpus = torch.cuda.device_count() if use_gpus and torch.cuda.is_available() else 0
        
        if n_cpus <= 0:
            logging.warning(f"Configured n_cpus ({n_cpus}) is not positive. Defaulting to all available CPUs.")
            n_cpus = os.cpu_count()

        if n_gpus > 0:
            num_workers = n_gpus
            threads_per_worker = max(1, n_cpus // n_gpus)
            
            cluster = LocalCUDACluster(
                n_workers=num_workers,
                threads_per_worker=threads_per_worker
            )
            client = Client(cluster)
            logging.info(f"Starting Dask CUDA cluster with {num_workers} GPU workers, each with {threads_per_worker} threads.")
        else:
            from dask.distributed import LocalCluster
            num_workers = min(n_cpus, len(all_tasks))
            cluster = LocalCluster(n_workers=num_workers)
            client = Client(cluster)
            logging.info(f"Starting Dask cluster with {num_workers} CPU workers.")

        futures = [client.submit(process_single_tomo_task, task) for task in all_tasks]

        with tqdm(total=len(all_tasks), desc="Chunking Tomograms") as pbar:
            for future, result in as_completed(futures, with_results=True):
                tomo_path, n_cubes = result
                if tomo_path is not None:
                    pbar.set_description(f"Processed {Path(tomo_path).name} ({n_cubes} cubes)")
                else:
                    pbar.set_description("A worker failed, see logs for details")
                pbar.update()

        client.close()
        cluster.close()
        
        if train_val_level == CrossValidationLevelSplit.cubes:
            all_tomo_dirs = [d for d in train_outDir.iterdir() if d.is_dir()]
            _, val_dirs = train_test_split(all_tomo_dirs, test_size=constants.PERCENT_TO_VALIDATE, random_state=42)
            for d in val_dirs: shutil.move(str(d), val_outDir)

        logging.info(f"Chunking finished. Data saved at: {train_outDir} and {val_outDir}")
        with open(get_chunking_name_done(chunkedDataDir, require_labels), "w") as f: f.write("done")

    except Exception:
        logging.error("A critical error occurred in the main do_chunking function!")
        logging.error(f"Full Traceback:\n{traceback.format_exc()}")
        raise
