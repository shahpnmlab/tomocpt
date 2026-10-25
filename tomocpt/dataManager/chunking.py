import gc
import json
import os
import random
import shutil
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import torch
import torchio as tio
from dask.distributed import Client, as_completed
from dask_cuda import LocalCUDACluster
from monai.data import PersistentDataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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


def get_chunk_layout_path(chunkedDataDir: str) -> str:
    """Returns the conventional path for the chunk layout manifest."""
    return f"{chunkedDataDir}/chunk_layout.json"


def assign_chunk_dir_names(tomos_df: pd.DataFrame) -> pd.Series:
    """Directory name for each row of the tracking table.

    The tomogram stem when every stem is unique (legacy layout). Otherwise
    "<dataset_dir>__<stem>" for every row, where <dataset_dir> is the name of the
    prepare_data dataset directory (Path(label_path).parents[2].name, e.g.
    "dataset_3_class_all"). Raises ValueError if names are still not unique.
    """
    stems = tomos_df["tomogram_path"].map(lambda p: Path(str(p)).stem)

    if stems.is_unique:
        return stems

    def _dataset_prefix(row) -> str:
        label_path = row.get("label_path")
        if label_path and not pd.isna(label_path) and str(label_path) != str(row["tomogram_path"]):
            try:
                return Path(str(label_path)).parents[2].name
            except IndexError:
                pass
        return f"csv{row.name}"

    dataset_prefixes = tomos_df.apply(_dataset_prefix, axis=1)
    names = dataset_prefixes.str.cat(stems, sep="__")

    n_renamed = int((names != stems).sum())
    logging.info(
        f"{n_renamed} of {len(names)} rows have a tomogram stem shared with another row; "
        "prefixing all chunk directory names with their dataset directory."
    )

    if not names.is_unique:
        dupes = names[names.duplicated()].unique().tolist()
        raise ValueError(
            "Cannot build unique chunk directory names: the following (dataset, tomogram) "
            f"names still collide after prefixing: {dupes}"
        )

    return names


def split_rows_by_tomogram(
    tomos_df: pd.DataFrame, test_size: float = constants.PERCENT_TO_VALIDATE, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/validation split that keeps all rows of one tomogram together.

    Legacy path: if every tomogram stem is unique, return exactly
    train_test_split(tomos_df, test_size=test_size, random_state=seed).
    Otherwise use GroupShuffleSplit(n_splits=1, test_size=test_size,
    random_state=seed) with groups = tomogram stem.
    Raises ValueError with a clear message if there are fewer than 2 distinct
    tomograms (the legacy code fails later with an empty train set).
    """
    stems = tomos_df["tomogram_path"].map(lambda p: Path(str(p)).stem)
    n_distinct = stems.nunique()
    if n_distinct < 2:
        raise ValueError(
            f"Cannot split into train/validation: only {n_distinct} distinct tomogram(s) "
            "found. At least 2 distinct tomograms are required."
        )

    if stems.is_unique:
        return train_test_split(tomos_df, test_size=test_size, random_state=seed)

    dup_stems = stems[stems.duplicated()].unique().tolist()
    if dup_stems:
        logging.warning(
            f"{len(dup_stems)} tomogram stem(s) are shared across datasets and will be "
            f"grouped together for the train/validation split: {dup_stems}"
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(tomos_df, groups=stems))
    return tomos_df.iloc[train_idx], tomos_df.iloc[val_idx]


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

        bn = task.get("chunk_dir_name") or Path(tomo_path_str).stem
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

def do_chunking(tomosDf: pd.DataFrame, chunkedDataDir: str, desired_particle_pixel_size: int,
                n_cpus: int, require_labels: bool = True,
                train_val_level: CrossValidationLevelSplit = CrossValidationLevelSplit.tomos, use_gpus: bool = True):
    try:
        tomosDf = tomosDf.reset_index(drop=True)
        chunk_dir_names = assign_chunk_dir_names(tomosDf)
        dir_names_layout = "unique" if (tomosDf["tomogram_path"].map(lambda p: Path(str(p)).stem)).is_unique else "prefixed"

        if train_val_level == CrossValidationLevelSplit.tomos:
            df_train, df_val = split_rows_by_tomogram(tomosDf, test_size=constants.PERCENT_TO_VALIDATE, seed=42)
            split_layout = "rows" if dir_names_layout == "unique" else "grouped"
            logging.info(
                f"Train split: {sorted(df_train['tomogram_path'].map(lambda p: Path(str(p)).stem).unique().tolist())} "
                f"({len(df_train)} rows). Validation split: "
                f"{sorted(df_val['tomogram_path'].map(lambda p: Path(str(p)).stem).unique().tolist())} ({len(df_val)} rows)."
            )
        else:
            df_train, df_val = tomosDf, None
            split_layout = "rows"
            logging.info(
                "train_on=chunks moves whole (dataset, tomogram) directories between train and "
                "validation after chunking; the same tomogram can end up in both splits by design."
            )

        train_outDir, val_outDir = Path(chunkedDataDir) / "train", Path(chunkedDataDir) / "val"
        if train_outDir.exists(): shutil.rmtree(train_outDir)
        if val_outDir.exists(): shutil.rmtree(val_outDir)
        makedir(train_outDir);
        makedir(val_outDir)

        cache_dir = Path(chunkedDataDir) / ".monai_cache"

        all_tasks: List[Dict[str, Any]] = []
        datasets = {'train': df_train}
        if df_val is not None: datasets['val'] = df_val

        for split, df in datasets.items():
            if df is None: continue
            outpath = train_outDir if split == 'train' else val_outDir
            for idx, row in df.iterrows():
                all_tasks.append({"tomo_info": row, "outpath": str(outpath), "cache_dir": cache_dir,
                                  "require_labels": require_labels,
                                  "desired_particle_pixel_size": desired_particle_pixel_size,
                                  "chunk_dir_name": chunk_dir_names.loc[idx]})

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
            logging.info(
                f"Starting Dask CUDA cluster with {num_workers} GPU workers, each with {threads_per_worker} threads.")
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
        with open(get_chunk_layout_path(chunkedDataDir), "w") as f:
            json.dump({"layout": 2, "dir_names": dir_names_layout, "split": split_layout}, f)
        with open(get_chunking_name_done(chunkedDataDir, require_labels), "w") as f:
            f.write("done")

    except Exception:
        logging.error("A critical error occurred in the main do_chunking function!")
        logging.error(f"Full Traceback:\n{traceback.format_exc()}")
        raise
