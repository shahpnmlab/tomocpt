import gc
import json
import os
import random
import shutil
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
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
from tomocpt.defaultConfigs.scale_config import ScaleConfig
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
        # Every value comes from the task (T2.6): this runs in a Dask worker process whose
        # mainConfig is a fresh default, not the caller's config. do_chunking always fills
        # these keys; the global config is only a fallback for task dicts built by older callers.
        target_px = task.get("desired_particle_pixel_size")
        if target_px is None:
            target_px = mainConfig.prepData.desired_particle_pixel_size
        chunk_size = int(task.get("chunk_size") or mainConfig.train.CHUNK_SIZE)
        chunk_stride = int(task.get("chunk_stride") or mainConfig.train.CHUNK_STRIDE)
        alpha = task.get("alpha_drop_empty")
        alpha = float(mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES if alpha is None else alpha)
        tomo_path_str = str(tomo_info["tomogram_path"])
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # target_px and chunk_size are part of the cached item, so the MONAI cache key changes
        # with them (the transform reads them from its constructor arguments).
        data_list = [{"tomo": tomo_path_str, 
                      "label": tomo_info["label_path"] if require_labels else tomo_path_str, 
                      "diameter": tomo_info["particle_diameter_angst"],
                      "target_px": target_px, "chunk_size": chunk_size}]
                      
        transform_pipeline = PreprocessTomogramd(tomo_key="tomo", label_key="label", particle_diameter_key="diameter",
                                                 device=device, target_px=target_px, chunk_size=chunk_size)
        
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
                    drop_probability = torch.clip(1 - alpha * (f1 / f0), min=0.0, max=1.0).item()
            
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
                    drop_probability = torch.clip(1 - alpha * (f1 / f0), min=0.0, max=1.0).item()
            
            subject = tio.Subject(volume=tio.ScalarImage(tensor=vol_tensor_cpu.unsqueeze(0)), 
                                  label=tio.LabelMap(tensor=label_tensor_cpu.unsqueeze(0)))
            del vol_tensor_cpu, label_tensor_cpu

        grid_sampler = tio.GridSampler(subject, patch_size=chunk_size,
                                       patch_overlap=chunk_size - chunk_stride)

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
                train_val_level: CrossValidationLevelSplit = CrossValidationLevelSplit.tomos, use_gpus: bool = True,
                pipeline_version: int = 1, scale_cfg: Optional[ScaleConfig] = None,
                class_table: Optional[list] = None, seed: int = 42,
                alpha_drop_empty: Optional[float] = None, chunk_size: Optional[int] = None,
                chunk_stride: Optional[int] = None, label_border_policy: str = "skip"):
    """Chunk the tomograms of ``tomosDf`` into ``chunkedDataDir/{train,val}``.

    Defaults run the pipeline v1 code path unchanged. ``pipeline_version=2`` runs pipeline
    v2 (band scale policy, multi-class points; see :func:`_do_chunking_v2`) and needs
    ``class_table``. ``alpha_drop_empty``, ``chunk_size`` and ``chunk_stride`` default to the
    caller's config, resolved here, before any task is submitted (T2.6): workers never read
    ``mainConfig``. ``label_border_policy`` is used by pipeline v2 only (v1 labels were drawn
    by ``prepare_data``).
    """
    if alpha_drop_empty is None:
        alpha_drop_empty = mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES
    if chunk_size is None:
        chunk_size = mainConfig.train.CHUNK_SIZE
    if chunk_stride is None:
        chunk_stride = mainConfig.train.CHUNK_STRIDE
    chunk_size, chunk_stride, alpha_drop_empty = int(chunk_size), int(chunk_stride), float(alpha_drop_empty)
    if not 0 < chunk_stride <= chunk_size:
        raise ValueError(f"chunk_stride must be in (0, chunk_size], got {chunk_stride} for chunk_size {chunk_size}")

    if int(pipeline_version) == 2:
        return _do_chunking_v2(
            tomosDf, chunkedDataDir, n_cpus=n_cpus, require_labels=require_labels,
            train_val_level=train_val_level, use_gpus=use_gpus,
            scale_cfg=scale_cfg if scale_cfg is not None else ScaleConfig(),
            class_table=class_table, seed=seed, alpha_drop_empty=alpha_drop_empty,
            chunk_size=chunk_size, chunk_stride=chunk_stride, label_border_policy=label_border_policy)
    if int(pipeline_version) != 1:
        raise ValueError(f"Unknown pipeline_version {pipeline_version} (expected 1 or 2)")

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
                                  "chunk_size": chunk_size, "chunk_stride": chunk_stride,
                                  "alpha_drop_empty": alpha_drop_empty,
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


# ==========================================================================================
# Pipeline v2 (Gate 2): band scale policy, levels, multi-class points
# ==========================================================================================

V2_POINTS_DIR_NAME = "points"
V2_POINTS_PREFIX = "points"


def _stable_seed(*parts) -> int:
    """A seed that depends only on ``parts`` (not on the process, the worker or the order)."""
    import hashlib

    digest = hashlib.sha256(repr(tuple(str(p) for p in parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _row_dataset_name(row) -> str:
    for key in ("coords_path", "label_path"):
        p = row.get(key) if hasattr(row, "get") else None
        if p is not None and not (isinstance(p, float) and pd.isna(p)):
            try:
                return Path(str(p)).parents[1 if key == "coords_path" else 2].name
            except IndexError:
                continue
    return ""


def _dataset_ids(df: pd.DataFrame) -> pd.Series:
    """Index of the prepare_data dataset of each row: ``dataset_<k>_...`` -> k - 1, otherwise
    the rank of the dataset directory name."""
    import re

    names = df.apply(_row_dataset_name, axis=1)
    matches = names.map(lambda n: re.match(r"dataset_(\d+)(_|$)", n))
    if len(names) and matches.map(lambda m: m is not None).all():
        return matches.map(lambda m: int(m.group(1)) - 1).astype(int)
    order = {n: i for i, n in enumerate(sorted(names.unique()))}
    return names.map(order).astype(int)


def _read_coords(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"x", "y", "z", "class_number"} - set(df.columns)
    if missing:
        raise ValueError(f"Coordinate file {path} lacks columns {sorted(missing)}")
    return df


def _annotated_numbers(df: pd.DataFrame, class_table) -> List[List[int]]:
    """Exhaustively annotated class numbers per row: the ``class_annotated`` column when
    present, otherwise every class present in the row's dataset (all its coordinate files)."""
    from tomocpt.labels.multiclass import parse_class_annotated

    ds_names = df.apply(_row_dataset_name, axis=1)
    present: Dict[str, set] = {}
    for name, coords_path in zip(ds_names, df["coords_path"]):
        present.setdefault(name, set()).update(int(c) for c in _read_coords(coords_path)["class_number"])
    out = []
    for i, (_, row) in enumerate(df.iterrows()):
        listed = parse_class_annotated(row.get("class_annotated"), class_table) if "class_annotated" in df.columns else None
        out.append(sorted(listed) if listed is not None else sorted(present[ds_names.iloc[i]]))
    return out


def _merge_rows_v2(tomosDf: pd.DataFrame, class_table) -> pd.DataFrame:
    """One row per tomogram file: the coordinate files of every row that refers to it, the
    smallest dataset id, and the classes annotated in any of its datasets."""
    from tomocpt.dataManager.scaling import read_apix

    df = tomosDf.reset_index(drop=True).copy()
    if "coords_path" not in df.columns:
        raise ValueError(
            "Pipeline v2 needs the coordinate CSVs written by prepare_data (tracking column "
            "'coords_path'). Re-run `tomocpt prepare_data` with this version.")
    df["_dataset_id"] = _dataset_ids(df)
    df["_annotated"] = _annotated_numbers(df, class_table)
    df["_key"] = df["tomogram_path"].map(lambda p: str(Path(str(p)).resolve()))
    numbers = [int(r.number) for r in class_table]

    rows = []
    for key, g in df.groupby("_key", sort=False):
        first = g.iloc[0]
        apix = None
        if "apix_native" in g.columns:
            vals = [float(v) for v in g["apix_native"] if not pd.isna(v) and float(v) > 0]
            apix = vals[0] if vals else None
            if vals and any(abs(v - vals[0]) > 1e-6 * vals[0] for v in vals):
                raise ValueError(f"Rows of {first['tomogram_path']} disagree on apix_native: {vals}")
        if apix is None:
            apix = read_apix(first["tomogram_path"])
        annotated = set()
        for a in g["_annotated"]:
            annotated.update(a)
        rows.append({
            "tomogram_path": str(first["tomogram_path"]),
            "label_path": first.get("label_path", str(first["tomogram_path"])),
            "coords_paths": [str(p) for p in g["coords_path"]],
            "dataset_id": int(g["_dataset_id"].min()),
            "apix_native": float(apix),
            "class_available": [n in annotated for n in numbers],
        })
    if len(rows) < len(df):
        logging.info(f"Pipeline v2: {len(df)} tracking rows refer to {len(rows)} tomogram files; "
                     "rows of the same tomogram are chunked together.")
    return pd.DataFrame(rows)


def _save_points(arr: np.ndarray, path: str):
    np.save(path, np.ascontiguousarray(arr, dtype=np.float32))


def process_single_tomo_task_v2(task: Dict[str, Any]) -> Dict[str, Any]:
    """Chunk one tomogram with pipeline v2 (every level and scale index).

    Every value comes from ``task`` (T2.6). Returns ``{"tomo", "n_chunks", "dirs", "error"}``.
    """
    from tomocpt.dataManager.metadata import ChunkMeta, write_chunk_meta
    from tomocpt.dataManager.preprocessing import load_mrc, symmetrize_and_pad
    from tomocpt.dataManager.scaling import (coordinate_map, in_band, jitter_range, plan_levels,
                                             resample_volume, working_shape)
    from tomocpt.labels.helpers import points_to_working, render_targets, touches_border
    from tomocpt.labels.multiclass import class_table_from_json

    tomo_path = str(task["tomogram_path"])
    try:
        table = class_table_from_json(task["class_table"])
        n_classes = len(table)
        diam = np.array([r.diameter_ang for r in table], dtype=np.float64)
        band_min, band_max = float(task["band_min"]), float(task["band_max"])
        apix_native = float(task["apix_native"])
        chunk_size, chunk_stride = int(task["chunk_size"]), int(task["chunk_stride"])
        alpha = float(task["alpha_drop_empty"])
        seed, name = int(task["seed"]), str(task["chunk_dir_name"])
        annotated = np.array(task["class_available"], dtype=bool)
        storage = str(task["target_storage"])
        resample = str(task["resample"])
        device = torch.device("cuda:0" if task["use_cuda"] and torch.cuda.is_available() else "cpu")

        coords = [_read_coords(p) for p in task["coords_paths"]]
        coords_df = pd.concat(coords, ignore_index=True) if coords else pd.DataFrame(
            columns=["x", "y", "z", "class_number", "diameter_ang"])
        vol_np = load_mrc(tomo_path, normalize=True)  # robust normalisation at native size (v1)
        native_shape = tuple(int(s) for s in vol_np.shape)

        levels = plan_levels(apix_native, table, band_min, band_max)
        n_written, dirs = 0, []
        for li, lv in enumerate(levels):
            apixes = [float(lv.apix_work)]
            n_scales = int(task["scales_per_level"])
            if n_scales > 1:
                lo, hi = jitter_range(apix_native, diam[lv.assigned], band_min, band_max)
                if hi > lo * (1.0 + 1e-6):
                    rng = np.random.default_rng(_stable_seed(seed, name, li, "scales"))
                    apixes += [float(v) for v in np.exp(rng.uniform(np.log(lo), np.log(hi), size=n_scales - 1))]
                else:
                    logging.info(f"{name}, level {li}: no room to jitter the voxel size "
                                 f"(range [{lo:.4g}, {hi:.4g}] A); only scale index 0 is written.")
            for k, apix_work in enumerate(apixes):
                available = np.array([bool(annotated[c]) and in_band(diam[c], apix_native, apix_work, band_min,
                                                                    band_max) for c in range(n_classes)])
                if not available.any():
                    logging.info(f"{name}, level {li} scale {k} ({apix_work:.4g} A): no annotated class is "
                                 "inside the band here; nothing written.")
                    continue
                ws = tuple(working_shape(native_shape, apix_native, apix_work))
                vol_t = torch.from_numpy(vol_np)
                if ws != native_shape:
                    vol_t = resample_volume(vol_t.to(device), ws, resample).cpu()
                # Exact map of the resample actually run (identity when none ran).
                cmap = coordinate_map(native_shape, ws, resample)
                padded, padding = symmetrize_and_pad(vol_t.float(), chunk_size)
                padded = padded.contiguous()
                pad_front = np.array([padding[4], padding[2], padding[0]], dtype=np.float64)
                shape = tuple(int(s) for s in padded.shape)

                # Targets as points: working voxels, shifted by the padding, border policy once.
                pts = np.asarray(points_to_working(coords_df, apix_native, apix_work, table, coord_map=cmap),
                                 dtype=np.float64).copy()
                pts[:, :3] += pad_front
                if str(task["label_border_policy"]) == "skip" and len(pts):
                    pts = pts[~touches_border(pts, shape)]
                pts32 = pts.astype(np.float32)
                cls_idx = pts32[:, 3].astype(np.int64)
                radius = pts32[:, 4].astype(np.float64) / 2.0

                # Legacy empty-chunk drop probability, f1 = voxels covered by available classes.
                n_vox = float(np.prod(shape))
                f1 = float(min(n_vox, np.sum(4.0 / 3.0 * np.pi * radius[available[cls_idx]] ** 3)))
                f0 = n_vox - f1
                drop_p = float(np.clip(1.0 - alpha * f1 / f0, 0.0, 1.0)) if f0 > 0 else 0.0
                drop_rng = np.random.default_rng(_stable_seed(seed, name, li, k, "drop"))

                dense = render_targets(pts32, shape, n_classes, "clip") if storage == "dense" else None

                out_dir = Path(task["outpath"]) / f"{name}__L{li}s{k}"
                vol_dir = out_dir / constants.VOLUMES_DIR_NAME_PREFIX
                tgt_dir = out_dir / (V2_POINTS_DIR_NAME if storage == "points"
                                     else constants.LABELS_DIR_NAME_PREFIX % "_supervised")
                subject = tio.Subject(volume=tio.ScalarImage(tensor=padded.unsqueeze(0)))
                sampler = tio.GridSampler(subject, patch_size=chunk_size, patch_overlap=chunk_size - chunk_stride)
                n_here = 0
                with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 1))) as executor:
                    futures = []
                    for i, patch in enumerate(sampler):
                        loc = [int(v) for v in patch[tio.LOCATION][:3].tolist()]
                        local = pts32[:, :3] - np.asarray(loc, dtype=np.float32)
                        sel = np.all((local >= -radius[:, None]) & (local < chunk_size + radius[:, None]), axis=1)
                        chunk_pts = np.concatenate([local[sel], pts32[sel, 3:5]], axis=1).astype(np.float32)
                        empty = not bool(available[cls_idx[sel]].any())
                        if empty and drop_rng.random() < drop_p:
                            continue
                        if n_here == 0:
                            makedir(vol_dir)
                            makedir(tgt_dir)
                        suffix = f"{i}_{loc[0]}_{loc[1]}_{loc[2]}"
                        vol_path = str(vol_dir / f"{constants.VOLUMES_DIR_NAME_PREFIX}_{suffix}{constants.CUBES_EXTENSION}")
                        futures.append(executor.submit(_save_chunk, patch["volume"].data.clone(), vol_path))
                        if storage == "points":
                            _save_points(chunk_pts, str(tgt_dir / f"{V2_POINTS_PREFIX}_{suffix}.npy"))
                        else:
                            crop = dense[:, loc[0]:loc[0] + chunk_size, loc[1]:loc[1] + chunk_size,
                                         loc[2]:loc[2] + chunk_size]
                            lab_path = str(tgt_dir / f"{tgt_dir.name}_{suffix}{constants.CUBES_EXTENSION}")
                            futures.append(executor.submit(_save_chunk, torch.from_numpy(np.ascontiguousarray(crop)),
                                                           lab_path))
                        n_here += 1
                    for f in futures:
                        f.result()
                if n_here:
                    write_chunk_meta(out_dir, ChunkMeta(
                        tomo_name=tomo_path, dataset_id=int(task["dataset_id"]), level=li, scale_index=k,
                        apix_native=apix_native, apix_work=float(apix_work),
                        class_available=[bool(a) for a in available],
                        class_d_vox=[float(d / apix_work) for d in diam],
                        pipeline_version=2, resample=resample))
                    dirs.append(str(out_dir))
                    n_written += n_here
        return {"tomo": tomo_path, "n_chunks": n_written, "dirs": dirs, "error": None}
    except Exception:
        logging.error(f"Pipeline v2 chunking of {tomo_path} FAILED:\n{traceback.format_exc()}")
        return {"tomo": tomo_path, "n_chunks": 0, "dirs": [], "error": traceback.format_exc()}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_tasks(func, all_tasks: List[Dict[str, Any]], n_cpus: int, use_gpus: bool) -> list:
    """Run ``func`` on every task on a Dask cluster (GPU workers when available)."""
    n_gpus = torch.cuda.device_count() if use_gpus and torch.cuda.is_available() else 0
    if n_cpus <= 0:
        logging.warning(f"Configured n_cpus ({n_cpus}) is not positive. Defaulting to all available CPUs.")
        n_cpus = os.cpu_count()
    if n_gpus > 0:
        cluster = LocalCUDACluster(n_workers=n_gpus, threads_per_worker=max(1, n_cpus // n_gpus))
        logging.info(f"Starting Dask CUDA cluster with {n_gpus} GPU workers.")
    else:
        from dask.distributed import LocalCluster
        cluster = LocalCluster(n_workers=min(n_cpus, len(all_tasks)))
        logging.info(f"Starting Dask cluster with {min(n_cpus, len(all_tasks))} CPU workers.")
    client = Client(cluster)
    try:
        futures = [client.submit(func, task, pure=False) for task in all_tasks]
        results = []
        with tqdm(total=len(all_tasks), desc="Chunking Tomograms (pipeline v2)") as pbar:
            for future, result in as_completed(futures, with_results=True):
                results.append(result)
                pbar.update()
        return results
    finally:
        client.close()
        cluster.close()


def _do_chunking_v2(tomosDf: pd.DataFrame, chunkedDataDir: str, n_cpus: int, require_labels: bool,
                    train_val_level: CrossValidationLevelSplit, use_gpus: bool, scale_cfg: ScaleConfig,
                    class_table: Optional[list], seed: int, alpha_drop_empty: float, chunk_size: int,
                    chunk_stride: int, label_border_policy: str):
    """Pipeline v2 chunking (spec T2.3). Output: ``<split>/<dir name>__L{level}s{k}/`` with
    ``volumes/``, ``points/`` (or dense ``labels_supervised/``) and ``chunk_meta.json``; the
    manifest ``chunk_layout.json`` (v1 keys plus ``pipeline_version: 2``, the band, the class
    table and the validation tomograms); the done file last."""
    from tomocpt.dataManager.metadata import write_chunk_layout
    from tomocpt.labels.multiclass import class_table_from_json, class_table_to_json

    if not require_labels:
        raise ValueError("Pipeline v2 chunking needs labels (picking mode); self-supervised v2 is not available yet.")
    if not class_table:
        raise ValueError("Pipeline v2 chunking needs a class_table (labels.multiclass.class_table_from_config)")
    storage = str(getattr(scale_cfg, "target_storage", "points"))
    if storage not in ("points", "dense"):
        raise ValueError(f"scale.target_storage must be 'points' or 'dense', got '{storage}'")
    policy = str(getattr(label_border_policy, "value", label_border_policy))
    if policy not in ("skip", "clip"):
        raise ValueError(f"label_border_policy must be 'skip' or 'clip', got '{label_border_policy}'")
    resample = str(getattr(scale_cfg.resample, "value", scale_cfg.resample))
    table = class_table_from_json(class_table)
    table_json = class_table_to_json(table)

    merged = _merge_rows_v2(tomosDf, table)
    chunk_dir_names = assign_chunk_dir_names(merged)
    dir_names_layout = "unique" if merged["tomogram_path"].map(lambda p: Path(str(p)).stem).is_unique else "prefixed"
    if train_val_level == CrossValidationLevelSplit.tomos:
        df_train, df_val = split_rows_by_tomogram(merged, test_size=constants.PERCENT_TO_VALIDATE, seed=seed)
        split_layout = "rows" if dir_names_layout == "unique" else "grouped"
    else:
        df_train, df_val = merged, None
        split_layout = "rows"

    train_outDir, val_outDir = Path(chunkedDataDir) / "train", Path(chunkedDataDir) / "val"
    for d in (train_outDir, val_outDir, Path(chunkedDataDir) / "val_coords"):
        if d.exists():
            shutil.rmtree(d)
    makedir(train_outDir)
    makedir(val_outDir)
    for stale in (get_chunking_name_done(chunkedDataDir, True), get_chunk_layout_path(chunkedDataDir)):
        if Path(stale).is_file():
            os.remove(stale)

    n_gpus = torch.cuda.device_count() if use_gpus and torch.cuda.is_available() else 0
    all_tasks = []
    for split, df in (("train", df_train), ("val", df_val)):
        if df is None:
            continue
        outpath = train_outDir if split == "train" else val_outDir
        for idx, row in df.iterrows():
            all_tasks.append({
                "tomogram_path": row["tomogram_path"], "coords_paths": list(row["coords_paths"]),
                "apix_native": float(row["apix_native"]), "dataset_id": int(row["dataset_id"]),
                "class_available": [bool(a) for a in row["class_available"]],
                "chunk_dir_name": str(chunk_dir_names.loc[idx]), "outpath": str(outpath),
                "class_table": table_json, "band_min": float(scale_cfg.band_min_vox),
                "band_max": float(scale_cfg.band_max_vox), "scales_per_level": int(scale_cfg.scales_per_level),
                "resample": resample, "target_storage": storage, "label_border_policy": policy,
                "chunk_size": chunk_size, "chunk_stride": chunk_stride, "alpha_drop_empty": alpha_drop_empty,
                "seed": int(seed), "use_cuda": n_gpus > 0,
            })
    logging.info(f"Pipeline v2: chunking {len(all_tasks)} tomograms "
                 f"({len(df_train)} train, {0 if df_val is None else len(df_val)} validation).")

    results = _run_tasks(process_single_tomo_task_v2, all_tasks, n_cpus, use_gpus)
    failed = [r["tomo"] for r in results if r is None or r.get("error")]
    if failed:
        raise RuntimeError(f"Pipeline v2 chunking failed for {failed}; see the log for the tracebacks.")

    if train_val_level == CrossValidationLevelSplit.cubes:
        all_tomo_dirs = sorted(d for d in train_outDir.iterdir() if d.is_dir())
        _, val_dirs = train_test_split(all_tomo_dirs, test_size=constants.PERCENT_TO_VALIDATE, random_state=42)
        for d in val_dirs:
            shutil.move(str(d), val_outDir)

    val_tomograms = []
    if df_val is not None:
        for idx, row in df_val.iterrows():
            paths = list(row["coords_paths"])
            if len(paths) == 1:
                coords_path = paths[0]
            else:
                coords_path = Path(chunkedDataDir) / "val_coords" / f"{chunk_dir_names.loc[idx]}.csv"
                coords_path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([_read_coords(p) for p in paths], ignore_index=True).to_csv(coords_path, index=False)
            val_tomograms.append({
                "tomogram_path": str(row["tomogram_path"]), "coords_path": str(coords_path),
                "dataset_id": int(row["dataset_id"]), "apix_native": float(row["apix_native"]),
                "class_available": [bool(a) for a in row["class_available"]],
            })

    write_chunk_layout(chunkedDataDir, {
        "layout": 2, "dir_names": dir_names_layout, "split": split_layout,
        "pipeline_version": 2,
        "band": [float(scale_cfg.band_min_vox), float(scale_cfg.band_max_vox)],
        "resample": resample, "scales_per_level": int(scale_cfg.scales_per_level),
        "target_storage": storage, "label_border_policy": policy,
        "chunk_size": chunk_size, "chunk_stride": chunk_stride, "seed": int(seed),
        "class_table": table_json, "val_tomograms": val_tomograms,
    })
    with open(get_chunking_name_done(chunkedDataDir, require_labels), "w") as f:
        f.write("done")
    logging.info(f"Pipeline v2 chunking finished: {sum(r['n_chunks'] for r in results)} chunks in "
                 f"{train_outDir} and {val_outDir}")
