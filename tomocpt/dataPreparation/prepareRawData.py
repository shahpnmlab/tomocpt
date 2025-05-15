import gc
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
import torch
import torchio as tio
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from tomocpt import constants
from tomocpt.dataManager.dataUtils import load_mrc, resize_volume
from tomocpt.dataPreparation.helpers import (
    _preprocess_data_mrc,
    get_labels_dirname,
    get_vol_chunks,
)
from tomocpt.defaultConfigs.train_config import CrossValidationLevelSplit
from tomocpt.mainConfig import mainConfig
from tomocpt.utils import accelerator_selector, makedir


def process_mrc(
        data_fname,
        target_fname,
        particle_diameter_angst,
        outputname_template: Optional[str],
        new_size: int,
        chunk_size: Optional[int] = None,
        stride: Optional[int] = None,
        normalization_function: str = "robust_normalization",
        require_labels: bool = True,
        use_gpu: bool = False,
        gpu_id: Optional[int] = None,
):
    """Process MRC files with proper device management"""
    particle_radius_angst = particle_diameter_angst * 0.5

    if outputname_template is None:
        outputname_template = constants.CUBES_FNAMES_TEMPLATES

    if chunk_size is None:
        chunk_size = mainConfig.train.CHUNK_SIZE

    if stride is None:
        stride = mainConfig.train.CHUNK_STRIDE

    # Determine and set target device
    if use_gpu and torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")
        use_gpu = False

    # Process with target device
    vol, new_shape, old_shape, voxel_size, padding_values, scalar = (
        _preprocess_data_mrc(
            data_fname,
            particle_radius_angst,
            normalization_function,
            new_size,
            chunk_size,
            use_gpu,
            device=device,
        )
    )

    alpha = mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES

    # Handle labels with proper device management
    drop_probablity = 0.0
    if target_fname:
        vol_target = load_mrc(target_fname, normalize=None, return_boxSize=False)
        vol_target = torch.tensor(vol_target, device=device)  # Ensure same device

        # Ensure vol is on the same device
        if vol.device != device:
            vol = vol.to(device)

        F0 = torch.isclose(vol_target, torch.zeros_like(vol_target)).sum()
        F1 = torch.numel(vol_target) - F0

        # Create zeros and ones on the SAME device
        zeros = torch.zeros(1, device=device)
        ones = torch.ones(1, device=device)

        # All tensors now on same device
        drop_probablity = torch.clip(
            1 - alpha * (F1 / F0), zeros, ones
        )
    else:
        vol_target = torch.zeros_like(vol, device=device)  # Ensure same device

    if vol_target.shape != tuple(new_shape):
        assert min(new_shape) >= chunk_size
        vol_target, _ = resize_volume(
            volume=vol_target,
            new_shape=new_shape,
            chunk_size=chunk_size,
            use_gpu=use_gpu,
        )

    labels_dirname = get_labels_dirname(not target_fname is None)

    n_cubes = 0
    chunk_data_fnames = []

    # Process chunks with updated amp.autocast
    with torch.amp.autocast('cuda' if use_gpu and torch.cuda.is_available() else 'cpu'):
        for i, ((chunk_data, chunk_target), coords) in enumerate(
                get_vol_chunks(vol, vol_target, chunk_size, stride=stride, device=device)
        ):
            if chunk_target.sum() <= 0:
                # Convert to scalar for comparison
                drop_prob_val = drop_probablity.item() if isinstance(drop_probablity, torch.Tensor) else drop_probablity
                if random.random() < drop_prob_val:
                    continue

            chunk_data_fname = outputname_template % (
                constants.VOLUMES_DIR_NAME_PREFIX,
                constants.VOLUMES_DIR_NAME_PREFIX,
                i,
                *tuple(coords),
            )
            chunk_data_fnames.append(chunk_data_fname)

            # Move to CPU for saving
            chunk_data_cpu = chunk_data.cpu() if chunk_data.device.type != 'cpu' else chunk_data
            chunk_data = tio.ScalarImage(tensor=chunk_data_cpu.unsqueeze(0))
            chunk_data.save(chunk_data_fname, squeeze=False)
            del chunk_data, chunk_data_cpu

            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Handle target with proper device management
            if target_fname:
                labels_dir_prefix = get_labels_dirname(require_labels)
                chunk_target_fname = outputname_template % (
                    labels_dir_prefix,
                    labels_dir_prefix,
                    i,
                    *tuple(coords),
                )
                chunk_target_cpu = chunk_target.cpu() if chunk_target.device.type != 'cpu' else chunk_target
                chunk_target = tio.ScalarImage(tensor=chunk_target_cpu.unsqueeze(0))
                chunk_target.save(chunk_target_fname, squeeze=False)
                del chunk_target, chunk_target_cpu

                if use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                fullparent, basename = os.path.split(chunk_data_fname)
                basename = labels_dirname + basename.removeprefix(
                    constants.VOLUMES_DIR_NAME_PREFIX
                )
                dirname = os.path.join(os.path.split(fullparent)[0], labels_dirname)
                chunk_target_fname = os.path.join(dirname, basename)
                os.makedirs(dirname, exist_ok=True)
                os.symlink(chunk_data_fname, chunk_target_fname)
            n_cubes += 1

    return data_fname, n_cubes, chunk_data_fnames


def get_chunking_name_done(chunkedDataDir, require_labels):
    return f"{chunkedDataDir}/done_labels_{require_labels}.txt"


def do_chunking(
        tomosDf: pd.DataFrame,
        chunkedDataDir,
        desired_particle_pixel_size: int,
        n_cpus: int,
        require_labels: bool = True,
        train_val_level=None,
        use_gpus: bool = True,
        oversubscribe_factor: int = 2,
):
    """
    Enhanced chunking function with multi-GPU distribution

    Args:
        tomosDf: DataFrame with volume-label filename pairs
        chunkedDataDir: Directory for chunked cubes storage
        desired_particle_pixel_size: Target particle size in pixels
        n_cpus: Number of CPUs to use
        require_labels: Whether labels are required
        train_val_level: Train-validation split level
        use_gpus: Whether to use GPUs
        oversubscribe_factor: Controls GPU oversubscription
    """
    if train_val_level is None:
        train_val_level = mainConfig.train.train_on

    if train_val_level == CrossValidationLevelSplit.tomos:
        df_train, df_val = train_test_split(
            tomosDf, test_size=constants.PERCENT_TO_VALIDATE
        )
    elif train_val_level == CrossValidationLevelSplit.cubes:
        df_train = tomosDf
        df_val = None
    else:
        raise ValueError()

    n_cpus = 1 if n_cpus == 0 else n_cpus
    train_outDir = f"{chunkedDataDir}/{constants.TRAIN_DIR_NAME}"
    val_outDir = f"{chunkedDataDir}/{constants.VAL_DIR_NAME}"

    if os.path.isdir(train_outDir):
        shutil.rmtree(train_outDir, ignore_errors=False)
    if os.path.isdir(val_outDir):
        shutil.rmtree(val_outDir, ignore_errors=False)

    outputname_template = constants.CUBES_FNAMES_TEMPLATES

    # Get GPU information
    n_gpus = 0
    if use_gpus and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs for processing")

    if n_gpus == 0:
        use_gpus = False
        print("No GPUs available, falling back to CPU processing")

    def dispatcher(i, info_row, outpath, gpu_id=None):

        # Use specific GPU if available
        if use_gpus and gpu_id is not None:
            torch.cuda.set_device(gpu_id)

        bn = Path(info_row["tomogram_path"]).stem
        bn_fname = f"{outpath}/{bn}/{outputname_template}"
        makedir(f"{outpath}/{bn}/{constants.VOLUMES_DIR_NAME_PREFIX}")

        if require_labels:
            labels_names_prefix = get_labels_dirname(True)
        else:
            labels_names_prefix = get_labels_dirname(False)

        makedir(f"{outpath}/{bn}/{labels_names_prefix}")
        if require_labels:
            target_fname = info_row["label_path"]
        else:
            target_fname = None
        particle_diameter_angst = info_row["particle_diameter_angst"]

        data_fname, n_cubes, chunk_data_fnames = process_mrc(
            data_fname=info_row["tomogram_path"],
            target_fname=target_fname,
            particle_diameter_angst=particle_diameter_angst,
            outputname_template=bn_fname,
            new_size=desired_particle_pixel_size,
            require_labels=require_labels,
            use_gpu=use_gpus,
            gpu_id=gpu_id,
        )

        print(f"{data_fname}: {n_cubes} cubes written" + (f" on GPU {gpu_id}" if gpu_id is not None else ""))
        return data_fname

    # Assign GPUs using the requested distribution formula
    if use_gpus and n_gpus > 0:
        train_tasks = []
        for i, row in df_train.iterrows():
            # Apply the requested GPU distribution formula
            if oversubscribe_factor > 0:
                gpu_id = (i % oversubscribe_factor) % n_gpus
            else:
                gpu_id = (i // abs(oversubscribe_factor)) % n_gpus
            train_tasks.append((i, row, train_outDir, gpu_id))

        # Use parallel execution with GPU assignments
        train_fnames = Parallel(n_cpus, batch_size=1)(
            delayed(dispatcher)(i, row, outdir, gpu) for i, row, outdir, gpu in train_tasks
        )

        if df_val is not None:
            val_tasks = []
            for i, row in df_val.iterrows():
                if oversubscribe_factor > 0:
                    gpu_id = (i % oversubscribe_factor) % n_gpus
                else:
                    gpu_id = (i // abs(oversubscribe_factor)) % n_gpus
                val_tasks.append((i, row, val_outDir, gpu_id))

            Parallel(n_cpus, batch_size=1)(
                delayed(dispatcher)(i, row, outdir, gpu) for i, row, outdir, gpu in val_tasks
            )
    else:
        # Original CPU-only processing
        train_fnames = Parallel(n_cpus, batch_size=1)(
            delayed(dispatcher)(i, trainObj, train_outDir)
            for i, trainObj in df_train.iterrows()
        )

        if df_val is not None:
            Parallel(n_cpus, batch_size=1)(
                delayed(dispatcher)(i, valObj, val_outDir)
                for i, valObj in df_val.iterrows()
            )

    # Handle cubes-level split if needed
    if df_val is None:
        train_fnames = train_test_split(
            train_fnames, test_size=constants.PERCENT_TO_VALIDATE
        )
        for fname in train_fnames:
            os.rename(fname, Path(val_outDir) / Path(fname).name)

    print(f"Prepared data saved at: \n{train_outDir}\n{val_outDir}")
    with open(get_chunking_name_done(chunkedDataDir, require_labels), "w") as f:
        f.write("done")