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
    """Process MRC files with simplified GPU handling"""
    import gc
    import os
    import random
    import torch
    import torchio as tio
    
    particle_radius_angst = particle_diameter_angst * 0.5

    if outputname_template is None:
        outputname_template = constants.CUBES_FNAMES_TEMPLATES

    if chunk_size is None:
        chunk_size = mainConfig.train.CHUNK_SIZE

    if stride is None:
        stride = mainConfig.train.CHUNK_STRIDE

    # Try to process with GPU if requested
    try:
        if use_gpu and torch.cuda.is_available():
            if gpu_id is not None:
                # Use specific GPU if provided
                try:
                    gpu_idx = int(gpu_id)
                    torch.cuda.set_device(gpu_idx)
                except:
                    # If conversion fails, use default device
                    pass
            
            # Run preprocessing with GPU
            vol, new_shape, old_shape, voxel_size, padding_values, scalar = (
                _preprocess_data_mrc(
                    data_fname,
                    particle_radius_angst,
                    normalization_function,
                    new_size,
                    chunk_size,
                    use_gpu=True
                )
            )
        else:
            # Run preprocessing on CPU
            vol, new_shape, old_shape, voxel_size, padding_values, scalar = (
                _preprocess_data_mrc(
                    data_fname,
                    particle_radius_angst,
                    normalization_function,
                    new_size,
                    chunk_size,
                    use_gpu=False
                )
            )
    except RuntimeError as e:
        if "CUDA error" in str(e) or "out of memory" in str(e):
            # If GPU processing fails, fall back to CPU
            print(f"GPU processing failed: {str(e)}. Falling back to CPU.")
            torch.cuda.empty_cache()
            vol, new_shape, old_shape, voxel_size, padding_values, scalar = (
                _preprocess_data_mrc(
                    data_fname,
                    particle_radius_angst,
                    normalization_function,
                    new_size,
                    chunk_size,
                    use_gpu=False
                )
            )
        else:
            raise

    alpha = mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES

    # Handle target processing
    drop_probablity = 0.0
    if target_fname:
        vol_target = load_mrc(target_fname, normalize=None, return_boxSize=False)
        vol_target = torch.tensor(vol_target)  # Keep on CPU initially

        # Check if we need to move to GPU
        if use_gpu and torch.cuda.is_available() and vol.device.type == 'cuda':
            try:
                vol_target = vol_target.cuda()
            except RuntimeError:
                # If GPU memory is insufficient, move main volume to CPU
                vol = vol.cpu()
                vol_target = vol_target  # Keep on CPU

        F0 = torch.isclose(vol_target, torch.zeros_like(vol_target)).sum()
        F1 = torch.numel(vol_target) - F0

        zeros = torch.zeros(1, device=vol_target.device)
        ones = torch.ones(1, device=vol_target.device)

        drop_probablity = torch.clip(
            1 - alpha * (F1 / F0), zeros, ones
        )
    else:
        # Create target with same device as volume
        if use_gpu and torch.cuda.is_available() and vol.device.type == 'cuda':
            try:
                vol_target = torch.zeros_like(vol)
            except RuntimeError:
                # If GPU memory is insufficient, move to CPU
                vol = vol.cpu()
                vol_target = torch.zeros_like(vol)
        else:
            vol_target = torch.zeros_like(vol)

    if vol_target.shape != tuple(new_shape):
        assert min(new_shape) >= chunk_size
        try:
            vol_target, _ = resize_volume(
                volume=vol_target,
                new_shape=new_shape,
                chunk_size=chunk_size,
                use_gpu=vol_target.device.type=='cuda',
            )
        except RuntimeError:
            # If resize fails, move to CPU and try again
            if vol_target.device.type != 'cpu':
                vol_target = vol_target.cpu()
            if vol.device.type != 'cpu':
                vol = vol.cpu()
            vol_target, _ = resize_volume(
                volume=vol_target,
                new_shape=new_shape,
                chunk_size=chunk_size,
                use_gpu=False,
            )

    labels_dirname = get_labels_dirname(not target_fname is None)

    n_cubes = 0
    chunk_data_fnames = []
    
    # Process chunks
    for i, ((chunk_data, chunk_target), coords) in enumerate(
            get_vol_chunks(vol, vol_target, chunk_size, stride=stride)
    ):
        if chunk_target.sum() <= 0:
            # Handle drop probability 
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
        chunk_data_torch = tio.ScalarImage(tensor=chunk_data_cpu.unsqueeze(0))
        chunk_data_torch.save(chunk_data_fname, squeeze=False)
        del chunk_data_torch, chunk_data_cpu

        # Clean up GPU memory
        if torch.cuda.is_available():
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
            chunk_target_torch = tio.ScalarImage(tensor=chunk_target_cpu.unsqueeze(0))
            chunk_target_torch.save(chunk_target_fname, squeeze=False)
            del chunk_target_torch, chunk_target_cpu

            # Clean up GPU memory
            if torch.cuda.is_available():
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
    """Reliable chunking function for single-GPU and multi-GPU setups"""
    import os
    import shutil
    import random
    import torch
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from joblib import Parallel, delayed
    
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

    # SIMPLIFIED GPU detection
    gpu_count = 0
    if use_gpus and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPUs for processing")
    else:
        use_gpus = False
        print("No GPUs available or use_gpus=False, falling back to CPU processing")

    def dispatcher(i, info_row, outpath, gpu_idx=None):
        """Process a single tomogram with proper error handling"""
        try:
            # Set GPU if specified and available
            if use_gpus and gpu_idx is not None and gpu_count > 0:
                try:
                    # Integer index for GPU - safest approach
                    device_idx = int(gpu_idx) % gpu_count
                    torch.cuda.set_device(device_idx)
                except Exception as e:
                    print(f"Warning: Could not set GPU device: {str(e)}")

            # Create directories
            bn = Path(info_row["tomogram_path"]).stem
            bn_fname = f"{outpath}/{bn}/{outputname_template}"
            makedir(f"{outpath}/{bn}/{constants.VOLUMES_DIR_NAME_PREFIX}")

            if require_labels:
                labels_names_prefix = get_labels_dirname(True)
            else:
                labels_names_prefix = get_labels_dirname(False)

            makedir(f"{outpath}/{bn}/{labels_names_prefix}")
            
            target_fname = info_row["label_path"] if require_labels else None
            particle_diameter_angst = info_row["particle_diameter_angst"]

            # Process the MRC file
            data_fname, n_cubes, chunk_data_fnames = process_mrc(
                data_fname=info_row["tomogram_path"],
                target_fname=target_fname,
                particle_diameter_angst=particle_diameter_angst,
                outputname_template=bn_fname,
                new_size=desired_particle_pixel_size,
                require_labels=require_labels,
                use_gpu=use_gpus and gpu_count > 0,
                gpu_id=device_idx if (use_gpus and gpu_idx is not None and gpu_count > 0) else None,
            )

            print(f"{data_fname}: {n_cubes} cubes written" + 
                  (f" on GPU {device_idx}" if use_gpus and gpu_idx is not None and gpu_count > 0 else ""))
            return data_fname
            
        except RuntimeError as e:
            if "CUDA error" in str(e) or "out of memory" in str(e):
                print(f"GPU memory error for {info_row['tomogram_path']}, falling back to CPU")
                # Free memory
                torch.cuda.empty_cache()
                
                # Try again on CPU
                try:
                    data_fname, n_cubes, chunk_data_fnames = process_mrc(
                        data_fname=info_row["tomogram_path"],
                        target_fname=target_fname if 'target_fname' in locals() else None,
                        particle_diameter_angst=info_row["particle_diameter_angst"],
                        outputname_template=bn_fname if 'bn_fname' in locals() else None,
                        new_size=desired_particle_pixel_size,
                        require_labels=require_labels,
                        use_gpu=False,
                        gpu_id=None,
                    )
                    print(f"{data_fname}: {n_cubes} cubes written (fallback to CPU)")
                    return data_fname
                except Exception as fallback_error:
                    print(f"CPU fallback also failed: {str(fallback_error)}")
                    raise
            else:
                raise

    # Simplified GPU assignment for reliable operation
    if use_gpus and gpu_count > 0:
        # Simple round-robin GPU assignment
        train_tasks = []
        for i, row in df_train.iterrows():
            gpu_idx = 0 if gpu_count == 1 else (i % gpu_count)
            train_tasks.append((i, row, train_outDir, gpu_idx))
        
        # Process in parallel
        train_fnames = Parallel(n_cpus, batch_size=1)(
            delayed(dispatcher)(i, row, outdir, gpu_idx) 
            for i, row, outdir, gpu_idx in train_tasks
        )

        if df_val is not None:
            val_tasks = []
            for i, row in df_val.iterrows():
                gpu_idx = 0 if gpu_count == 1 else (i % gpu_count)
                val_tasks.append((i, row, val_outDir, gpu_idx))
            
            Parallel(n_cpus, batch_size=1)(
                delayed(dispatcher)(i, row, outdir, gpu_idx) 
                for i, row, outdir, gpu_idx in val_tasks
            )
    else:
        # CPU-only processing
        train_fnames = Parallel(n_cpus, batch_size=1)(
            delayed(dispatcher)(i, row, train_outDir, None)
            for i, row in df_train.iterrows()
        )

        if df_val is not None:
            Parallel(n_cpus, batch_size=1)(
                delayed(dispatcher)(i, row, val_outDir, None)
                for i, row in df_val.iterrows()
            )

    # Handle cubes-level split if needed
    if df_val is None and train_fnames:
        train_val_split = train_test_split(
            train_fnames, test_size=constants.PERCENT_TO_VALIDATE
        )
        if len(train_val_split) >= 2:
            val_fnames = train_val_split[1]
            for fname in val_fnames:
                if fname is not None:
                    try:
                        os.rename(fname, Path(val_outDir) / Path(fname).name)
                    except (FileNotFoundError, OSError) as e:
                        print(f"Warning: Unable to move file {fname}: {str(e)}")

    print(f"Prepared data saved at:\n{train_outDir}\n{val_outDir}")
    with open(get_chunking_name_done(chunkedDataDir, require_labels), "w") as f:
        f.write("done")
