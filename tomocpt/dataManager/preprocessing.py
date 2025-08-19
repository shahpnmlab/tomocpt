import gc
import traceback
from typing import Callable, Tuple, Union
from pathlib import Path

import mrcfile
import numpy as np
import torch
from tomocpt.logger import get_logger

logging = get_logger()

def load_mrc(
    fname: str, normalize: bool = True, return_voxel_size: bool = False
) -> Union[np.array, Tuple[np.array, float]]:
    """
    Loads an MRC file and optionally normalizes it and returns its voxel size.
    """
    try:
        with mrcfile.open(fname, permissive=True) as f:
            em_map = f.data.astype(np.float32).copy()
            voxel_size = f.voxel_size.x.astype(np.float32)

            if normalize:
                em_map = robust_normalization(em_map)

    except ValueError:
        logging.error(f"Error reading MRC file: {fname}")
        raise

    if return_voxel_size:
        return em_map, voxel_size
    else:
        return em_map


def robust_normalization(data: np.ndarray) -> np.ndarray:
    """

    Normalizes data using 5th and 95th percentiles, clipping values.
    """
    p5 = np.percentile(data, 5)
    p95 = np.percentile(data, 95)
    if p95 - p5 > 1e-6:
        median = np.median(data)
        return np.clip((data - median) / (p95 - p5), a_min=-3, a_max=3)
    else:
        return data - data.mean()


def get_shape_for_resizing(
    volume_shape: Tuple[int, ...], original_particle_px: float, target_particle_px: float
) -> Tuple[Tuple[int, ...], float]:
    """
    Computes the target shape for volume resizing based on particle size.
    """
    if abs(original_particle_px - target_particle_px) < 1:
        return volume_shape, 1.0

    resize_factor = target_particle_px / original_particle_px
    output_shape = tuple(int(s * resize_factor) for s in volume_shape)
    return output_shape, resize_factor


def resize_volume(
    volume: torch.Tensor,
    target_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Resizes a 3D volume tensor. It will attempt to use the GPU if the input tensor
    is on a CUDA device, but will fall back to the CPU if a CUDA OOM error occurs.
    """
    input_device = volume.device
    
    def _perform_resize(device: torch.device) -> torch.Tensor:
        """Inner function to perform the resize on a specific device."""
        vol_on_device = volume.to(device)
        vol_4d = vol_on_device.unsqueeze(0).unsqueeze(0)
        with torch.amp.autocast(device_type=device.type):
            resized_vol = torch.nn.functional.interpolate(
                vol_4d, size=target_shape, mode="trilinear", align_corners=False
            )
        return resized_vol.squeeze(0).squeeze(0)

    if input_device.type == 'cuda':
        try:
            return _perform_resize(input_device)
        except torch.cuda.OutOfMemoryError:
            logging.warning(
                "CUDA out of memory during resize operation. "
                "Falling back to CPU for this task. This will be slower."
            )
            gc.collect()
            torch.cuda.empty_cache()
            return _perform_resize(torch.device("cpu"))
    else:
        return _perform_resize(input_device)


def symmetrize_and_pad(
    volume: torch.Tensor,
    min_size: int,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Applies symmetric padding to a volume to ensure its dimensions are at least `min_size`.
    This operation is lightweight and will run on the device of the input tensor.
    """
    shape = volume.shape
    discrepancy = [max(0, min_size - s) for s in shape]

    if not any(d > 0 for d in discrepancy):
        return volume, (0, 0, 0, 0, 0, 0)

    padding = []
    for d in reversed(discrepancy):
        pad_1 = d // 2
        pad_2 = d - pad_1
        padding.extend([pad_1, pad_2])

    padding_tuple = tuple(padding)
    vol_5d = volume.unsqueeze(0).unsqueeze(0)
    padded_vol = torch.nn.functional.pad(
        vol_5d, pad=padding_tuple, mode="constant", value=volume.mean().item()
    )
    return padded_vol.squeeze(0).squeeze(0), padding_tuple

def get_mrc_metadata(mrc_path: str) -> Tuple[Tuple[int, ...], float]:
    """
    Correctly and efficiently reads the shape and voxel size from an MRC file.
    """
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        shape = mrc.data.shape
        voxel_size = mrc.voxel_size.x.astype(np.float32)
    return shape, voxel_size
def preprocess_tomogram(
    mrc_path: str,
    particle_diameter_angst: float,
    target_particle_px: int,
    chunk_size: int,
    device: torch.device,
    invert_contrast: bool = False,
) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    """
    Processes a tomogram by loading, resizing, and padding it, returning the
    result on the CPU for safe caching and the padding values.
    """
    vol_np = load_mrc(mrc_path, normalize=True, return_voxel_size=False)
    
    if invert_contrast:
        vol_np *= -1
        logging.info(f"Tomogram contrast inverted for {Path(mrc_path).name}")

    _, original_voxel_size = get_mrc_metadata(mrc_path)

    original_particle_px = particle_diameter_angst / original_voxel_size
    target_shape_for_resize, _ = get_shape_for_resizing(vol_np.shape, original_particle_px, target_particle_px)

    try:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        vol_tensor = torch.from_numpy(vol_np).to(device)
        resized_tensor = resize_volume(vol_tensor, target_shape_for_resize)
    except torch.cuda.OutOfMemoryError:
        logging.warning(
            f"CUDA OOM when loading tomogram {Path(mrc_path).name}. "
            f"Preprocessing will proceed on CPU."
        )
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        cpu_device = torch.device("cpu")
        vol_tensor = torch.from_numpy(vol_np).to(cpu_device)
        resized_tensor = resize_volume(vol_tensor, target_shape_for_resize)

    padded_tensor, padding_tuple = symmetrize_and_pad(resized_tensor, chunk_size)
    final_shape = tuple(padded_tensor.shape)

    result_cpu = padded_tensor.cpu()

    del vol_np, vol_tensor, resized_tensor, padded_tensor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return result_cpu, final_shape, padding_tuple


def process_label_to_match_tomogram(
    mrc_path: str,
    final_tomogram_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """
    Processes a label mask, forcing it to the exact final shape of a
    pre-processed tomogram and returning it on the CPU.
    """
    label_np = load_mrc(mrc_path, normalize=False, return_voxel_size=False)
    try:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        label_tensor = torch.from_numpy(label_np).to(device)
        resized_label = resize_volume(label_tensor, final_tomogram_shape)
    except torch.cuda.OutOfMemoryError:
        logging.warning(
            f"CUDA OOM when loading label {Path(mrc_path).name}. "
            f"Label processing will proceed on CPU."
        )
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        cpu_device = torch.device("cpu")
        label_tensor = torch.from_numpy(label_np).to(cpu_device)
        resized_label = resize_volume(label_tensor, final_tomogram_shape)
        
    result_cpu = resized_label.cpu()

    del label_np, label_tensor, resized_label
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return result_cpu
