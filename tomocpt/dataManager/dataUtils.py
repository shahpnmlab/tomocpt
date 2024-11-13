import gc
import re
import shutil
from pathlib import Path

import pandas as pd
import torch
import mrcfile
import numpy as np

from typing import Callable, Union, Tuple, List
from tomocpt.constants import LABELS_DIR_NAME_PREFIX
from tomocpt import mainConfig, constants

config = mainConfig


def symmetrize_padding(inp):
    """
     Symmetrizes the padding values in a given input sequence.

    The function takes an iterable 'inp' as its input and returns a tuple of symmetrized values.
    The output tuple is obtained by iterating over each value in 'inp' and checking whether it is even or odd.
    If the value is even, it is divided by 2 and the quotient is appended twice to the output tuple, creating a symmetric pattern.
    If the value is odd, it is divided by 2 and the quotient and quotient+1 are appended to the output tuple.
    The final tuple is reversed before being returned.

    Args:
    - inp: an iterable containing the padding values

    Returns:
    - A tuple of  symmetrize padding values

    Example:
     symmetrize_padding([1, 2, 3, 4])
    (2, 1, 2, 2, 3, 2, 1, 2)
    """
    symmetric = []
    for i in inp:
        # print(i)
        if i % 2 == 0:
            symmetric.append(i // 2)
            symmetric.append(i // 2)
        if i % 2 == 1:
            symmetric.append(i // 2)
            symmetric.append((i // 2) + 1)
    return tuple(symmetric[::-1])


def load_mrc(fname: str, normalize: Union[str, Callable, None] = None,
             return_boxSize: bool = False) -> Union[np.array, Tuple[np.array, float]]:
    """

    :param fname:
    :param normalize:
    :param return_boxSize:
    :return:
    """
    try:
        with mrcfile.open(fname, permissive=True) as f:
            emMap = f.data.astype(np.float32).copy()
            voxel_size = f.voxel_size
            # print("loading", fnameIn, emMap.mean(), emMap.std(), emMap.min(), emMap.max())
            if normalize is not None and normalize is not False:
                if callable(normalize):
                    emMap = normalize(emMap)
                elif normalize is True or normalize.startswith("minMax"):
                    emMap = (emMap - emMap.min()) / (emMap.max() - emMap.min())
                else:
                    raise ValueError("Normalization type not recognized")
    except ValueError:
        print("Error with %s" % fname)
        raise
    assert emMap.shape[0] > 0, "Error, %s is empty" % fname
    if return_boxSize:
        voxel_size = voxel_size.x
        return emMap, voxel_size
    else:
        return emMap



def get_shape_for_resizing(volume: np.array, original_size: float, new_size: float) -> Tuple:
    """
    Computes the new shape of a volume after resizing it from the original size to the new size.

    Args:
    volume (np.array): The input volume to be resized.
    original_size (float): The original size of the volume.
    new_size (float): The new size of the volume.

    Returns:
    tuple: The new shape of the resized volume.
    """

    if original_size > new_size:
        resize_factor = original_size / new_size
        output_shape = tuple(int(s / resize_factor) for s in volume.shape)
    elif original_size < new_size:
        resize_factor = original_size / new_size
        output_shape = tuple(int(s / resize_factor) for s in volume.shape)
    else:
        output_shape = volume.shape
        resize_factor = 1
    return resize_factor, output_shape


def resize_volume(volume: np.array, new_shape: Union[Tuple[int], List[int]], chunk_size,
                  use_gpu: bool | None = None, calculate_mean: bool = True) -> np.array:
    """
    Resizes a 3D volume to a new shape using bicubic interpolation. If any dimension of the new shape is smaller
    than the given chunk_size, the function symmetrically pads the volume to match the chunk size.

    Args:
        volume (np.array): A 3D numpy array representing the volume to be resized.
        new_shape (Union[Tuple[int], List[int]]): A tuple or list of 3 integers representing the new shape of the volume.
        chunk_size (int): An integer representing the chunk size.

    Returns:
        np.array: A 3D numpy array representing the resized volume.

    Raises:
        ValueError: If volume is not a 3D numpy array.
        ValueError: If new_shape is not a tuple or list of 3 integers.
        ValueError: If chunk_size is not an integer.
        ValueError: If any element of new_shape is smaller than or equal to 0.
        ValueError: If chunk_size is smaller than or equal to 0.

    Examples:
        # Resize a volume to a new shape of (256, 256, 128)
        volume = np.random.rand(200, 200, 100)
        new_shape = (256, 256, 128)
        resized = resize_volume(volume, new_shape, chunk_size=32)

        # Resize a volume to a new shape of [192, 192, 64]
        volume = np.random.rand(200, 200, 100)
        new_shape = [192, 192, 64]
        resized = resize_volume(volume, new_shape, chunk_size=16)
    """
    #TODO: Document what calculate mean does
    assert use_gpu is not None, "Error, you need to tell if gpu is going to be used"
    if calculate_mean:
        intensity = volume.mean()
    else:
        intensity = 0

    shape = np.array(new_shape)
    ndim = len(volume.shape)
    if ndim > 3:
        resized = torch.nn.functional.interpolate(
            torch.as_tensor(volume, device='cuda' if use_gpu else 'cpu').unsqueeze(0),
            size=new_shape, mode='trilinear', antialias=False).squeeze().cpu()
    else:
        resized = torch.nn.functional.interpolate(
            torch.as_tensor(volume, device='cuda' if use_gpu else 'cpu').unsqueeze(0).unsqueeze(0),
            size=new_shape, mode='trilinear', antialias=False).squeeze().squeeze().cpu()
    del volume
    gc.collect()

    if any(shape) < chunk_size:
        discrepancy = chunk_size - shape
        amount_to_pad = np.zeros_like(discrepancy)
        for idx, (s, d) in enumerate(zip(shape, discrepancy)):
            if d > 0:
                amount_to_pad[idx] = d
        symmetric_padding = symmetrize_padding(amount_to_pad)
        # print(symmetric_padding)
        if ndim < 4:
            resized = torch.nn.functional.pad(torch.as_tensor(resized).unsqueeze(0).unsqueeze(0), symmetric_padding,
                                              mode='constant', value=intensity).squeeze().squeeze()
        else:
            resized = torch.nn.functional.pad(torch.as_tensor(resized).unsqueeze(0), symmetric_padding,
                                              mode='constant', value=intensity).squeeze()
    return resized, symmetric_padding


def robust_normalization(data: np.array) -> np.array:
    p5 = np.percentile(data, q=5)
    p95 = np.percentile(data, q=95)
    median = np.median(data)
    return np.clip((data - median) / (p95 - p5), a_min=-3, a_max=3)


def write_segmentation_mask(tensor_mask, outputname, angpix: float = 1, overwrite=True):
    if isinstance(tensor_mask, torch.Tensor):
        m = tensor_mask.numpy()
    else:
        m = tensor_mask
    if not overwrite:
        mrcfile.write(outputname, data=m, voxel_size=angpix)
    else:
        mrcfile.write(outputname, data=m, voxel_size=angpix, overwrite=True)


def ang_to_pix(val, sampling):
    return int(val / sampling)


def get_tomo_dims(raw_tomo_path: str):
    raw_tomo_files = list(Path(raw_tomo_path).rglob('*.mrc'))
    return _get_single_tomo_dims(raw_tomo_files[0])


def _get_single_tomo_dims(fname):
    with mrcfile.open(fname, 'r') as f_in:
        one_vol_shape = f_in.data.shape
        one_vol_voxel_size = f_in.voxel_size['x'].astype('float')
    return one_vol_shape, one_vol_voxel_size


def get_labels_dirname(require_labels: bool):
    if require_labels:
        return LABELS_DIR_NAME_PREFIX % "_supervised"
    else:
        return LABELS_DIR_NAME_PREFIX % "_selfSup"

def plot_example(x, label):
    from matplotlib import pyplot as plt
    if type(x) != np.ndarray:
        x = x.cpu().detach().numpy()
    if type(label) != np.ndarray:
        label = label.cpu().detach().numpy()
    if len(x.shape) == 4:
        x = x[0, ...]
    if len(label.shape) == 4:
        label = label[0, ...]

    print(x.shape)
    print(label.shape)

    n = x.shape[0]
    central = n//2
    fig, ax = plt.subplots(2,5)
    ax[0,0].imshow(x[central-central//2,...], cmap="gray")
    ax[0,1].imshow(x[central-1,...], cmap="gray")
    ax[0,2].imshow(x[central,...], cmap="gray")
    ax[0,3].imshow(x[central+1,...], cmap="gray")
    ax[0,4].imshow(x[central+central//2,...], cmap="gray")

    ax[1,0].imshow(label[central-central//2,...], cmap="gray")
    ax[1,1].imshow(label[central-1,...], cmap="gray")
    ax[1,2].imshow(label[central,...], cmap="gray")
    ax[1,3].imshow(label[central+1,...], cmap="gray")
    ax[1,4].imshow(label[central+central//2,...], cmap="gray")
    plt.show()
    print("")