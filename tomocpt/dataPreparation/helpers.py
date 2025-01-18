import functools
import numpy as np
import torch
from tomocpt import constants
from tomocpt.dataManager.dataUtils import load_mrc, get_shape_for_resizing, resize_volume
from tomocpt.logger import get_logger

logger = get_logger()


def _preprocess_data_mrc(data_fname:str, particle_radius_angst, normalization_function, new_particle_size, chunk_size, use_gpu):
    """

Preprocesses MRC (Medical Research Council) data by loading, normalizing, and resizing the volume.

    Args:
        data_fname (str): Path to the input MRC file.
        particle_radius_angst (float): Half the particle length in Angstroms. Used for scaling even if particle isn't spherical.
        normalization_function (str): Normalization method to use. Currently only supports "robust_normalization".
        new_particle_size (int): Target size in pixels for half the particle length after resizing.
        chunk_size (int): Size of chunks used during volume resizing to manage memory usage.
        use_gpu (bool): Whether to use GPU acceleration for volume resizing.

    Returns:
        tuple: Contains:
            - vol (torch.Tensor): Processed and resized volume data
            - new_shape (List[int]): Dimensions of resized volume [Z,Y,X]
            - old_shape (tuple): Original dimensions of input volume
            - voxel_size (float): Size of each voxel in Angstroms
            - padding_values (tuple, optional): Padding values used in resizing, None if no resize
            - scalar (float): Scaling factor used in resize operation

    Raises:
        NotImplementedError: If normalization_function is not "robust_normalization"
    """

    if normalization_function == "robust_normalization":
        from tomocpt.dataManager.dataUtils import robust_normalization
        normalization_function = robust_normalization
    else:
        raise NotImplementedError(f"We only have robust_normalization and you used {normalization_function}")

    vol, voxel_size = load_mrc(data_fname, normalize=normalization_function, return_boxSize=True)
    particle_size_pix = particle_radius_angst / voxel_size
    old_shape = vol.shape
    vol = torch.tensor(vol)
    scalar, new_shape = get_shape_for_resizing(vol, particle_size_pix, new_size=new_particle_size)
    if vol.shape != tuple(new_shape):
        # print("\n")
        logger.debug(f"Resizing {data_fname} from {tuple(vol.shape)} to {new_shape}, such that the particle size is {new_particle_size}px.")
        vol, padding_values = resize_volume(vol, new_shape, chunk_size=chunk_size, use_gpu=use_gpu)
    else:
        padding_values = None
    return vol, new_shape, old_shape, voxel_size, padding_values, scalar


@functools.cache
def get_labels_dirname(require_labels:bool):
    if require_labels:
        return constants.LABELS_DIR_NAME_PREFIX%"_supervised"
    else:
        return constants.LABELS_DIR_NAME_PREFIX%"_selfSup"


def _getRange(origin: int, shape_on_axis: int, chunk_size: int, stride: int, randomFractionToTake: float = -1):
    """

  :param origin:
  :param shape_on_axis:
  :param chunk_size:
  :param stride:
  :param randomFractionToTake:
  :return:
  """

    raw_range = range(origin, shape_on_axis - (chunk_size - 1), stride)
    if 0 < randomFractionToTake < 1:
        actual_range = np.random.choice(raw_range, int(len(raw_range) * randomFractionToTake), replace=False)
        return actual_range
    else:
        return raw_range


def get_vol_chunks(volume: np.array, label: np.array,
                   chunk_size: int,
                   stride: int,
                   prepare_for_training=False):
    """
    :param volume: A numpy array representing an EM map
    :param label:  A numpy array representing an EM target (e.g. tightMask)
    :param chunk_size: Size of the chunks to be extracted
    :param stride:  Stride between two consecutive cubes
    :param prepare_for_training: If activated, cubes outside the sphere are rejected. It also samples randomly some cubes
    :return:
    """

    volume_shape = np.array(volume.shape)
    if prepare_for_training:
        i_origin, j_origin, k_origin = np.random.randint(0, constants.CHUNK_STRIDE, 3)  # torch.random.randint
        randomFractionToTake = constants.RANDOM_FRACTION_TO_SAMPLE_TRAIN if constants.RANDOM_FRACTION_TO_SAMPLE_TRAIN < 1 else -1
    else:
        i_origin, j_origin, k_origin = (0, 0, 0)
        randomFractionToTake = -1

    def _partial_getRange(origin, shape_on_axis):  # This is to avoid repeating the last 3 arguments of _getRange
        return _getRange(origin, shape_on_axis, chunk_size, stride, randomFractionToTake)

    for i in _partial_getRange(i_origin, volume_shape[-3]):
        for j in _partial_getRange(j_origin, volume_shape[-2]):
            for k in _partial_getRange(k_origin, volume_shape[-1]):
                volume_chunk = volume[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                label_chunk = label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                yield (volume_chunk, label_chunk), (i, j, k)
