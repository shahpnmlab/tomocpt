import functools
import numpy as np
import torch
from tomocpt import constants
from tomocpt.dataManager.dataUtils import load_mrc, get_shape_for_resizing, resize_volume
from tomocpt.logger import get_logger

logger = get_logger()


def _preprocess_data_mrc(data_fname: str, particle_radius_angst, normalization_function,
                         new_particle_size, chunk_size, use_gpu, device=None):
    """
    Enhanced preprocessing with specific GPU device support

    Args:
        data_fname: Path to input MRC file
        particle_radius_angst: Particle radius in Angstroms
        normalization_function: Normalization method
        new_particle_size: Target size in pixels
        chunk_size: Size of chunks
        use_gpu: Whether to use GPU acceleration
        device: Specific device to use (None = auto-select)

    Returns:
        Tuple containing processed data
    """
    if normalization_function == "robust_normalization":
        from tomocpt.dataManager.dataUtils import robust_normalization
        normalization_function = robust_normalization
    else:
        raise NotImplementedError(f"We only have robust_normalization and you used {normalization_function}")

    # Set specific device if provided
    if use_gpu and torch.cuda.is_available():
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}"

        # Ensure we're using the right device
        if isinstance(device, str) and device.startswith("cuda:"):
            device_id = int(device.split(":")[1])
            torch.cuda.set_device(device_id)

    vol, voxel_size = load_mrc(data_fname, normalize=normalization_function, return_boxSize=True)
    particle_size_pix = particle_radius_angst / voxel_size
    old_shape = vol.shape

    # Use device-specific tensor creation
    if use_gpu and torch.cuda.is_available():
        vol = torch.tensor(vol, device=device)
    else:
        vol = torch.tensor(vol)

    scalar, new_shape = get_shape_for_resizing(vol, particle_size_pix, new_size=new_particle_size)

    if vol.shape != tuple(new_shape):
        logger.debug(
            f"Resizing {data_fname} from {tuple(vol.shape)} to {new_shape}, such that the particle size is {new_particle_size}px.")
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


def get_vol_chunks(volume, label, chunk_size, stride, prepare_for_training=False, device=None):
    """Process volume chunks with proper device handling"""
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Get volume shape
    volume_shape = volume.shape

    # Handle training vs. inference
    if prepare_for_training:
        if constants.RANDOM_FRACTION_TO_SAMPLE_TRAIN < 1:
            i_origin, j_origin, k_origin = torch.randint(0, constants.CHUNK_STRIDE, (3,)).tolist()
            randomFractionToTake = constants.RANDOM_FRACTION_TO_SAMPLE_TRAIN
        else:
            i_origin, j_origin, k_origin = (0, 0, 0)
            randomFractionToTake = -1
    else:
        i_origin, j_origin, k_origin = (0, 0, 0)
        randomFractionToTake = -1

    # Range calculation helper
    def _partial_getRange(origin, shape_on_axis):
        return _getRange(origin, shape_on_axis, chunk_size, stride, randomFractionToTake)

    # Process in batches for GPU efficiency
    if device.type == 'cuda':
        # Adjust batch size based on GPU memory
        free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        batch_size = max(1, min(8, int(free_memory / (2 ** 30))))  # Rough estimate based on GB free

        # Generate all coordinates first
        coords_list = []
        for i in _partial_getRange(i_origin, volume_shape[-3]):
            for j in _partial_getRange(j_origin, volume_shape[-2]):
                for k in _partial_getRange(k_origin, volume_shape[-1]):
                    coords_list.append((i, j, k))

        # Process coordinates in batches to manage memory
        for batch_idx in range(0, len(coords_list), batch_size):
            batch_coords = coords_list[batch_idx:batch_idx + batch_size]

            # Process each coordinate in batch
            with torch.amp.autocast(device.type):
                for i, j, k in batch_coords:
                    # Get tensor slices ensuring they're on the right device
                    if isinstance(volume, torch.Tensor):
                        volume_chunk = volume[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size].to(device)
                        label_chunk = label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size].to(device)
                    else:
                        vol_slice = volume[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                        lab_slice = label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]

                        volume_chunk = torch.tensor(vol_slice, device=device)
                        label_chunk = torch.tensor(lab_slice, device=device)

                    yield (volume_chunk, label_chunk), (i, j, k)

            # Clear cache after batch
            torch.cuda.empty_cache()
    else:
        # CPU processing
        for i in _partial_getRange(i_origin, volume_shape[-3]):
            for j in _partial_getRange(j_origin, volume_shape[-2]):
                for k in _partial_getRange(k_origin, volume_shape[-1]):
                    if isinstance(volume, torch.Tensor):
                        volume_chunk = volume[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                        label_chunk = label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                    else:
                        volume_chunk = torch.tensor(volume[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size])
                        label_chunk = torch.tensor(label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size])

                    yield (volume_chunk, label_chunk), (i, j, k)