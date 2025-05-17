import functools
import numpy as np
import torch
from tomocpt import constants
from tomocpt.dataManager.dataUtils import load_mrc, get_shape_for_resizing, resize_volume
from tomocpt.logger import get_logger

logger = get_logger()


def _preprocess_data_mrc(data_fname: str, particle_radius_angst, normalization_function,
                         new_particle_size, chunk_size, use_gpu):
    """Super simplified preprocessing function that avoids device variable"""
    logger = get_logger()
    
    if normalization_function == "robust_normalization":
        from tomocpt.dataManager.dataUtils import robust_normalization
        normalization_function = robust_normalization
    else:
        raise NotImplementedError(f"We only have robust_normalization and you used {normalization_function}")

    # Determine if GPU is available
    use_cuda = use_gpu and torch.cuda.is_available()

    # Load data (always on CPU initially)
    vol, voxel_size = load_mrc(data_fname, normalize=normalization_function, return_boxSize=True)
    particle_size_pix = particle_radius_angst / voxel_size
    old_shape = vol.shape

    # Create tensor
    vol_tensor = torch.tensor(vol)
    
    # Move to GPU if requested
    if use_cuda:
        try:
            vol_tensor = vol_tensor.cuda()
        except RuntimeError as e:
            logger.warning(f"Failed to move tensor to GPU: {str(e)}")
            use_cuda = False

    # Calculate new shape
    scalar, new_shape = get_shape_for_resizing(vol_tensor, particle_size_pix, new_size=new_particle_size)

    # Resize if needed
    if vol_tensor.shape != tuple(new_shape):
        logger.debug(
            f"Resizing {data_fname} from {tuple(vol_tensor.shape)} to {new_shape}, such that the particle size is {new_particle_size}px."
        )
        try:
            vol_tensor, padding_values = resize_volume(vol_tensor, new_shape, chunk_size=chunk_size, use_gpu=use_cuda)
        except RuntimeError as e:
            logger.warning(f"Error during resize: {str(e)}")
            if vol_tensor.device.type != 'cpu':
                vol_tensor = vol_tensor.cpu()
            vol_tensor, padding_values = resize_volume(vol_tensor, new_shape, chunk_size=chunk_size, use_gpu=False)
    else:
        padding_values = None

    return vol_tensor, new_shape, old_shape, voxel_size, padding_values, scalar

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
    """Process volume chunks with simplified device handling"""
    import torch
    import numpy as np
    from tomocpt import constants
    
    # Avoid torch.device objects entirely - just use strings and direct CUDA methods
    gpu_available = torch.cuda.is_available()
    
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

    # Check if we're on GPU
    using_gpu = gpu_available and ((isinstance(volume, torch.Tensor) and volume.device.type == 'cuda') or 
                                 (device is not None and (
                                     (isinstance(device, str) and device.startswith('cuda')) or
                                     (isinstance(device, int) and device >= 0)
                                 )))

    # Process based on whether we're using GPU or not
    if using_gpu:
        # Generate all coordinates first
        coords_list = []
        for i in _partial_getRange(i_origin, volume_shape[-3]):
            for j in _partial_getRange(j_origin, volume_shape[-2]):
                for k in _partial_getRange(k_origin, volume_shape[-1]):
                    coords_list.append((i, j, k))

        # Process coordinates in batches to manage memory
        batch_size = 4  # Small fixed batch size to be safe
        for batch_idx in range(0, len(coords_list), batch_size):
            batch_coords = coords_list[batch_idx:batch_idx + batch_size]

            # Process each coordinate in batch
            with torch.no_grad():
                for i, j, k in batch_coords:
                    # Get tensor slices ensuring they're on GPU
                    if isinstance(volume, torch.Tensor):
                        if volume.device.type == 'cuda':
                            volume_chunk = volume[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                        else:
                            volume_chunk = volume[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size].cuda()
                            
                        if isinstance(label, torch.Tensor):
                            if label.device.type == 'cuda':
                                label_chunk = label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                            else:
                                label_chunk = label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size].cuda()
                        else:
                            label_slice = label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                            label_chunk = torch.tensor(label_slice).cuda()
                    else:
                        vol_slice = volume[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                        label_slice = label[..., i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                        
                        volume_chunk = torch.tensor(vol_slice).cuda()
                        label_chunk = torch.tensor(label_slice).cuda()

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
