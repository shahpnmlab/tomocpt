import torch

from tomocpt.dataManager.dataUtils import load_mrc, get_shape_for_resizing, resize_volume


def _preprocess_data_mrc(data_fname:str, particle_size_angst, normalization_function, new_particle_size, chunk_size, use_gpu):
    if normalization_function == "robust_normalization":
        from pycotool.dataManager.dataUtils import robust_normalization
        normalization_function = robust_normalization
    else:
        raise NotImplementedError(f"We only have robust_normalization and you used {normalization_function}")

    vol, voxel_size = load_mrc(data_fname, normalize=normalization_function, return_boxSize=True)
    particle_size_pix = particle_size_angst / voxel_size #TODO: pranav check if this is correct
    old_shape = vol.shape
    vol = torch.tensor(vol)
    scalar, new_shape = get_shape_for_resizing(vol, particle_size_pix, new_size=new_particle_size)
    if vol.shape != tuple(new_shape):
        print("\n")
        print(f"Resizing {data_fname} from {tuple(vol.shape)} to {new_shape}, such that the particle size is {new_particle_size}px.")
        vol, padding_values = resize_volume(vol, new_shape, chunk_size=chunk_size, use_gpu=use_gpu)
    else:
        padding_values = None
    return vol, new_shape, old_shape, voxel_size, padding_values, scalar
