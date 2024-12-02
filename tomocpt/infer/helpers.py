import gc
import logging
import re
from pathlib import Path
from typing import Union, List, Tuple, Optional

import torch
import torchio as tio
import numpy as np
import pandas as pd
import starfile
from skimage.feature import peak_local_max
from skimage.morphology import cube
from torch.utils.data import DataLoader
from tqdm import tqdm

from tomocpt.dataManager.dataUtils import load_mrc, write_segmentation_mask, resize_volume, plot_example
from tomocpt.dataPreparation.helpers import _preprocess_data_mrc
from tomocpt.mainConfig import mainConfig

infer_config = mainConfig.infer

logger = logging.getLogger(__name__)


def write_relion_star_file(output_dir: str,
                           tomo_names: List[str],
                           predicted_centroids_with_scores: List[List[float]],
                           voxel_size: float,
                           output_filename: str = "tomocpt_coords.star",
                           version: Union[float, int] = 3.1) -> Path:
    """
    Write particle coordinates and scores to a Relion-compatible star file format.

    Args:
        output_dir: Directory where the star file will be saved
        tomo_names: List of tomogram names
        predicted_centroids_with_scores: List of [x, y, z, score] coordinates and scores
        voxel_size: Pixel size in Angstroms
        output_filename: Name of the output star file
        version: Relion version (3.1 or 5.0)

    Returns:
        Path: Path to the written star file

    Raises:
        ValueError: If an unsupported Relion version is specified
    """
    # Determine the correct label based on Relion version
    if version == 3.1:
        tomo_label = "rlnMicrographName"
    elif version == 5.0:
        tomo_label = "rlnTomoName"
    else:
        raise ValueError(f"Unsupported Relion version: {version}. Supported versions are 3.1 and 5.0")

    # Create optics table
    df_optics = pd.DataFrame({
        'rlnOpticsGroup': [1],
        "rlnOpticsGroupName": ["OpticsGroup1"],
        'rlnSphericalAberration': [2.7],
        'rlnVoltage': [300],
        'rlnImagePixelSize': [voxel_size],
        'rlnImageDimensionality': [3]
    })

    # Initialize particles data dictionary
    all_tomo_centroids_and_scores = {
        tomo_label: [],
        "rlnCoordinateX": [],
        "rlnCoordinateY": [],
        "rlnCoordinateZ": [],
        "rlnAutopickFigureOfMerit": []
    }

    # Process each coordinate
    for tomoName, predicted_centroid in zip(tomo_names, predicted_centroids_with_scores):
        # Add data to the dictionary
        all_tomo_centroids_and_scores[tomo_label].append(tomoName)
        all_tomo_centroids_and_scores["rlnCoordinateX"].append(predicted_centroid[2])
        all_tomo_centroids_and_scores["rlnCoordinateY"].append(predicted_centroid[1])
        all_tomo_centroids_and_scores["rlnCoordinateZ"].append(predicted_centroid[0])
        all_tomo_centroids_and_scores["rlnAutopickFigureOfMerit"].append(predicted_centroid[3])

    # Create particles table
    df_particles = pd.DataFrame(data=all_tomo_centroids_and_scores)

    # Prepare star file data
    star_data = {
        'optics': df_optics,
        'particles': df_particles
    }

    # Write the star file
    output_path = Path(output_dir) / output_filename
    starfile.write(star_data, output_path, float_format="%0.2f", overwrite=True)
    logger.info(f"Predicted coordinates are stored here: {output_path}")

    return output_path


def write_warp_star_file(output_dir: str,
                         tomo_names: List[str],
                         predicted_centroids_with_scores: List[List[float]],
                         output_filename: str | None = None) -> Path:
    """
    Write particle coordinates and scores to a WARP-compatible star file format.

    Args:
        output_dir: Directory where the star file will be saved
        tomo_names: List of tomogram names
        predicted_centroids_with_scores: List of [x, y, z, score] coordinates and scores
        voxel_size: Pixel size in Angstroms
        output_filename: Name of the output star file

    Returns:
        Path: Path to the written star file
    """
    # Initialize the data dictionary for particle coordinatesç
    if output_filename is None:
        output_filename = mainConfig.infer.outCoordFname

    all_tomo_centroids_and_scores = {
        "rlnMicrographName": [],
        "rlnCoordinateX": [],
        "rlnCoordinateY": [],
        "rlnCoordinateZ": [],
        "rlnAutopickFigureOfMerit": []
    }

    # Process each coordinate
    for tomoName, predicted_centroid in zip(tomo_names, predicted_centroids_with_scores):
        # Convert filename format
        tomoName = re.sub(r'_\d+\.\d+Apx', '.tomostar', tomoName)

        # Add data to the dictionary
        all_tomo_centroids_and_scores["rlnMicrographName"].append(tomoName)
        all_tomo_centroids_and_scores["rlnCoordinateX"].append(predicted_centroid[2])
        all_tomo_centroids_and_scores["rlnCoordinateY"].append(predicted_centroid[1])
        all_tomo_centroids_and_scores["rlnCoordinateZ"].append(predicted_centroid[0])
        all_tomo_centroids_and_scores["rlnAutopickFigureOfMerit"].append(predicted_centroid[3])

    # Create particles table
    df_particles = pd.DataFrame(data=all_tomo_centroids_and_scores)

    # Prepare star file data
    star_data = {
        'particles': df_particles
    }

    # Write the star file
    output_path = Path(output_dir) / output_filename
    starfile.write(star_data, output_path, float_format="%0.2f", overwrite=True)
    logger.info(f"Predicted coordinates are stored here: {output_path}")

    return output_path


def write_sg_motive_list(output_dir: str,
                         tomo_names: List[str],
                         predicted_centroids_with_scores: List[List[float]],
                         output_filename: str = "tomocpt_coords.star") -> Path:  # TODO: This is incomple
    raise NotImplementedError()


def apply_mask(tomo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to a tomogram.

    :param tomo: The tomogram as a numpy array
    :param mask: The mask as a numpy array
    :return: The masked tomogram
    """
    if tomo.shape != mask.shape:
        raise ValueError(f"Tomogram shape {tomo.shape} does not match mask shape {mask.shape}")
    return tomo * mask


def unpad(inp_tensor: torch.Tensor, padding_values: Tuple):
    if padding_values is None:
        padding_values = (0, 0, 0, 0, 0, 0)
    _padding_values = list(reversed(padding_values))
    for i in range(3):
        _i = 2 * i
        if _padding_values[_i] == 0 and _padding_values[_i + 1] == 0:
            _padding_values[_i] = 0
            _padding_values[_i + 1] = inp_tensor.shape[i]
        else:
            _padding_values[_i + 1] = -_padding_values[_i + 1]

    inp_tensor = inp_tensor[_padding_values[0]:_padding_values[1],
                 _padding_values[2]:_padding_values[3],
                 _padding_values[4]:_padding_values[5]]
    return inp_tensor


def extract_cube_at_predicted_centroid(predicted_segmentation_mask: np.array, centroid: np.array):
    x_min, x_max = centroid[0] - 1, centroid[0] + 1
    y_min, y_max = centroid[1] - 1, centroid[1] + 1
    z_min, z_max = centroid[2] - 1, centroid[2] + 1
    subvol = predicted_segmentation_mask[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    return np.max(subvol)


def extract_centroids_from_pred(input_tensor: np.array, nn_ang_distance: Optional[float], particle_ang_length: float,
                                threshold: float, angpix: float):
    if nn_ang_distance is None:
        min_dist = particle_ang_length / angpix
    else:
        min_dist = nn_ang_distance / angpix
    kernel = cube(3)
    centroid_peaks = peak_local_max(input_tensor, threshold_abs=threshold, footprint=kernel, min_distance=int(min_dist),
                                    exclude_border=True)
    return centroid_peaks


def _infer_one_tomo(tomoFname: str, output_fname: str, particle_ang_length: float, model: torch.nn.Module, gpu_id: int,
                    patch_size: int, batch_size: int,
                    plot: bool, save_pred_mask: bool, extract_coords: bool,
                    nearest_neigs_angs: Optional[float], threshold: float, maskFname: Optional[str]):
    """
    :param tomoFname: the filename with the tomogram
    :param output_fname: path to directory to store inferred output
    :param particle_ang_length: particle dimension (Å)
    :param model: Path to trained model weights
    :param gpu_id: The gpu_id
    :param patch_size: Size of chunk to run inference on
    :param batch_size: batch size. Number of chunks to process together in the GPU
    :param plot: plot raw inference_data and segmentation mask for quick viz.
    :param save_pred_mask: whether to save the predicted segmentation mask
    :param extract_coords: whether to extract coordinates from the predicted segmentation mask
    :param nearest_neigs_angs: nearest neighbor distance (Å
    :param threshold: threshold for peak detection
    :param maskFname: filename of the mask to apply (optional)
    :return:
    """

    voxel_size = np.nan
    if not Path(output_fname).exists():
        (vol, new_shape, old_shape, voxel_size,
         padding_values, scalar) = _preprocess_data_mrc(tomoFname, normalization_function="robust_normalization",
                                                        new_particle_size=model.DESIRED_PARTICLE_PIXELS,
                                                        particle_size_angst=particle_ang_length,
                                                        chunk_size=patch_size,
                                                        use_gpu=infer_config.USE_CUDA_FOR_DATA)
        if maskFname and Path(maskFname).exists():
            logger.info("Loading mask")
            try:
                m = load_mrc(maskFname, normalize=None, return_boxSize=False)
                mask, _ = resize_volume(m, new_shape=new_shape, chunk_size=patch_size,
                                        use_gpu=infer_config.USE_CUDA_FOR_DATA, mean_as_padding_value=False)
                mask = torch.as_tensor(mask, dtype=vol.dtype)
                if mask.shape != vol.shape:
                    raise ValueError(f"Resized mask shape {mask.shape} does not match volume shape {vol.shape}")
                vol = vol * mask
                del mask
                gc.collect()
            except Exception as e:
                logger.warning(f"Error processing mask: {str(e)}. Using the whole volume for inference.")
        else:
            logger.warning(
                f"No mask provided or mask file not found for {Path(tomoFname).name}. Using the whole volume for inference.")
        vol = vol.unsqueeze(0)
        subject = tio.Subject({"input_data": tio.ScalarImage(tensor=vol)})

        grid_sampler = tio.GridSampler(subject, patch_size=patch_size, patch_overlap=patch_size // mainConfig.infer.patch_overlap_factor,
                                       padding_mode='reflect')
        patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
        del vol
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")

        with torch.inference_mode():
            for idx, patches_batch in enumerate(tqdm(patch_loader)):
                input_tensor = patches_batch['input_data'][tio.DATA].cuda(device=gpu_id)
                locations = patches_batch[tio.LOCATION]
                outputs = model.predict_step(input_tensor, idx)
                aggregator.add_batch(outputs, locations)

        output_tensor = aggregator.get_output_tensor()
        output_tensor = output_tensor.squeeze(0)
        output_tensor = unpad(output_tensor, padding_values)
        output_tensor = output_tensor.cpu().numpy()
        # output_tensor = resize(output_tensor, output_shape=old_shape, mode='constant', cval=0)
        output_tensor, sym_padding = resize_volume(output_tensor, new_shape=old_shape, chunk_size=patch_size,
                                                   use_gpu=gpu_id >= 0, mean_as_padding_value=False)
        if save_pred_mask:
            write_segmentation_mask(output_tensor, output_fname, angpix=voxel_size, overwrite=True)
        if plot:
            plot_example(subject["input_data"][tio.DATA], output_tensor)

    elif extract_coords:
        output_tensor, voxel_size = load_mrc(output_fname, normalize=None, return_boxSize=True)
        logger.info(f"Mask is already available for {output_fname}")
    else:
        return [], [], voxel_size

    if extract_coords:
        coords_array = extract_centroids_from_pred(output_tensor,
                                                   nn_ang_distance=nearest_neigs_angs,
                                                   particle_ang_length=particle_ang_length,
                                                   threshold=threshold, angpix=voxel_size)
        tomoFnameList = []
        centroids_and_scores = []
        if len(coords_array) == 0:
            logger.warning(f"There are no particles to be found in {output_fname.split('/')[-1]}")
            pass
        else:
            centroids_and_scores = np.zeros((coords_array.shape[0], 4))
            for i, centroid_peak in enumerate(coords_array):
                tomoFnameList.append(Path(tomoFname).stem)
                centroids_and_scores[i, 0] = centroid_peak[0]
                centroids_and_scores[i, 1] = centroid_peak[1]
                centroids_and_scores[i, 2] = centroid_peak[2]
                centroids_and_scores[i, 3] = extract_cube_at_predicted_centroid(output_tensor, centroid_peak)
            centroids_and_scores = centroids_and_scores.tolist()
        return tomoFnameList, centroids_and_scores, voxel_size
    return [], [], voxel_size
