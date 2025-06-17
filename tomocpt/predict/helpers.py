import logging
import re
from pathlib import Path
from typing import Union, List, Tuple

import torch
import numpy as np
import pandas as pd
import starfile
from skimage.feature import peak_local_max
from skimage.morphology import cube

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
                         output_filename: str = "tomocpt_coords.star") -> Path:
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
    # Initialize the data dictionary for particle coordinates
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
                         output_filename: str = "tomocpt_coords.star") -> Path:
    return None


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


def extract_centroids_from_pred(input_tensor: np.array, nn_ang_distance: float, particle_ang_length: float,
                                threshold: float, angpix: float):
    if nn_ang_distance is None:
        min_dist = particle_ang_length / angpix
    else:
        min_dist = nn_ang_distance / angpix
    kernel = cube(3)
    centroid_peaks = peak_local_max(input_tensor, threshold_abs=threshold, footprint=kernel, min_distance=int(min_dist),
                                    exclude_border=True)
    return centroid_peaks

def _refine_peak_subpixel(peak_coord: np.ndarray, pred_map: np.ndarray, patch_size: int = 5) -> Tuple[np.ndarray, float]:
    """
    Refines an integer peak coordinate to sub-pixel accuracy using a 3D quadratic fit.
    """
    cz, cy, cx = peak_coord.astype(int)
    patch_radius = patch_size // 2
    
    # Ensure patch does not go out of bounds
    z_min, z_max = cz - patch_radius, cz + patch_radius + 1
    y_min, y_max = cy - patch_radius, cy + patch_radius + 1
    x_min, x_max = cx - patch_radius, cx + patch_radius + 1
    if not (z_min >= 0 and z_max <= pred_map.shape[0] and y_min >= 0 and y_max <= pred_map.shape[1] and x_min >= 0 and x_max <= pred_map.shape[2]):
        return peak_coord.astype(float), pred_map[cz, cy, cx]

    patch = pred_map[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Use log of values for better numerical stability with Gaussian-like peaks
    patch_log = np.log(np.maximum(patch, 1e-6)) # Add epsilon to avoid log(0)

    # Create coordinate system relative to the patch center
    z, y, x = np.mgrid[-patch_radius:patch_radius+1, -patch_radius:patch_radius+1, -patch_radius:patch_radius+1]
    
    # Design matrix for quadratic: z^2, y^2, x^2, zy, zx, yx, z, y, x, 1
    A = np.vstack([z.ravel()**2, y.ravel()**2, x.ravel()**2, 
                   z.ravel()*y.ravel(), z.ravel()*x.ravel(), y.ravel()*x.ravel(),
                   z.ravel(), y.ravel(), x.ravel(), np.ones(x.size)]).T
    b = patch_log.ravel()

    try:
        # Solve for the coefficients of the quadratic
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b, c, d, e, f, g, h, i, _ = coeffs

        # Hessian matrix for finding the peak
        H = np.array([[2*a, d, e], [d, 2*b, f], [e, f, 2*c]])
        H_inv = np.linalg.inv(H)
        
        # Gradient vector
        grad = -np.array([g, h, i])
        
        # Sub-pixel offset from the patch center
        offset = H_inv @ grad
        
        # Ensure the peak is within a reasonable distance (e.g., 1 voxel from center)
        if np.max(np.abs(offset)) > 1.0:
            return peak_coord.astype(float), pred_map[cz, cy, cx]
        
        refined_coord = peak_coord.astype(float) + offset
        refined_score = np.exp(np.polyval(coeffs, offset)) # Use fitted value as score

        return refined_coord, refined_score
    except np.linalg.LinAlgError:
        # Fallback to integer coordinate if fit fails
        return peak_coord.astype(float), pred_map[cz, cy, cx]


def _infer_one_tomo(tomo_fname: str, model: torch.nn.Module, device: torch.device) -> Tuple[np.ndarray, float]:
    """Performs inference on a single tomogram."""
    infer_conf = mainConfig.infer
    prep_conf = mainConfig.prepData
    original_vol, original_voxel_size = load_mrc(tomo_fname, normalize=False, return_voxel_size=True)
    original_shape = original_vol.shape
    del original_vol

    processed_vol, _, _, _ = preprocess_volume(
        mrc_path=tomo_fname, particle_diameter_angst=infer_conf.length,
        target_particle_px=prep_conf.desired_particle_pixel_size,
        chunk_size=model.patch_size, device=device, is_label=False)

    subject = tio.Subject(volume=tio.ScalarImage(tensor=processed_vol.unsqueeze(0)))
    grid_sampler = tio.GridSampler(subject, patch_size=model.patch_size, patch_overlap=model.patch_size // infer_conf.PATCH_OVERLAP_FACTOR)
    patch_loader = DataLoader(grid_sampler, batch_size=infer_conf.predictions_batch_size)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")

    with torch.inference_mode():
        for batch in patch_loader:
            input_tensor = batch['volume'][tio.DATA].to(device)
            with torch.amp.autocast(device_type=device.type):
                predictions = model.predict_step(input_tensor, 0)
            aggregator.add_batch(predictions, batch[tio.LOCATION])

    aggregated_mask = aggregator.get_output_tensor().squeeze(0)
    unpadded_mask_cpu = resize_volume(aggregated_mask.cpu(), original_shape, torch.device('cpu'))
    del aggregated_mask
    return unpadded_mask_cpu.numpy(), original_voxel_size


def infer_tomos(tomo_fnames: List[Path], gpu_id: Optional[int], model_fname: str, particle_length_ang: float,
                preds_dir: str, save_pred_mask: bool, extract_coords: bool, threshold: float,
                nearest_neigs_angs: Optional[float], masks_dir: Optional[str]) -> Tuple[List[str], List[List[float]], List[float]]:
    """Worker function for inference, now with sub-pixel refinement."""
    from tomocpt.networks.pickingModel import BasePickingModel
    device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None and torch.cuda.is_available() else torch.device("cpu")
    if gpu_id is not None: torch.cuda.set_device(device)
    
    model = BasePickingModel.load_from_checkpoint(model_fname, map_location=device).eval()

    results = []
    for tomo_fname in tqdm(tomo_fnames, desc=f"Device {device}", position=gpu_id or 0):
        pred_mask, voxel_size = _infer_one_tomo(str(tomo_fname), model, device)
        if masks_dir and (mask_path := Path(masks_dir) / tomo_fname.name).exists():
            pred_mask *= load_mrc(str(mask_path), normalize=False)
        if save_pred_mask:
            write_segmentation_mask(pred_mask, str(Path(preds_dir) / tomo_fname.name), angpix=voxel_size, overwrite=True)
        if extract_coords:
            int_coords = peak_local_max(pred_mask, min_distance=int((nearest_neigs_angs or particle_length_ang) / voxel_size), 
                                        threshold_abs=threshold, exclude_border=True)
            if int_coords.shape[0] > 0:
                refined_coords_and_scores = [_refine_peak_subpixel(peak, pred_mask) for peak in int_coords]
                final_coords = [c for c, s in refined_coords_and_scores]
                final_scores = [s for c, s in refined_coords_and_scores]
                centroids_with_scores = np.hstack([np.array(final_coords), np.array(final_scores)[:, np.newaxis]])
                results.append((tomo_fname.stem, centroids_with_scores.tolist(), voxel_size))
        gc.collect()

    all_names, all_centroids, all_voxel_sizes = [], [], []
    for name, centroids, vx in results:
        all_names.extend([name] * len(centroids))
        all_centroids.extend(centroids)
        all_voxel_sizes.extend([vx] * len(centroids))
    return all_names, all_centroids, all_voxel_sizes
