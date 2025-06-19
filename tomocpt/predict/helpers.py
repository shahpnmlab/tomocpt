import gc
import re
import warnings
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any

import mrcfile
import numpy as np
import pandas as pd
import starfile
import imodmodel
import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm

from tomocpt.dataManager.preprocessing import (
    preprocess_tomogram,
    resize_volume,
    get_mrc_metadata,
    load_mrc,
    symmetrize_and_pad
)
from tomocpt.logger import get_logger
from tomocpt.mainConfig import mainConfig
from tomocpt.defaultConfigs.infer_config import InferConfig, OutputFormat

logger = get_logger()

def unpad(inp_tensor: torch.Tensor, padding_values: Tuple[int, ...]) -> torch.Tensor:
    """
    Removes padding from a tensor given the padding values from F.pad.
    The padding tuple is expected in the order (pad_x1, pad_x2, pad_y1, pad_y2, ...).
    """
    if padding_values is None or all(p == 0 for p in padding_values):
        return inp_tensor

    # PyTorch's padding format is (X_begin, X_end, Y_begin, Y_end, Z_begin, Z_end)
    # We need to slice from the end of the begin-padding to the beginning of the end-padding.
    z_pad_begin, z_pad_end, y_pad_begin, y_pad_end, x_pad_begin, x_pad_end = padding_values

    z_slice = slice(z_pad_begin, inp_tensor.shape[0] - z_pad_end)
    y_slice = slice(y_pad_begin, inp_tensor.shape[1] - y_pad_end)
    x_slice = slice(x_pad_begin, inp_tensor.shape[2] - x_pad_end)

    return inp_tensor[z_slice, y_slice, x_slice]


def write_segmentation_mask(tensor_mask, outputname: str, angpix: float = 1.0, overwrite=True):
    """Saves a numpy array or torch tensor as an MRC file."""
    if isinstance(tensor_mask, torch.Tensor):
        m = tensor_mask.cpu().detach().numpy()
    else:
        m = tensor_mask
    output_path = Path(outputname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing segmentation mask to {output_path}")
    mrcfile.write(output_path, data=m.astype(np.float32), voxel_size=angpix, overwrite=overwrite)


def write_imod_model(
        output_dir: str,
        tomo_name: str,
        predicted_centroids_with_scores: List[List[float]],
        output_filename: str,
):
    """
    Writes particle coordinates to an IMOD model file (.mod) using the
    high-level pandas DataFrame API.
    """
    if not predicted_centroids_with_scores:
        logger.warning(f"No coordinates found for {tomo_name}, skipping .mod file creation.")
        return

    df = pd.DataFrame(predicted_centroids_with_scores, columns=['z', 'y', 'x', 'score'])
    df_for_imod = df[['x', 'y', 'z']]

    output_path = Path(output_dir) / output_filename
    try:
        imodmodel.write(df_for_imod, str(output_path))
        logger.info(f"IMOD model for {tomo_name} stored in: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write IMOD model file for {tomo_name}: {e}")


def write_relion_star_file(output_dir: str,
                           tomo_names: List[str],
                           predicted_centroids_with_scores: List[List[float]],
                           voxel_size: float,
                           output_filename: str = "tomocpt_coords.star",
                           version: Union[float, int] = 3.1) -> Path:
    """Write particle coordinates to a Relion-compatible star file."""
    if version == 3.1:
        tomo_label = "rlnMicrographName"
    elif version == 5.0:
        tomo_label = "rlnTomoName"
    else:
        raise ValueError(f"Unsupported Relion version: {version}. Supported versions are 3.1 and 5.0")

    df_optics = pd.DataFrame({
        'rlnOpticsGroup': [1], "rlnOpticsGroupName": ["OpticsGroup1"],
        'rlnSphericalAberration': [2.7], 'rlnVoltage': [300],
        'rlnImagePixelSize': [voxel_size], 'rlnImageDimensionality': [3]
    })
    particles_data = {
        tomo_label: [], "rlnCoordinateX": [], "rlnCoordinateY": [],
        "rlnCoordinateZ": [], "rlnAutopickFigureOfMerit": []
    }
    for tomoName, centroid in zip(tomo_names, predicted_centroids_with_scores):
        particles_data[tomo_label].append(tomoName)
        particles_data["rlnCoordinateX"].append(centroid[2])
        particles_data["rlnCoordinateY"].append(centroid[1])
        particles_data["rlnCoordinateZ"].append(centroid[0])
        particles_data["rlnAutopickFigureOfMerit"].append(centroid[3])

    df_particles = pd.DataFrame(data=particles_data)
    star_data = {'optics': df_optics, 'particles': df_particles}
    output_path = Path(output_dir) / output_filename
    starfile.write(star_data, output_path, float_format="%.2f", overwrite=True)
    logger.info(f"Predicted coordinates are stored in: {output_path}")
    return output_path


def write_warp_star_file(output_dir: str,
                         tomo_names: List[str],
                         predicted_centroids_with_scores: List[List[float]],
                         output_filename: str = "tomocpt_coords.star") -> Path:
    """Write particle coordinates to a WARP-compatible star file."""
    particles_data = {
        "rlnMicrographName": [], "rlnCoordinateX": [], "rlnCoordinateY": [],
        "rlnCoordinateZ": [], "rlnAutopickFigureOfMerit": []
    }
    for tomoName, centroid in zip(tomo_names, predicted_centroids_with_scores):
        tomoName_warp = re.sub(r'_\d+\.\d+Apx', '.tomostar', tomoName)
        particles_data["rlnMicrographName"].append(tomoName_warp)
        particles_data["rlnCoordinateX"].append(centroid[2])
        particles_data["rlnCoordinateY"].append(centroid[1])
        particles_data["rlnCoordinateZ"].append(centroid[0])
        particles_data["rlnAutopickFigureOfMerit"].append(centroid[3])

    df_particles = pd.DataFrame(data=particles_data)
    output_path = Path(output_dir) / output_filename
    starfile.write({'particles': df_particles}, output_path, float_format="%.2f", overwrite=True)
    logger.info(f"Predicted coordinates are stored in: {output_path}")
    return output_path


def peak_local_max_torch(
        image: torch.Tensor,
        min_distance: int,
        threshold_abs: float,
        exclude_border: bool = True,
) -> torch.Tensor:
    """
    A corrected and vectorized PyTorch implementation of scikit-image's peak_local_max.
    Finds peaks in an image as coordinate list.
    """
    device = image.device

    # Non-maximum suppression
    max_pooled = torch.nn.functional.max_pool3d(
        image.unsqueeze(0).unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1
    ).squeeze(0).squeeze(0)

    # Find locations of local maxima
    local_maxima = (image == max_pooled)
    all_peaks = local_maxima & (image > threshold_abs)

    if exclude_border:
        b = min_distance
        all_peaks[:b, :, :] = False;
        all_peaks[-b:, :, :] = False
        all_peaks[:, :b, :] = False;
        all_peaks[:, -b:, :] = False
        all_peaks[:, :, :b] = False;
        all_peaks[:, :, -b:] = False

    # Get coordinates of all peaks
    candidates = all_peaks.nonzero(as_tuple=False)
    if candidates.shape[0] == 0:
        return torch.empty((0, 3), device=device, dtype=torch.long)

    # Sort candidates by intensity
    candidate_values = image[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    sorted_indices = torch.argsort(candidate_values, descending=True)
    candidates = candidates[sorted_indices]

    # Create a suppression grid
    suppress_grid = torch.zeros_like(image, dtype=torch.bool)
    final_peaks = []

    # Pre-calculate a spherical suppression mask
    dist = min_distance
    z, y, x = torch.meshgrid(
        torch.arange(-dist, dist + 1, device=device),
        torch.arange(-dist, dist + 1, device=device),
        torch.arange(-dist, dist + 1, device=device),
        indexing='ij'
    )
    suppress_mask = (z ** 2 + y ** 2 + x ** 2) <= dist ** 2

    # Iterate through sorted candidates
    for i in range(candidates.shape[0]):
        coord = candidates[i]
        # Check if this peak is already suppressed
        if suppress_grid[coord[0], coord[1], coord[2]]:
            continue

        # This peak is valid, add it to the list
        final_peaks.append(coord)

        # Suppress neighbors
        z_min, z_max = max(0, coord[0] - dist), min(image.shape[0], coord[0] + dist + 1)
        y_min, y_max = max(0, coord[1] - dist), min(image.shape[1], coord[1] + dist + 1)
        x_min, x_max = max(0, coord[2] - dist), min(image.shape[2], coord[2] + dist + 1)

        # Calculate slices for the pre-computed mask to align with the image slice
        mask_z_min, mask_z_max = dist - (coord[0] - z_min), dist + (z_max - coord[0] - 1)
        mask_y_min, mask_y_max = dist - (coord[1] - y_min), dist + (y_max - coord[1] - 1)
        mask_x_min, mask_x_max = dist - (coord[2] - x_min), dist + (x_max - coord[2] - 1)

        # Apply the suppression mask to the grid
        suppress_grid[z_min:z_max, y_min:y_max, x_min:x_max] |= suppress_mask[
                                                                mask_z_min:mask_z_max + 1, mask_y_min:mask_y_max + 1,
                                                                mask_x_min:mask_x_max + 1
                                                                ]

    if not final_peaks:
        return torch.empty((0, 3), device=device, dtype=torch.long)
    return torch.stack(final_peaks)


def _refine_peaks_subpixel_gpu(pred_mask_gpu: torch.Tensor, peak_coords_gpu: torch.Tensor, patch_size: int = 5) -> \
Tuple[torch.Tensor, torch.Tensor]:
    """Refines integer peak coordinates to sub-pixel accuracy on the GPU by fitting a 3D quadratic function."""
    if peak_coords_gpu.shape[0] == 0:
        return torch.empty_like(peak_coords_gpu, dtype=torch.float32), torch.empty(0, device=pred_mask_gpu.device,
                                                                                   dtype=torch.float32)

    device = pred_mask_gpu.device
    num_peaks = peak_coords_gpu.shape[0]
    patch_radius = patch_size // 2

    # Create coordinate grid for the patch
    z, y, x = torch.meshgrid(
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Design matrix A for the least-squares problem log(I) = a*z^2 + b*y^2 + ... + j
    A = torch.stack([
        z.ravel() ** 2, y.ravel() ** 2, x.ravel() ** 2,
        z.ravel() * y.ravel(), z.ravel() * x.ravel(), y.ravel() * x.ravel(),
        z.ravel(), y.ravel(), x.ravel(),
        torch.ones(x.numel(), device=device)
    ]).T.expand(num_peaks, -1, -1)

    # Extract patches around each peak
    patch_coords_offsets = torch.stack(torch.meshgrid(
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.long),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.long),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.long),
        indexing='ij'
    ))
    # Shape: (num_peaks, 3, patch_size, patch_size, patch_size)
    patches_coords = peak_coords_gpu[:, :, None, None, None] + patch_coords_offsets

    # Clamp coordinates to be within the volume bounds
    shape_tensor = torch.tensor(pred_mask_gpu.shape, device=device, dtype=torch.long)
    patches_coords[:, 0, ...] = torch.clamp(patches_coords[:, 0, ...], 0, shape_tensor[0] - 1)
    patches_coords[:, 1, ...] = torch.clamp(patches_coords[:, 1, ...], 0, shape_tensor[1] - 1)
    patches_coords[:, 2, ...] = torch.clamp(patches_coords[:, 2, ...], 0, shape_tensor[2] - 1)

    z_indices, y_indices, x_indices = patches_coords[:, 0], patches_coords[:, 1], patches_coords[:, 2]
    patches = pred_mask_gpu[z_indices, y_indices, x_indices].view(num_peaks, -1)

    # Target vector b (log of intensities)
    b = torch.log(torch.clamp(patches, min=1e-6)).unsqueeze(-1)

    # Solve Ax = b for x (the quadratic coefficients)
    try:
        solution = torch.linalg.lstsq(A, b, driver='gels').solution.squeeze(-1)
    except torch.linalg.LinAlgError:
        return peak_coords_gpu.float(), pred_mask_gpu[
            peak_coords_gpu[:, 0], peak_coords_gpu[:, 1], peak_coords_gpu[:, 2]]

    coeffs = solution
    c_zz, c_yy, c_xx, c_zy, c_zx, c_yx, c_z, c_y, c_x, _ = coeffs.T

    # Calculate sub-pixel offset: offset = -H^-1 * grad
    # Hessian matrix H
    H = torch.stack([
        torch.stack([2 * c_zz, c_zy, c_zx]),
        torch.stack([c_zy, 2 * c_yy, c_yx]),
        torch.stack([c_zx, c_yx, 2 * c_xx])
    ]).permute(2, 0, 1)
    # Gradient vector grad
    grad = -torch.stack([c_z, c_y, c_x]).T

    try:
        H_inv = torch.linalg.inv(H)
        offset = torch.bmm(H_inv, grad.unsqueeze(-1)).squeeze(-1)
    except torch.linalg.LinAlgError:
        offset = torch.zeros_like(grad)

    # Filter out large offsets (fit is likely poor)
    valid_mask = torch.max(torch.abs(offset), dim=1).values <= 1.0
    refined_coords = peak_coords_gpu.float()
    refined_coords[valid_mask] += offset[valid_mask]

    # Recalculate scores at sub-pixel locations
    refined_scores = pred_mask_gpu[peak_coords_gpu[:, 0], peak_coords_gpu[:, 1], peak_coords_gpu[:, 2]]
    if valid_mask.any():
        valid_offsets = offset[valid_mask]
        off_z, off_y, off_x = valid_offsets.T

        A_offset = torch.stack([
            off_z ** 2, off_y ** 2, off_x ** 2, off_z * off_y, off_z * off_x, off_y * off_x,
            off_z, off_y, off_x, torch.ones_like(off_z)
        ]).T

        log_scores_valid = torch.sum(A_offset * coeffs[valid_mask], dim=1)
        refined_scores_valid = torch.exp(log_scores_valid)

        # Ensure refined scores are not NaN or Inf before assigning
        score_is_finite = torch.isfinite(refined_scores_valid)
        final_valid_mask = valid_mask.clone()
        final_valid_mask[valid_mask] = score_is_finite  # Update the mask

        refined_scores[final_valid_mask] = refined_scores_valid[score_is_finite]

    return refined_coords, refined_scores


def process_extracted_coordinates(output_dir: str, tomo_names: List[str],
                                  predicted_centroids_with_scores: List[List[float]], voxel_sizes: List[float],
                                  output_format: OutputFormat, output_filename: str):
    """Processes and saves the extracted coordinates to a file."""
    if not predicted_centroids_with_scores:
        logger.warning("No coordinates were extracted, skipping file writing.")
        return

    first_voxel_size = voxel_sizes[0]
    if not all(abs(vs - first_voxel_size) < 1e-4 for vs in voxel_sizes):
        logger.warning(
            f"Detected multiple voxel sizes. Using the first ({first_voxel_size:.2f} Å/px) for metadata in combined file formats.")

    if output_format == OutputFormat.relion_31:
        write_relion_star_file(output_dir, tomo_names, predicted_centroids_with_scores, first_voxel_size,
                               output_filename, version=3.1)
    elif output_format == OutputFormat.relion_50:
        write_relion_star_file(output_dir, tomo_names, predicted_centroids_with_scores, first_voxel_size,
                               output_filename, version=5.0)
    elif output_format == OutputFormat.warp:
        write_warp_star_file(output_dir, tomo_names, predicted_centroids_with_scores, output_filename)
    elif output_format == OutputFormat.imod:
        logger.info("IMOD format selected. Creating one .mod file per tomogram.")
        df = pd.DataFrame({'tomo_name': tomo_names, 'coords': predicted_centroids_with_scores})
        for name, group in df.groupby('tomo_name'):
            tomo_coords = group['coords'].tolist()
            mod_filename = f"{Path(name).stem}.mod"
            write_imod_model(output_dir=output_dir, tomo_name=name, predicted_centroids_with_scores=tomo_coords,
                             output_filename=mod_filename)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _get_prediction_mask_with_tta(
        model,
        padded_vol: torch.Tensor,
        infer_config: InferConfig,
        device: torch.device
) -> torch.Tensor:
    """
    Performs inference with Test-Time Augmentation (TTA).

    Args:
        model: The trained model.
        padded_vol: The preprocessed and padded input tomogram tensor.
        infer_config: The inference configuration object.
        device: The device to run inference on.

    Returns:
        A single, averaged prediction mask tensor.
    """
    deaugmented_masks = []

    if infer_config.use_tta:
        # Define geometric transforms for TTA
        tta_transforms = [
            tio.Compose([]),  # Identity
            tio.Flip(axes='D'),
            tio.Flip(axes='H'),
            tio.Flip(axes='W'),
        ]
        logger.info(f"Using TTA with {len(tta_transforms)} transforms.")
    else:
        tta_transforms = [tio.Compose([])]  # Just the identity transform

    for transform in tta_transforms:
        # Augment the volume. TorchIO transforms expect a 4D tensor (C, D, H, W).
        augmented_vol_4d = transform(padded_vol.unsqueeze(0))

        subject = tio.Subject(volume=tio.ScalarImage(tensor=augmented_vol_4d))
        grid_sampler = tio.GridSampler(
            subject,
            patch_size=model.patch_size,
            patch_overlap=model.patch_size // infer_config.PATCH_OVERLAP_FACTOR
        )
        patch_loader = DataLoader(grid_sampler, batch_size=infer_config.predictions_batch_size, num_workers=0)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")

        with torch.inference_mode():
            for batch in patch_loader:
                input_tensor = batch['volume'][tio.DATA].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=infer_config.use_cuda):
                    predictions = model.predict_step(input_tensor, 0)
                aggregator.add_batch(predictions, batch[tio.LOCATION])

        # Get the aggregated mask for the *augmented* volume
        augmented_mask = aggregator.get_output_tensor()

        # Apply the inverse transform to bring the mask back to the original orientation
        deaugmented_mask = transform.inverse(augmented_mask)
        deaugmented_masks.append(deaugmented_mask)

    # Average the predictions from all augmentations and remove channel dim
    final_mask = torch.stack(deaugmented_masks).mean(dim=0).squeeze(0)
    return final_mask


def _predict_single_tomogram(
        model,
        tomo_fname: Path,
        infer_config: InferConfig,
        target_px_from_model: int,
        device: torch.device
) -> Optional[Tuple[str, List[List[float]], float]]:
    """
    Runs the full inference pipeline for a single tomogram.

    Args:
        model: The trained model.
        tomo_fname: Path to the tomogram MRC file.
        infer_config: The inference configuration object.
        target_px_from_model: The target particle pixel size from the model.
        device: The device to run processing on.

    Returns:
        A tuple containing (tomo_stem, list_of_coordinates, voxel_size), or None if no coords are found.
    """
    original_shape, original_voxel_size = get_mrc_metadata(str(tomo_fname))

    # 1. Preprocess the tomogram (resize, pad, etc.)
    processed_vol, _, padding_values = preprocess_tomogram(
        mrc_path=str(tomo_fname),
        particle_diameter_angst=infer_config.length,
        target_particle_px=target_px_from_model,
        chunk_size=model.patch_size,
        device=device,
        invert_contrast=infer_config.invert_contrast
    )
    padded_vol, _ = symmetrize_and_pad(volume=processed_vol, min_size=model.patch_size)

    # 2. Get the prediction mask using the core inference logic (with TTA)
    aggregated_mask = _get_prediction_mask_with_tta(model, padded_vol, infer_config, device)

    # 3. Post-process the mask (unpad, resize back to original shape)
    unpadded_mask = unpad(aggregated_mask, padding_values)
    pred_mask_gpu = resize_volume(unpadded_mask, original_shape)

    # 4. Apply optional user-provided mask
    if infer_config.masks_dir:
        mask_path = Path(infer_config.masks_dir) / tomo_fname.name
        if mask_path.exists():
            user_mask_np = load_mrc(str(mask_path), normalize=False)
            user_mask_gpu = torch.from_numpy(user_mask_np).to(device)
            if user_mask_gpu.shape != pred_mask_gpu.shape:
                user_mask_gpu = resize_volume(user_mask_gpu, pred_mask_gpu.shape)
            pred_mask_gpu *= user_mask_gpu
        else:
            logger.warning(f"Mask file not found for {tomo_fname.name} at {mask_path}, proceeding without it.")

    # 5. Save the final confidence map if requested
    if infer_config.save_prediction_confidence_map:
        output_mask_path = Path(infer_config.predictions_dir) / f"{tomo_fname.stem}.mrc"
        write_segmentation_mask(pred_mask_gpu, str(output_mask_path), angpix=original_voxel_size, overwrite=True)

    # 6. Extract coordinates if requested
    if infer_config.save_predicted_coords:
        min_dist_px = (infer_config.distance_threshold or infer_config.length) / original_voxel_size
        int_coords_gpu = peak_local_max_torch(
            image=pred_mask_gpu,
            min_distance=int(min_dist_px),
            threshold_abs=infer_config.confidence_threshold,
            exclude_border=True
        )
        if int_coords_gpu.shape[0] > 0:
            refined_coords_gpu, refined_scores_gpu = _refine_peaks_subpixel_gpu(pred_mask_gpu, int_coords_gpu)
            centroids_with_scores = torch.hstack(
                [refined_coords_gpu, refined_scores_gpu.unsqueeze(-1)]).cpu().numpy().tolist()
            return tomo_fname.stem, centroids_with_scores, original_voxel_size

    return None


def infer_tomos(tomo_fnames: List[Path], gpu_id: Optional[int], model_fname: str, infer_config: InferConfig) -> Tuple[
    List[str], List[List[float]], List[float]]:
    """
    Worker function for Dask, performing inference on a batch of tomograms.
    """
    warnings.filterwarnings("ignore", message=".*`img_size` has been deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Using TorchIO images without a torchio.SubjectsLoader.*",
                            category=UserWarning)

    from tomocpt.networks.pickingModel import BasePickingModel

    device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None and torch.cuda.is_available() else torch.device("cpu")
    if gpu_id is not None: torch.cuda.set_device(device)

    try:
        model = BasePickingModel.load_from_checkpoint(model_fname, map_location=device).eval()
        logger.info(f"PickingModel using {type(model.model).__name__} loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load model on {device}: {e}")
        return [], [], []

    try:
        target_px_from_model = model.hparams.config.prepData.desired_particle_pixel_size
    except AttributeError:
        logger.warning(
            "Could not find 'desired_particle_pixel_size' in model checkpoint. Falling back to global config value.")
        target_px_from_model = mainConfig.prepData.desired_particle_pixel_size

    results_aggregator = []
    for tomo_fname in tqdm(tomo_fnames, desc=f"Inferring on {device}", position=gpu_id or 0, leave=False):
        try:
            # The main loop now calls the dedicated function for single tomogram processing.
            result = _predict_single_tomogram(model, tomo_fname, infer_config, target_px_from_model, device)
            if result:
                results_aggregator.append(result)
        except Exception as e:
            logger.error(f"Error processing {tomo_fname.name} on {device}: {e}", exc_info=True)
        finally:
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()

    all_names, all_centroids, all_voxel_sizes = [], [], []
    for name, centroids, vx in results_aggregator:
        # Each 'name' is associated with a list of its centroids
        for centroid in centroids:
            all_names.append(name)
            all_centroids.append(centroid)
            all_voxel_sizes.append(vx)
    return all_names, all_centroids, all_voxel_sizes
