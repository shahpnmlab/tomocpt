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
    #logger.info(f"Writing segmentation mask to {output_path}")
    mrcfile.write(output_path, data=m.astype(np.float16), voxel_size=angpix, overwrite=overwrite)


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
    max_pooled = torch.nn.functional.max_pool3d(
        image.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0).squeeze(0)
    local_maxima = (image == max_pooled)
    all_peaks = local_maxima & (image > threshold_abs)

    if exclude_border:
        b = min_distance
        all_peaks[:b, :, :] = False; all_peaks[-b:, :, :] = False
        all_peaks[:, :b, :] = False; all_peaks[:, -b:, :] = False
        all_peaks[:, :, :b] = False; all_peaks[:, :, -b:] = False

    candidates = all_peaks.nonzero(as_tuple=False)
    if candidates.shape[0] == 0:
        return torch.empty((0, 3), device=device, dtype=torch.long)

    candidate_values = image[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    sorted_indices = torch.argsort(candidate_values, descending=True)
    candidates = candidates[sorted_indices]

    suppress_grid = torch.zeros_like(image, dtype=torch.bool)
    final_peaks = []
    dist = min_distance
    z, y, x = torch.meshgrid(
        torch.arange(-dist, dist + 1, device=device),
        torch.arange(-dist, dist + 1, device=device),
        torch.arange(-dist, dist + 1, device=device),
        indexing='ij'
    )
    suppress_mask = (z ** 2 + y ** 2 + x ** 2) <= dist ** 2

    for i in range(candidates.shape[0]):
        coord = candidates[i]
        if suppress_grid[coord[0], coord[1], coord[2]]:
            continue
        final_peaks.append(coord)
        z_min, z_max = max(0, coord[0] - dist), min(image.shape[0], coord[0] + dist + 1)
        y_min, y_max = max(0, coord[1] - dist), min(image.shape[1], coord[1] + dist + 1)
        x_min, x_max = max(0, coord[2] - dist), min(image.shape[2], coord[2] + dist + 1)
        mask_z_min, mask_z_max = dist - (coord[0] - z_min), dist + (z_max - coord[0] - 1)
        mask_y_min, mask_y_max = dist - (coord[1] - y_min), dist + (y_max - coord[1] - 1)
        mask_x_min, mask_x_max = dist - (coord[2] - x_min), dist + (x_max - coord[2] - 1)
        suppress_grid[z_min:z_max, y_min:y_max, x_min:x_max] |= suppress_mask[
            mask_z_min:mask_z_max + 1, mask_y_min:mask_y_max + 1, mask_x_min:mask_x_max + 1
        ]

    if not final_peaks:
        return torch.empty((0, 3), device=device, dtype=torch.long)
    return torch.stack(final_peaks)

def _prune_refined_peaks_gpu(
    coords: torch.Tensor,
    scores: torch.Tensor,
    min_distance: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prunes refined, sub-pixel coordinates to enforce a minimum distance after refinement.

    Iterates through coordinates sorted by score, keeping the highest-scoring peak
    and removing any other lower-scoring peaks within the specified minimum distance.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 3) with sub-pixel coordinates.
        scores (torch.Tensor): Tensor of shape (N,) with corresponding peak scores.
        min_distance (float): The minimum distance to enforce between peaks in pixels.

    Returns:
        A tuple containing:
        - torch.Tensor: The pruned coordinates.
        - torch.Tensor: The scores of the pruned coordinates.
    """
    if coords.shape[0] < 2:
        return coords, scores

    device = coords.device
    min_distance_sq = min_distance ** 2

    # Sort peaks by score in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_coords = coords[sorted_indices]
    sorted_scores = scores[sorted_indices]

    # Mask to track which peaks to keep
    keep_mask = torch.ones(sorted_coords.shape[0], dtype=torch.bool, device=device)

    for i in range(sorted_coords.shape[0]):
        # If this peak has already been suppressed by a higher-scoring one, skip it
        if not keep_mask[i]:
            continue

        # This is a good peak. Now, suppress any subsequent, lower-scoring peaks that are too close.
        # Calculate squared distances to all subsequent peaks.
        distances_sq = torch.sum((sorted_coords[i+1:] - sorted_coords[i])**2, dim=1)
        
        # Find which of the *subsequent* peaks are too close
        too_close_mask = distances_sq < min_distance_sq
        
        # Update the 'keep_mask' for the subsequent peaks.
        # A peak is marked as False if it's too close to the current 'good' peak.
        if too_close_mask.any():
            keep_mask[i+1:][too_close_mask] = False

    final_coords = sorted_coords[keep_mask]
    final_scores = sorted_scores[keep_mask]

    return final_coords, final_scores


def _refine_peaks_subpixel_gpu(pred_mask_gpu: torch.Tensor, peak_coords_gpu: torch.Tensor, patch_size: int = 3) -> \
Tuple[torch.Tensor, torch.Tensor]:
    """Refines integer peak coordinates to sub-pixel accuracy on the GPU by fitting a 3D quadratic function."""
    if peak_coords_gpu.shape[0] == 0:
        return torch.empty_like(peak_coords_gpu, dtype=torch.float32), torch.empty(0, device=pred_mask_gpu.device, dtype=torch.float32)

    device = pred_mask_gpu.device
    num_peaks = peak_coords_gpu.shape[0]
    patch_radius = patch_size // 2
    z, y, x = torch.meshgrid(
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32),
        indexing='ij'
    )
    A = torch.stack([
        z.ravel() ** 2, y.ravel() ** 2, x.ravel() ** 2,
        z.ravel() * y.ravel(), z.ravel() * x.ravel(), y.ravel() * x.ravel(),
        z.ravel(), y.ravel(), x.ravel(),
        torch.ones(x.numel(), device=device)
    ]).T.expand(num_peaks, -1, -1)
    patch_coords_offsets = torch.stack(torch.meshgrid(
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.long),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.long),
        torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.long),
        indexing='ij'
    ))
    patches_coords = peak_coords_gpu[:, :, None, None, None] + patch_coords_offsets
    shape_tensor = torch.tensor(pred_mask_gpu.shape, device=device, dtype=torch.long)
    patches_coords[:, 0, ...] = torch.clamp(patches_coords[:, 0, ...], 0, shape_tensor[0] - 1)
    patches_coords[:, 1, ...] = torch.clamp(patches_coords[:, 1, ...], 0, shape_tensor[1] - 1)
    patches_coords[:, 2, ...] = torch.clamp(patches_coords[:, 2, ...], 0, shape_tensor[2] - 1)
    z_indices, y_indices, x_indices = patches_coords[:, 0], patches_coords[:, 1], patches_coords[:, 2]
    patches = pred_mask_gpu[z_indices, y_indices, x_indices].view(num_peaks, -1)
    b = torch.log(torch.clamp(patches, min=1e-6)).unsqueeze(-1)

    try:
        solution = torch.linalg.lstsq(A, b, driver='gels').solution.squeeze(-1)
    except torch.linalg.LinAlgError:
        return peak_coords_gpu.float(), pred_mask_gpu[peak_coords_gpu[:, 0], peak_coords_gpu[:, 1], peak_coords_gpu[:, 2]]

    coeffs = solution
    c_zz, c_yy, c_xx, c_zy, c_zx, c_yx, c_z, c_y, c_x, _ = coeffs.T
    H = torch.stack([
        torch.stack([2 * c_zz, c_zy, c_zx]),
        torch.stack([c_zy, 2 * c_yy, c_yx]),
        torch.stack([c_zx, c_yx, 2 * c_xx])
    ]).permute(2, 0, 1)
    grad = -torch.stack([c_z, c_y, c_x]).T

    try:
        H_inv = torch.linalg.inv(H)
        offset = torch.bmm(H_inv, grad.unsqueeze(-1)).squeeze(-1)
    except torch.linalg.LinAlgError:
        offset = torch.zeros_like(grad)

    refined_coords = peak_coords_gpu.float()
    refined_scores = pred_mask_gpu[peak_coords_gpu[:, 0], peak_coords_gpu[:, 1], peak_coords_gpu[:, 2]]

    fit_is_plausible_mask = torch.max(torch.abs(offset), dim=1).values <= 1.0
    if fit_is_plausible_mask.any():
        plausible_offsets = offset[fit_is_plausible_mask]
        off_z, off_y, off_x = plausible_offsets.T
        A_offset = torch.stack([
            off_z ** 2, off_y ** 2, off_x ** 2, off_z * off_y, off_z * off_x, off_y * off_x,
            off_z, off_y, off_x, torch.ones_like(off_z)
        ]).T
        log_scores_plausible = torch.sum(A_offset * coeffs[fit_is_plausible_mask], dim=1)
        refined_scores_plausible = torch.exp(log_scores_plausible)
        score_is_finite_mask = torch.isfinite(refined_scores_plausible)
        final_update_mask = fit_is_plausible_mask.clone()
        final_update_mask[fit_is_plausible_mask] = score_is_finite_mask
        if final_update_mask.any():
            refined_coords[final_update_mask] += offset[final_update_mask]
            refined_scores[final_update_mask] = refined_scores_plausible[score_is_finite_mask]
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
        device: torch.device,
        tomo_name: str,
        gpu_id: Optional[int],
        pbar: tqdm  # Accept the worker's progress bar
) -> torch.Tensor:
    """
    Performs TTA, reconfiguring the passed-in progress bar for each step.
    """
    deaugmented_masks = []
    use_tta = getattr(infer_config, 'use_tta', False)
    tta_transforms = [tio.Compose([])]
    if use_tta:
        flips = [tio.Flip(axes=i) for i in (0, 1, 2)]
        tta_transforms.extend([tio.Compose([t]) for t in flips])
    num_augmentations = len(tta_transforms)

    for i, transform in enumerate(tta_transforms):
        augmented_vol_4d = transform(padded_vol.unsqueeze(0))
        subject = tio.Subject(volume=tio.ScalarImage(tensor=augmented_vol_4d))
        grid_sampler = tio.GridSampler(
            subject,
            patch_size=model.patch_size,
            patch_overlap=model.patch_size // infer_config.PATCH_OVERLAP_FACTOR
        )
        patch_loader = DataLoader(grid_sampler, batch_size=infer_config.predictions_batch_size, num_workers=0)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")
        
        worker_name = f"GPU-{gpu_id}" if gpu_id is not None else "CPU"
        tta_str = f" [TTA {i+1}/{num_augmentations}]" if use_tta else ""
        pbar_desc = f"{worker_name} | {tomo_name}{tta_str}"
        
        pbar.reset(total=len(patch_loader))
        pbar.set_description(pbar_desc)
        
        with torch.inference_mode():
            for batch in patch_loader:
                input_tensor = batch['volume'][tio.DATA].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=infer_config.use_cuda):
                    predictions = model.predict_step(input_tensor, 0)
                aggregator.add_batch(predictions, batch[tio.LOCATION])
                pbar.update(1)
        
        augmented_mask = aggregator.get_output_tensor()
        deaugmented_mask = transform(augmented_mask)
        deaugmented_masks.append(deaugmented_mask)

    final_mask = torch.stack(deaugmented_masks).mean(dim=0).squeeze(0)
    return final_mask


def _predict_single_tomogram(
        model,
        tomo_fname: Path,
        infer_config: InferConfig,
        target_px_from_model: int,
        device: torch.device,
        gpu_id: Optional[int],
        pbar: tqdm  # Accept the worker's progress bar
) -> Optional[Dict[str, Any]]:
    """
    Runs the full inference pipeline for a single tomogram, passing the pbar down.
    """
    original_shape, original_voxel_size = get_mrc_metadata(str(tomo_fname))
    processed_vol, _, padding_values = preprocess_tomogram(
        mrc_path=str(tomo_fname),
        particle_diameter_angst=infer_config.length,
        target_particle_px=target_px_from_model,
        chunk_size=model.patch_size,
        device=device,
        invert_contrast=infer_config.invert_contrast
    )
    padded_vol, _ = symmetrize_and_pad(volume=processed_vol, min_size=model.patch_size)
    aggregated_mask = _get_prediction_mask_with_tta(
        model=model,
        padded_vol=padded_vol,
        infer_config=infer_config,
        device=device,
        tomo_name=tomo_fname.name,
        gpu_id=gpu_id,
        pbar=pbar  # Pass the pbar object
    )
    unpadded_mask = unpad(aggregated_mask, padding_values)
    pred_mask_gpu = resize_volume(unpadded_mask, original_shape)

    if infer_config.masks_dir:
        mask_path = Path(infer_config.masks_dir) / tomo_fname.name
        if mask_path.exists():
            user_mask_np = load_mrc(str(mask_path), normalize=False)
            user_mask_gpu = torch.from_numpy(user_mask_np).to(device)
            if user_mask_gpu.shape != pred_mask_gpu.shape:
                user_mask_gpu = resize_volume(user_mask_gpu, pred_mask_gpu.shape)
            pred_mask_gpu *= user_mask_gpu.to(pred_mask_gpu.device)
        else:
            logger.warning(f"Mask file not found for {tomo_fname.name} at {mask_path}, proceeding without it.")

    if infer_config.save_prediction_confidence_map:
        output_mask_path = Path(infer_config.predictions_dir) / f"{tomo_fname.stem}.mrc"
        write_segmentation_mask(pred_mask_gpu, str(output_mask_path), angpix=original_voxel_size, overwrite=True)

    if infer_config.save_predicted_coords:
        # Calculate min_distance in pixels as a float for precision
        min_dist_px = (infer_config.distance_threshold or infer_config.length) / original_voxel_size
        
        # peak_local_max_torch requires an integer distance; use ceiling for safety
        int_coords_gpu = peak_local_max_torch(
            image=pred_mask_gpu,
            min_distance=int(np.ceil(min_dist_px)),
            threshold_abs=infer_config.confidence_threshold,
            exclude_border=True
        )
        if int_coords_gpu.shape[0] > 0:
            refined_coords_gpu, refined_scores_gpu = _refine_peaks_subpixel_gpu(pred_mask_gpu, int_coords_gpu)

            # Prune the refined coordinates to enforce the minimum distance constraint again
            final_coords_gpu, final_scores_gpu = _prune_refined_peaks_gpu(
                refined_coords_gpu,
                refined_scores_gpu,
                min_distance=min_dist_px  # Use the precise float value for pruning
            )
            
            if final_coords_gpu.shape[0] > 0:
                centroids_with_scores = torch.hstack(
                    [final_coords_gpu, final_scores_gpu.unsqueeze(-1)]).cpu().numpy().tolist()
                return {"name": tomo_fname.stem, "coords": centroids_with_scores, "voxel_size": original_voxel_size}
    return None


def infer_tomos(tomo_fnames: List[Path], gpu_id: Optional[int], model_fname: str, infer_config: InferConfig) -> List[Dict[str, Any]]:
    """
    Worker function for Dask. Creates ONE progress bar and reuses it for all tomograms.
    """
    warnings.filterwarnings("ignore", message=".*`img_size` has been deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Using TorchIO images without a torchio.SubjectsLoader.*", category=UserWarning)

    from tomocpt.networks.pickingModel import BasePickingModel

    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    try:
        model = BasePickingModel.load_from_checkpoint(model_fname, map_location=device).eval()
        logger.info(f"PickingModel using {type(model.model).__name__} loaded on {device} (Worker {gpu_id})")
    except Exception as e:
        logger.error(f"Failed to load model on worker {gpu_id}: {e}")
        return []

    try:
        target_px_from_model = model.hparams.config.prepData.desired_particle_pixel_size
    except AttributeError:
        logger.warning("Could not find 'desired_particle_pixel_size' in model. Falling back to global config.")
        target_px_from_model = mainConfig.prepData.desired_particle_pixel_size

    results_aggregator = []
    # Create one pbar for the entire life of this worker function.
    # leave=True ensures it remains visible after this function returns.
    worker_desc = f"GPU-{gpu_id}" if gpu_id is not None else "CPU"
    with tqdm(total=1, desc=f"{worker_desc} | Waiting...", position=gpu_id or 0, leave=True) as pbar:
        for tomo_fname in tomo_fnames:
            try:
                # Pass the single pbar instance to the processing function.
                result = _predict_single_tomogram(model, tomo_fname, infer_config, target_px_from_model, device, gpu_id, pbar)
                if result:
                    results_aggregator.append(result)
            except Exception as e:
                logger.error(f"Error processing {tomo_fname.name} on worker {gpu_id}: {e}", exc_info=True)
            finally:
                gc.collect()
                if device.type == 'cuda': torch.cuda.empty_cache()
    
    return results_aggregator
