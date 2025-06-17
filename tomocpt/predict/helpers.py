import gc
import logging
import re
from pathlib import Path
from typing import Union, List, Tuple, Optional

import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
import torchio as tio
from torch.utils.data import DataLoader
from tomocpt.mainConfig import mainConfig
from tomocpt.defaultConfigs.infer_config import InferConfig, OutputFormat
from tomocpt.logger import get_logger
from tomocpt.dataManager.preprocessing import (
    preprocess_tomogram,
    resize_volume,
    get_mrc_metadata,
    load_mrc,
)
logger = get_logger()


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
    A PyTorch implementation of scikit-image's peak_local_max.
    Finds peaks in a 3D image as coordinates.
    """
    device = image.device

    candidates = (image > threshold_abs).nonzero(as_tuple=False)
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

    if exclude_border:
        b = min_distance if isinstance(min_distance, int) else int(min_distance)
        suppress_grid[:b, :, :] = True
        suppress_grid[-b:, :, :] = True
        suppress_grid[:, :b, :] = True
        suppress_grid[:, -b:, :] = True
        suppress_grid[:, :, :b] = True
        suppress_grid[:, :, -b:] = True

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
                                                                mask_z_min:mask_z_max + 1, mask_y_min:mask_y_max + 1,
                                                                mask_x_min:mask_x_max + 1
                                                                ]

    if not final_peaks:
        return torch.empty((0, 3), device=device, dtype=torch.long)

    return torch.stack(final_peaks)


def _refine_peaks_subpixel_gpu(
        pred_mask_gpu: torch.Tensor,
        peak_coords_gpu: torch.Tensor,
        patch_size: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refines integer peak coordinates to sub-pixel accuracy on the GPU."""
    if peak_coords_gpu.shape[0] == 0:
        return torch.empty_like(peak_coords_gpu, dtype=torch.float32), torch.empty(0, device=pred_mask_gpu.device,
                                                                                   dtype=torch.float32)

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
        z.ravel() ** 2, y.ravel() ** 2, x.ravel() ** 2, z.ravel() * y.ravel(), z.ravel() * x.ravel(),
        y.ravel() * x.ravel(),
        z.ravel(), y.ravel(), x.ravel(), torch.ones(x.numel(), device=device)
    ]).T.expand(num_peaks, -1, -1)

    patch_coords_offsets = torch.stack(torch.meshgrid(
        torch.arange(-patch_radius, patch_radius + 1, device=device),
        torch.arange(-patch_radius, patch_radius + 1, device=device),
        torch.arange(-patch_radius, patch_radius + 1, device=device),
        indexing='ij'
    ))
    patches_coords = peak_coords_gpu[:, :, None, None, None] + patch_coords_offsets

    shape_tensor = torch.tensor(pred_mask_gpu.shape, device=device, dtype=torch.float32).view(1, 3, 1, 1, 1) - 1
    grid = (patches_coords.permute(0, 2, 3, 4, 1) * 2 / shape_tensor.squeeze() - 1).flip(-1)

    patches = torch.nn.functional.grid_sample(
        pred_mask_gpu.view(1, 1, *pred_mask_gpu.shape), grid, mode='bilinear', align_corners=False
    ).squeeze().view(num_peaks, -1)

    b = torch.log(torch.clamp(patches, min=1e-6)).unsqueeze(-1)

    try:
        solution = torch.linalg.lstsq(A, b, driver='gels').solution.squeeze(-1)
    except torch.linalg.LinAlgError:
        return peak_coords_gpu.float(), pred_mask_gpu[
            peak_coords_gpu[:, 0], peak_coords_gpu[:, 1], peak_coords_gpu[:, 2]]

    coeffs = solution
    c_zz, c_yy, c_xx, c_zy, c_zx, c_yx, c_z, c_y, c_x, _ = coeffs.T

    H = torch.stack([
        torch.stack([2 * c_zz, c_zy, c_zx]), torch.stack([c_zy, 2 * c_yy, c_yx]), torch.stack([c_zx, c_yx, 2 * c_xx])
    ]).permute(2, 0, 1)
    grad = -torch.stack([c_z, c_y, c_x]).T

    try:
        H_inv = torch.linalg.inv(H)
        offset = torch.bmm(H_inv, grad.unsqueeze(-1)).squeeze(-1)
    except torch.linalg.LinAlgError:
        offset = torch.zeros_like(grad)

    valid_mask = torch.max(torch.abs(offset), dim=1).values <= 1.0
    refined_coords = peak_coords_gpu.float()
    refined_coords[valid_mask] += offset[valid_mask]

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

        score_is_finite = torch.isfinite(refined_scores_valid)
        final_valid_mask = valid_mask.clone()
        final_valid_mask[valid_mask] = score_is_finite

        refined_scores[final_valid_mask] = refined_scores_valid[score_is_finite]

    return refined_coords, refined_scores


def infer_tomos(tomo_fnames: List[Path], gpu_id: Optional[int], model_fname: str, infer_config: InferConfig) -> Tuple[
    List[str], List[List[float]], List[float]]:
    """Worker function for Dask, performing inference on a batch of tomograms."""
    from tomocpt.networks.pickingModel import BasePickingModel
    from tqdm import tqdm

    device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None and torch.cuda.is_available() else torch.device("cpu")
    if gpu_id is not None: torch.cuda.set_device(device)

    try:
        model = BasePickingModel.load_from_checkpoint(model_fname, map_location=device).eval()
    except Exception as e:
        logger.error(f"Failed to load model on {device}: {e}")
        return [], [], []

    results = []
    for tomo_fname in tqdm(tomo_fnames, desc=f"Inferring on {device}", position=gpu_id or 0, leave=False):
        try:
            prep_config = mainConfig.prepData
            original_shape, original_voxel_size = get_mrc_metadata(str(tomo_fname))
            processed_vol, _ = preprocess_tomogram(
                mrc_path=str(tomo_fname), particle_diameter_angst=infer_config.length,
                target_particle_px=prep_config.desired_particle_pixel_size,
                chunk_size=model.patch_size, device=device,
            )
            subject = tio.Subject(volume=tio.ScalarImage(tensor=processed_vol.unsqueeze(0)))
            grid_sampler = tio.GridSampler(subject, patch_size=model.patch_size,
                                           patch_overlap=model.patch_size // infer_config.PATCH_OVERLAP_FACTOR)
            patch_loader = DataLoader(grid_sampler, batch_size=infer_config.predictions_batch_size, num_workers=0)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")
            with torch.inference_mode():
                for batch in patch_loader:
                    input_tensor = batch['volume'][tio.DATA].to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=infer_config.use_cuda):
                        predictions = model.predict_step(input_tensor, 0)
                    aggregator.add_batch(predictions, batch[tio.LOCATION])
            aggregated_mask = aggregator.get_output_tensor().squeeze(0)
            pred_mask_gpu = resize_volume(aggregated_mask, original_shape)

            if infer_config.masks_dir and (mask_path := Path(infer_config.masks_dir) / tomo_fname.name).exists():
                user_mask_np = load_mrc(str(mask_path), normalize=False)
                user_mask_gpu = torch.from_numpy(user_mask_np).to(device)
                if user_mask_gpu.shape != pred_mask_gpu.shape:
                    user_mask_gpu = resize_volume(user_mask_gpu, pred_mask_gpu.shape)
                pred_mask_gpu *= user_mask_gpu

            if infer_config.save_prediction_confidence_map:
                output_mask_path = Path(infer_config.predictions_dir) / f"{tomo_fname.stem}_pred_mask.mrc"
                write_segmentation_mask(pred_mask_gpu, str(output_mask_path), angpix=original_voxel_size,
                                        overwrite=True)

            if infer_config.save_predicted_coords:
                min_dist_px = (infer_config.distance_threshold or infer_config.length) / original_voxel_size

                int_coords_gpu = peak_local_max_torch(
                    image=pred_mask_gpu, min_distance=int(min_dist_px),
                    threshold_abs=infer_config.confidence_threshold, exclude_border=True
                )

                if int_coords_gpu.shape[0] > 0:
                    refined_coords_gpu, refined_scores_gpu = _refine_peaks_subpixel_gpu(pred_mask_gpu, int_coords_gpu)
                    centroids_with_scores = torch.hstack([
                        refined_coords_gpu, refined_scores_gpu.unsqueeze(-1)
                    ]).cpu().numpy()
                    results.append((tomo_fname.stem, centroids_with_scores.tolist(), original_voxel_size))
        except Exception as e:
            logger.error(f"Error processing {tomo_fname.name} on {device}: {e}", exc_info=True)
        finally:
            del pred_mask_gpu, aggregated_mask, subject, grid_sampler, patch_loader, aggregator, processed_vol
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()

    all_names, all_centroids, all_voxel_sizes = [], [], []
    for name, centroids, vx in results:
        all_names.extend([name] * len(centroids))
        all_centroids.extend(centroids)
        all_voxel_sizes.extend([vx] * len(centroids))
    return all_names, all_centroids, all_voxel_sizes


def process_extracted_coordinates(output_dir: str,
                                  tomo_names: List[str],
                                  predicted_centroids_with_scores: List[List[float]],
                                  voxel_sizes: List[float],
                                  output_format: OutputFormat,
                                  output_filename: str = "tomocpt_coords.star"):
    """Processes and saves the extracted coordinates to a file."""
    if not voxel_sizes:
        logger.warning("No coordinates were extracted, skipping file writing.")
        return

    first_voxel_size = voxel_sizes[0]
    if not all(abs(vs - first_voxel_size) < 1e-4 for vs in voxel_sizes):
        logger.warning(
            f"Detected multiple voxel sizes. Using the first ({first_voxel_size:.2f} Å/px) for the output file.")

    if output_format == OutputFormat.relion_31:
        write_relion_star_file(
            output_dir, tomo_names, predicted_centroids_with_scores, first_voxel_size, output_filename, version=3.1
        )
    elif output_format == OutputFormat.relion_50:
        write_relion_star_file(
            output_dir, tomo_names, predicted_centroids_with_scores, first_voxel_size, output_filename, version=5.0
        )
    elif output_format == OutputFormat.warp:
        write_warp_star_file(
            output_dir, tomo_names, predicted_centroids_with_scores, output_filename
        )
    else:
        raise ValueError(f"Unsupported output format: {output_format}")