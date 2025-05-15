import gc
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

from tomocpt.logger import get_logger
from tomocpt.dataManager.dataUtils import (
    load_mrc,
    write_segmentation_mask,
    resize_volume,
    plot_example,
)
from tomocpt.dataPreparation.helpers import _preprocess_data_mrc
from tomocpt.defaultConfigs.infer_config import OutputFormat
from tomocpt.mainConfig import mainConfig

infer_config = mainConfig.infer


logger = get_logger()


def write_relion_star_file(
    output_dir: str,
    tomo_names: List[str],
    predicted_centroids_with_scores: List[List[float]],
    voxel_size: Union[float, List[float]],
    output_filename: str = "tomocpt_coords.star",
    version: Union[float, int] = 3.1,
) -> Path:
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
        raise ValueError(
            f"Unsupported Relion version: {version}. Supported versions are 3.1 and 5.0"
        )

    voxel_size = float(
        voxel_size[0] if isinstance(voxel_size, (list, np.ndarray)) else voxel_size
    )

    # Create optics table
    df_optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1],
            "rlnOpticsGroupName": ["OpticsGroup1"],
            "rlnSphericalAberration": [2.7],
            "rlnVoltage": [300],
            "rlnImagePixelSize": [voxel_size],
            "rlnImageDimensionality": [3],
        }
    )

    # Initialize particles data dictionary
    all_tomo_centroids_and_scores = {
        tomo_label: [],
        "rlnCoordinateX": [],
        "rlnCoordinateY": [],
        "rlnCoordinateZ": [],
        "rlnAutopickFigureOfMerit": [],
        "rlnClassNumber": [],
    }

    # Process each coordinate
    for tomoName, predicted_centroid in zip(
        tomo_names, predicted_centroids_with_scores
    ):
        # Convert filename format
        if re.search(r"_\d+\.\d+Apx", tomoName):
            tomoName = re.sub(r"_\d+\.\d+Apx", "", tomoName)
        # Add data to the dictionary
        all_tomo_centroids_and_scores[tomo_label].append(tomoName)
        all_tomo_centroids_and_scores["rlnCoordinateX"].append(predicted_centroid[2])
        all_tomo_centroids_and_scores["rlnCoordinateY"].append(predicted_centroid[1])
        all_tomo_centroids_and_scores["rlnCoordinateZ"].append(predicted_centroid[0])
        all_tomo_centroids_and_scores["rlnAutopickFigureOfMerit"].append(
            predicted_centroid[3]
        )
        all_tomo_centroids_and_scores["rlnClassNumber"].append(1)

    # Create particles table
    df_particles = pd.DataFrame(data=all_tomo_centroids_and_scores)

    # Prepare star file data
    star_data = {"optics": df_optics, "particles": df_particles}

    # Write the star file
    output_path = Path(output_dir) / output_filename
    starfile.write(star_data, output_path, float_format="%0.2f", overwrite=True)
    logger.info(f"Predicted coordinates are stored here: {output_path.resolve()}")

    return output_path


def write_warp_star_file(
    output_dir: str,
    tomo_names: List[str],
    predicted_centroids_with_scores: List[List[float]],
    output_filename: str | None = None,
) -> Path:
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
        output_filename = mainConfig.predict.outCoordFname

    all_tomo_centroids_and_scores = {
        "rlnMicrographName": [],
        "rlnCoordinateX": [],
        "rlnCoordinateY": [],
        "rlnCoordinateZ": [],
        "rlnAutopickFigureOfMerit": [],
    }

    # Process each coordinate
    for tomoName, predicted_centroid in zip(
        tomo_names, predicted_centroids_with_scores
    ):
        # Convert filename format
        if re.search(r"_\d+\.\d+Apx", tomoName):
            tomoName = re.sub(r"_\d+\.\d+Apx", ".tomostar", tomoName)
        else:
            tomoName = tomoName + ".tomostar"

        # Add data to the dictionary
        all_tomo_centroids_and_scores["rlnMicrographName"].append(tomoName)
        all_tomo_centroids_and_scores["rlnCoordinateX"].append(predicted_centroid[2])
        all_tomo_centroids_and_scores["rlnCoordinateY"].append(predicted_centroid[1])
        all_tomo_centroids_and_scores["rlnCoordinateZ"].append(predicted_centroid[0])
        all_tomo_centroids_and_scores["rlnAutopickFigureOfMerit"].append(
            predicted_centroid[3]
        )

    # Create particles table
    df_particles = pd.DataFrame(data=all_tomo_centroids_and_scores)

    # Prepare star file data
    star_data = {"particles": df_particles}

    # Write the star file
    output_path = Path(output_dir) / output_filename
    starfile.write(star_data, output_path, float_format="%0.2f", overwrite=True)
    logger.info(f"Predicted coordinates are stored here: {output_path.resolve()}")

    return output_path


def write_imod_file(
    output_dir: str,
    tomo_names: List[str],
    predicted_centroids_with_scores: List[List[float]],
    output_filename: str | None = None,
) -> Path:
    for tomoName in tomo_names:
        if re.search(r"_\d+\.\d+Apx", tomoName):
            tomoName = re.sub(r"_\d+\.\d+Apx", ".tomostar", tomoName)

        output_path = Path(f"{output_dir}/{tomoName}.txt")
        np.savetxt(
            output_path, predicted_centroids_with_scores, fmt="%.1f", delimiter="\t"
        )
        raise NotImplementedError("This is still work in progress!")


def write_sg_motive_list(
    output_dir: str,
    tomo_names: List[str],
    predicted_centroids_with_scores: List[List[float]],
    output_filename: str = "tomocpt_coords.star",
) -> Path:
    raise NotImplementedError()


def apply_mask(tomo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to a tomogram.

    :param tomo: The tomogram as a numpy array
    :param mask: The mask as a numpy array
    :return: The masked tomogram
    """
    if tomo.shape != mask.shape:
        raise ValueError(
            f"Tomogram shape {tomo.shape} does not match mask shape {mask.shape}"
        )
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

    inp_tensor = inp_tensor[
        _padding_values[0] : _padding_values[1],
        _padding_values[2] : _padding_values[3],
        _padding_values[4] : _padding_values[5],
    ]
    return inp_tensor


def extract_cube_at_predicted_centroid(
    predicted_segmentation_mask: np.array, centroid: np.array
):
    predicted_segmentation_mask = np.asarray(predicted_segmentation_mask)
    x_min, x_max = centroid[0] - 1, centroid[0] + 1
    y_min, y_max = centroid[1] - 1, centroid[1] + 1
    z_min, z_max = centroid[2] - 1, centroid[2] + 1
    subvol = predicted_segmentation_mask[
        x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1
    ]
    return np.amax(subvol)


def extract_centroids_from_pred(
    input_tensor: np.array,
    nn_ang_distance: Optional[float],
    particle_ang_length: float,
    threshold: float,
    angpix: float,
):
    input_tensor = np.asarray(input_tensor, dtype=np.float32)
    if nn_ang_distance is None:
        min_dist = particle_ang_length / angpix
    else:
        min_dist = nn_ang_distance / angpix
    kernel = cube(3)
    centroid_peaks = peak_local_max(
        input_tensor,
        threshold_abs=threshold,
        footprint=kernel,
        min_distance=int(min_dist),
        exclude_border=True,
    )
    return centroid_peaks


def process_extracted_coordinates(
    results, output_dir: str, output_format: OutputFormat, output_filename: str
):
    """Process and save coordinates from parallel inference results."""
    tomoNames = []
    predicted_centroids_with_scores = []
    voxel_sizes = []

    # Unpack the results from parallel processing
    for res in results:
        if not all(isinstance(x, list) for x in res[:2]):  # Basic validation
            logger.warning(f"Skipping invalid result: {res}")
            continue
        tomoNames.extend(res[0])
        predicted_centroids_with_scores.extend(res[1])
        voxel_sizes.extend([res[2]] * len(res[0]))

    if not tomoNames:
        logger.warning("No valid coordinates were extracted.")
        return

    # Write coordinates based on format
    if output_format == OutputFormat.relion31:
        write_relion_star_file(
            output_dir=output_dir,
            tomo_names=tomoNames,
            predicted_centroids_with_scores=predicted_centroids_with_scores,
            voxel_size=voxel_sizes[0],  # Using first voxel size as reference
            output_filename=output_filename,
            version=3.1,
        )
    elif output_format == OutputFormat.relion50:
        write_relion_star_file(
            output_dir=output_dir,
            tomo_names=tomoNames,
            predicted_centroids_with_scores=predicted_centroids_with_scores,
            voxel_size=voxel_sizes[0],  # Using first voxel size as reference
            output_filename=output_filename,
            version=5.0,
        )

    elif output_format == OutputFormat.warp:
        write_warp_star_file(
            output_dir=output_dir,
            tomo_names=tomoNames,
            predicted_centroids_with_scores=predicted_centroids_with_scores,
            output_filename=output_filename,
        )
    else:
        raise NotImplementedError(f"Output format {output_format} not supported")


def _infer_one_tomo(
    tomoFname: str,
    output_fname: str,
    particle_ang_length: float,
    model: torch.nn.Module,
    gpu_id: int,
    patch_size: int,
    batch_size: int,
    plot: bool,
    save_pred_mask: bool,
    extract_coords: bool,
    nearest_neigs_angs: Optional[float],
    threshold: float,
    maskFname: Optional[str],
):
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

    particle_radius_angst = particle_ang_length / 2
    voxel_size = np.nan
    desired_particle_size_in_pixels = (
        model.hparams.config.prepData.desired_particle_pixel_size
    )
    if not Path(output_fname).exists():
        # Preprocess the tomogram data
        (vol, new_shape, old_shape, voxel_size, padding_values, scalar) = (
            _preprocess_data_mrc(
                tomoFname,
                normalization_function="robust_normalization",
                new_particle_size=desired_particle_size_in_pixels,
                particle_radius_angst=particle_radius_angst,
                chunk_size=patch_size,
                use_gpu=infer_config.USE_CUDA_FOR_DATA if gpu_id is not None else False,
            )
        )

        # Add batch dimension for inference
        vol = vol.unsqueeze(0)
        subject = tio.Subject({"input_data": tio.ScalarImage(tensor=vol)})

        grid_sampler = tio.GridSampler(
            subject,
            patch_size=patch_size,
            patch_overlap=patch_size // mainConfig.infer.PATCH_OVERLAP_FACTOR,
            padding_mode="reflect",
        )
        patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
        del vol
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")

        # Perform inference
        with torch.inference_mode():
            for idx, patches_batch in enumerate(
                tqdm(patch_loader, position=gpu_id, desc=Path(output_fname).stem)
            ):
                input_tensor = patches_batch["input_data"][tio.DATA]
                if gpu_id is not None:
                    input_tensor = input_tensor.cuda(device=gpu_id)
                locations = patches_batch[tio.LOCATION]
                outputs = model.predict_step(input_tensor, idx)
                aggregator.add_batch(outputs, locations)

        # Get full volume prediction and restore original size
        output_tensor = aggregator.get_output_tensor()
        output_tensor = output_tensor.squeeze(0)
        output_tensor = unpad(output_tensor, padding_values)
        output_tensor = output_tensor.cpu().numpy()
        output_tensor, _ = resize_volume(
            output_tensor,
            new_shape=old_shape,
            chunk_size=patch_size,
            use_gpu=gpu_id is not None and gpu_id >= 0,
            mean_as_padding_value=False,
        )

        # Apply mask if provided - mask must match the restored volume dimensions
        if maskFname and Path(maskFname).exists():
            logger.info("Loading mask")
            try:
                mask = load_mrc(maskFname, normalize=None, return_boxSize=False)
                if mask.shape != output_tensor.shape:
                    raise ValueError(
                        f"Mask shape {mask.shape} does not match restored volume shape {output_tensor.shape}. "
                        "The mask must have the same dimensions as the original volume."
                    )
                output_tensor = output_tensor * mask
                del mask
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing mask: {str(e)}. Aborting.")
                raise

        if save_pred_mask:
            write_segmentation_mask(
                output_tensor, output_fname, angpix=voxel_size, overwrite=True
            )
        if plot:
            plot_example(subject["input_data"][tio.DATA], output_tensor)

    elif extract_coords:
        output_tensor, voxel_size = load_mrc(
            output_fname, normalize=None, return_boxSize=True
        )
        logger.info(f"Found previous prediction results for {output_fname}")
    else:
        return [], [], voxel_size

    if extract_coords:
        coords_array = extract_centroids_from_pred(
            output_tensor,
            nn_ang_distance=nearest_neigs_angs,
            particle_ang_length=particle_ang_length,
            threshold=threshold,
            angpix=voxel_size,
        )
        tomoFnameList = []
        centroids_and_scores = []
        if len(coords_array) == 0:
            logger.warning(
                f"There are no particles to be found in {output_fname.split('/')[-1]}"
            )
            pass
        else:
            centroids_and_scores = np.zeros((coords_array.shape[0], 4))
            for i, centroid_peak in enumerate(coords_array):
                tomoFnameList.append(Path(tomoFname).stem)
                centroids_and_scores[i, 0] = centroid_peak[0]
                centroids_and_scores[i, 1] = centroid_peak[1]
                centroids_and_scores[i, 2] = centroid_peak[2]
                centroids_and_scores[i, 3] = extract_cube_at_predicted_centroid(
                    output_tensor, centroid_peak
                )
            centroids_and_scores = centroids_and_scores.tolist()
        return tomoFnameList, centroids_and_scores, voxel_size
    return [], [], voxel_size


MODEL = None
def infer_tomos(
    tomoFnames: List[Path],
    predsDir: str,
    modelFname: str,
    gpu_id: int,
    particleLengthAng: float,
    batch_size: int,
    plot: bool,
    save_pred_mask: bool,
    extract_coords: bool,
    nearest_neigs_angs: Optional[float],
    threshold: float,
    masksDir: Optional[str],
):
    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.mainConfig import mainConfig

    global MODEL
    if MODEL is None:
        if gpu_id is not None:
            kwargs = {"map_location": f"cuda:{gpu_id}"}
        else:
            kwargs = {"map_location": "cpu"}
            
        # Load the checkpoint
        checkpoint = torch.load(modelFname, **kwargs)
        has_teacher = any(k.startswith('teacher') for k in checkpoint['state_dict'].keys())
        
        if has_teacher:
            # This is a distillation model
            logger.info("Detected distillation model checkpoint")
            
            # Create a base model with the correct hyperparameters
            # Ensure we pass the same hyperparameters that would be used in training
            MODEL = BasePickingModel(config=mainConfig)
            
            # Extract only the student model weights
            # The student model has keys that start with 'model.'
            student_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    # Remove the 'model.' prefix for the base model
                    student_state_dict[k[6:]] = v  # Skip 'model.'
            
            # Load filtered state dict into model
            missing_keys, unexpected_keys = MODEL.model.load_state_dict(student_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys when loading student model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading student model: {unexpected_keys}")
                
            logger.info("Successfully loaded student model from distillation checkpoint")
        else:
            # Regular picking model
            MODEL = BasePickingModel.load_from_checkpoint(modelFname, **kwargs)
            logger.info("Loaded standard picking model")

    model = MODEL.eval()
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    patch_size = model.patch_size

    # Rest of the function remains unchanged
    tomo_names, one_tomo_centroids_and_scores, voxel_sizes = [], [], []
    for data_fname in tomoFnames:
        output_fname = str(Path(predsDir) / data_fname.name)
        mask_fname = Path(masksDir) / data_fname.name if masksDir else None
        _tomo_names, _one_tomo_centroids_and_scores, vx = _infer_one_tomo(
            tomoFname=str(data_fname),
            output_fname=output_fname,
            particle_ang_length=particleLengthAng,
            model=model,
            gpu_id=gpu_id,
            patch_size=patch_size,
            batch_size=batch_size,
            plot=plot,
            save_pred_mask=save_pred_mask,
            extract_coords=extract_coords,
            nearest_neigs_angs=nearest_neigs_angs,
            threshold=threshold,
            maskFname=str(mask_fname) if mask_fname else None,
        )
        tomo_names.extend(_tomo_names)
        one_tomo_centroids_and_scores.extend(_one_tomo_centroids_and_scores)
        voxel_sizes.append(vx)
    return tomo_names, one_tomo_centroids_and_scores, voxel_sizes
