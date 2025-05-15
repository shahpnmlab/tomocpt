import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from tomocpt.logger import get_logger
from tomocpt.defaultConfigs.prepdata_config import PrepareDataType


logging = get_logger()


class ConfigurationError(Exception):
    """Raised when there are issues with the configuration"""

    pass


def validate_config_lists(config: DictConfig) -> None:
    """Validate that all input lists have matching lengths"""
    list_lengths = {
        "particle_length_ang": (
            len(config.particle_length_ang)
            if isinstance(config.particle_length_ang, (list, tuple))
            else 1
        ),
        "raw_data_dir": (
            len(config.raw_data_dir)
            if isinstance(config.raw_data_dir, (list, tuple))
            else 1
        ),
        "coordinate_files": (
            len(config.coordinate_files)
            if isinstance(config.coordinate_files, (list, tuple))
            else 1
        ),
        "class_id": (
            len(config.class_id) if isinstance(config.class_id, (list, tuple)) else 1
        ),
    }

    reference_length = next(iter(list_lengths.values()))
    mismatched = {k: v for k, v in list_lengths.items() if v != reference_length}

    if mismatched:
        raise ConfigurationError(
            f"Configuration lists must have matching lengths. Mismatched lengths: {mismatched}. Expected: {reference_length}"
        )


def prepare_data(config: DictConfig = None) -> None:
    """
    Prepares particle picking datasets by processing multiple tomograms and their corresponding coordinate files.

    Takes raw tomogram data and coordinate files (in either STAR or IMOD format) and processes them into
    a standardized format for model training. Handles multiple datasets in parallel, with support for
    different particle classes and sizes.

    Parameters
    ----------
    config : DictConfig, optional
        Configuration object containing data preparation parameters including:
        - particle_length_ang: List of particle lengths in Angstroms
        - raw_data_dir: List of directories containing raw tomogram data
        - coordinate_files: List of paths to coordinate files
        - class_id: List of class identifiers for each dataset
        - coordinate_file_type: Type of coordinate files (STAR or IMOD)
        - training_data_dir: Output directory for processed data

    Outputs
    -------
    Creates a directory structure containing:
    - Processed coordinate files in standardized format
    - Dataset-specific subdirectories
    - Saved configuration file (prep_config.yaml)

    Directory Structure
    ------------------
    training_data_dir/
    ├── dataset_1_class_{id}/
    ├── dataset_2_class_{id}/
    └── prep_config.yaml

    Notes
    -----
    - Supports both STAR and IMOD coordinate file formats
    - Creates separate directories for each dataset/class combination
    - Handles multiple particle sizes and classes
    - Validates input configuration before processing
    - Continues processing remaining datasets if one fails
    - Logs progress and errors for each dataset

    Examples
    --------
    >>> from omegaconf import DictConfig
    >>> config = DictConfig({
    ...     'particle_length_ang': '100,150',
    ...     'raw_data_dir': '/path/to/data1,/path/to/data2',
    ...     'coordinate_files': '/path/to/coords1.star,/path/to/coords2.star',
    ...     'class_id': '1,2',
    ...     'coordinate_file_type': 'star',
    ...     'training_data_dir': '/output/path'
    ... })
    >>> prepare_data(config)

    Raises
    ------
    ValueError
        If configuration lists have mismatched lengths or invalid values
    FileNotFoundError
        If input directories or files don't exist
    """

    from tomocpt.labels.helpers import prepare_picking_star, prepare_picking_imod

    # Parse comma-separated strings into lists
    particle_length_ang, raw_data_dir, coordinate_files, class_id = config.parse_lists()

    # Rest of your existing prepare_labels code...
    validate_config_lists(config)

    base_output_dir = Path(config.training_data_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_params = zip(particle_length_ang, raw_data_dir, coordinate_files, class_id)

    # Process each dataset
    for idx, (particle_length, raw_data_dir, coordinate_files, class_id) in enumerate(
        dataset_params, 1
    ):
        logging.info(f"Processing dataset:{idx} and classID: {class_id}")

        # Create dataset-specific output directory
        dataset_output_dir = Path(
            base_output_dir / f"dataset_{idx}_class_{class_id}"
        ).resolve()
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if config.coordinate_file_type == PrepareDataType.star:
                prepare_picking_star(
                    input_file=coordinate_files,
                    tomo_path=raw_data_dir,
                    output_dir=dataset_output_dir,
                    class_id=class_id,
                    particle_diameter_angst=particle_length,
                )
            elif config.coordinate_file_type == PrepareDataType.imod:
                prepare_picking_imod(
                    input_file=coordinate_files,
                    tomo_path=raw_data_dir,
                    output_dir=dataset_output_dir,
                    particle_diameter_angst=particle_length,
                )

            logging.info(f"Successfully processed dataset {idx}")

        except Exception as e:
            logging.error(f"Error processing dataset {idx}: {str(e)}")
            continue

    logging.info("Completed processing all datasets")
    OmegaConf.save(config, os.path.join(base_output_dir, "prep_config.yaml"))


if __name__ == "__main__":
    prepare_data()
