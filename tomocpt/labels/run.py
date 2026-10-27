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

    border_policy = _border_policy(config)
    apix_override = getattr(config, "apix_override", None)

    if bool(getattr(config, "multiclass", False)):
        _prepare_data_multiclass(config, border_policy, apix_override)
        return

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
                    border_policy=border_policy,
                    apix_override=apix_override,
                )
            elif config.coordinate_file_type == PrepareDataType.imod:
                prepare_picking_imod(
                    input_file=coordinate_files,
                    tomo_path=raw_data_dir,
                    output_dir=dataset_output_dir,
                    particle_diameter_angst=particle_length,
                    border_policy=border_policy,
                    apix_override=apix_override,
                )

            logging.info(f"Successfully processed dataset {idx}")

        except Exception as e:
            logging.error(f"Error processing dataset {idx}: {str(e)}")
            continue

    logging.info("Completed processing all datasets")
    OmegaConf.save(config, os.path.join(base_output_dir, "prep_config.yaml"))


def _border_policy(config) -> str:
    policy = getattr(config, "label_border_policy", "skip")
    policy = str(getattr(policy, "value", policy))
    if policy not in ("skip", "clip"):
        raise ConfigurationError(f"label_border_policy must be 'skip' or 'clip', got '{policy}'")
    return policy


def _split_csv(text) -> list:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _prepare_data_multiclass(config, border_policy: str, apix_override) -> None:
    """prepare_data with prepData.multiclass=True: every (raw_data_dir, coordinate_files)
    pair is one dataset that holds all species (STAR files with rlnClassNumber)."""
    from tomocpt.labels.helpers import prepare_picking_star_multiclass
    from tomocpt.labels.multiclass import class_table_from_config

    if config.coordinate_file_type != PrepareDataType.star:
        raise ConfigurationError("prepData.multiclass=True needs STAR coordinate files with rlnClassNumber")
    class_table = class_table_from_config(config)  # raises if class_diameters is missing or invalid

    raw_dirs = [Path(x) for x in _split_csv(config.raw_data_dir)]
    coord_files = [Path(x) for x in _split_csv(config.coordinate_files)]
    if len(raw_dirs) != len(coord_files):
        raise ConfigurationError(
            f"raw_data_dir has {len(raw_dirs)} entries but coordinate_files has {len(coord_files)}")
    class_ids = _split_csv(getattr(config, "class_id", "all")) or ["all"]
    label = class_ids[0] if len(class_ids) == 1 else "all"

    base_output_dir = Path(config.training_data_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (raw_dir, coord_file) in enumerate(zip(raw_dirs, coord_files), 1):
        logging.info(f"Processing multi-class dataset:{idx} ({coord_file})")
        dataset_output_dir = Path(base_output_dir / f"dataset_{idx}_class_{label}").resolve()
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        # Errors are raised (not logged and skipped): a missing class diameter must stop here.
        prepare_picking_star_multiclass(
            input_file=coord_file,
            tomo_path=raw_dir,
            output_dir=dataset_output_dir,
            class_table=class_table,
            class_annotated=getattr(config, "class_annotated", None),
            class_id_label=label,
            border_policy=border_policy,
            apix_override=apix_override,
        )
        logging.info(f"Successfully processed dataset {idx}")

    logging.info("Completed processing all datasets")
    OmegaConf.save(config, os.path.join(base_output_dir, "prep_config.yaml"))


if __name__ == "__main__":
    prepare_data()
