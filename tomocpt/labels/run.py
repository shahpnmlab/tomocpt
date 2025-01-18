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
        'particle_length_ang': len(config.particle_length_ang) if isinstance(config.particle_length_ang,
                                                                             (list, tuple)) else 1,
        'raw_data_dir': len(config.raw_data_dir) if isinstance(config.raw_data_dir, (list, tuple)) else 1,
        'coordinate_files': len(config.coordinate_files) if isinstance(config.coordinate_files, (list, tuple)) else 1,
        'class_id': len(config.class_id) if isinstance(config.class_id, (list, tuple)) else 1
    }

    reference_length = next(iter(list_lengths.values()))
    mismatched = {k: v for k, v in list_lengths.items() if v != reference_length}

    if mismatched:
        raise ConfigurationError(
            f"Configuration lists must have matching lengths. Mismatched lengths: {mismatched}. Expected: {reference_length}")


def prepare_vol_label_pairs(config: DictConfig = None) -> None:
    """Process multiple datasets based on configuration"""

    from tomocpt.labels.helpers import prepare_picking_star, prepare_picking_imod
    # Parse comma-separated strings into lists
    particle_length_ang, raw_data_dir, coordinate_files, class_id = config.parse_lists()

    # Rest of your existing prepare_labels code...
    validate_config_lists(config)

    base_output_dir = Path(config.prepared_data_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_params = zip(
        particle_length_ang,
        raw_data_dir,
        coordinate_files,
        class_id
    )

    # Process each dataset
    for idx, (particle_length, raw_data_dir, coordinate_files, class_id) in enumerate(dataset_params, 1):
        logging.info(f"Processing dataset:{idx} and classID: {class_id}")

        # Create dataset-specific output directory
        dataset_output_dir = Path(base_output_dir / f"dataset_{idx}_class_{class_id}").resolve()
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if config.coordinate_file_type == PrepareDataType.star:
                prepare_picking_star(
                    input_file=coordinate_files,
                    tomo_path=raw_data_dir,
                    output_dir=dataset_output_dir,
                    class_id=class_id,
                    particle_diameter_angst=particle_length
                )
            elif config.coordinate_file_type == PrepareDataType.imod:
                prepare_picking_imod(
                    input_file=coordinate_files,
                    tomo_path=raw_data_dir,
                    output_dir=dataset_output_dir,
                    particle_diameter_angst=particle_length
                )

            logging.info(f"Successfully processed dataset {idx}")

        except Exception as e:
            logging.error(f"Error processing dataset {idx}: {str(e)}")
            continue

    logging.info("Completed processing all datasets")
    OmegaConf.save(config, os.path.join(base_output_dir, "prep_config.yaml"))


if __name__ == '__main__':
    prepare_vol_label_pairs()