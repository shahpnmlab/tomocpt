from typer import Option
import pandas as pd
from omegaconf import DictConfig


from tomocpt.labels.helpers import *
from tomocpt.defaultConfigs.prepdata_config import PrepareDataType


class ConfigurationError(Exception):
    """Raised when there are issues with the configuration"""
    pass


def validate_config_lists(config: DictConfig) -> None:
    """
    Validate that all input lists in the config have matching lengths

    Args:
        config: The configuration object containing the lists

    Raises:
        ConfigurationError: If lists have mismatched lengths
    """
    list_lengths = {
        'particle_length_ang': len(config.particle_length_ang),
        'raw_data_dir': len(config.raw_data_dir),
        'input_file': len(config.input_file),
        'class_id': len(config.class_id)
    }

    # Get the first length as reference
    reference_length = next(iter(list_lengths.values()))

    # Check if all lists have the same length
    mismatched = {key: length for key, length in list_lengths.items()
                  if length != reference_length}

    if mismatched:
        raise ConfigurationError(
            f"Configuration lists must have matching lengths. "
            f"Mismatched lengths found: {mismatched}. "
            f"Expected length: {reference_length}"
        )


def prepare_labels(config: DictConfig = None) -> None:
    """
    Process multiple datasets based on configuration with parallel lists

    Args:
        config: Configuration object containing lists of parameters for multiple datasets
    """

    from tomocpt.labels.helpers import prepare_picking_star, prepare_picking_imod
    # Validate that all input lists have matching lengths
    validate_config_lists(config)

    # Create base output directory
    base_output_dir = Path(config.prepared_data_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Zip together all the parallel lists
    dataset_params = zip(
        config.particle_length_ang,
        config.raw_data_dir,
        config.input_file,
        config.class_id
    )

    # Process each dataset
    for idx, (particle_length, raw_data_dir, input_file, class_id) in enumerate(dataset_params, 1):
        logging.info(f"Processing dataset {idx}")

        # Create dataset-specific output directory
        dataset_output_dir = base_output_dir / f"dataset_{idx}"
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if config.coordinate_file_type == PrepareDataType.star:
                prepare_picking_star(
                    input_file=input_file,
                    tomo_path=raw_data_dir,
                    output_dir=dataset_output_dir,
                    class_id=class_id,
                    particle_diameter_angst=particle_length
                )
            elif config.coordinate_file_type == PrepareDataType.imod:
                prepare_picking_imod(
                    input_file=input_file,
                    tomo_path=raw_data_dir,
                    output_dir=dataset_output_dir,
                    particle_diameter_angst=particle_length
                )

            logging.info(f"Successfully processed dataset {idx}")

        except Exception as e:
            logging.error(f"Error processing dataset {idx}: {str(e)}")
            continue

    logging.info("Completed processing all datasets")


if __name__ == '__main__':
    prepare_labels()