from tomocpt.defaultConfigs.train_config import TrainConfig
from tomocpt.defaultConfigs.network_config import NetworkConfig
from pathlib import Path
import yaml
from enum import Enum


def convert_config_to_dict(config_obj):
    """Helper function to convert config object to dict with proper enum handling"""
    config_dict = {}
    for key, value in config_obj.__dict__.items():
        # Convert enums to their string values
        if isinstance(value, Enum):
            config_dict[key] = value.name
        else:
            config_dict[key] = value
    return config_dict


def init(path_to_save_file: Path = Path.cwd() / "config.yaml") -> None:
    """
    Function to create a template config file for running tomoCPT

    Args:
        path_to_save_file (Path): Path where the config file should be saved.
            Defaults to 'config.yaml' in the current working directory.

    Returns:
        None: Creates a config file at the specified location
    """
    # Convert path to Path object if string is provided
    if isinstance(path_to_save_file, str):
        path_to_save_file = Path(path_to_save_file)

    # Create config instances
    train_conf = TrainConfig()
    network_conf = NetworkConfig()

    # Prepare the config dictionary with proper enum handling and correct section names
    config_dict = {
        "train": convert_config_to_dict(train_conf),
        "network": convert_config_to_dict(network_conf)
    }

    # Create parent directories if they don't exist
    path_to_save_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the combined config to file
    with open(path_to_save_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Config file created at: {path_to_save_file}")


if __name__ == '__main__':
    init(Path("/tmp/config.yaml"))