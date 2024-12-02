from omegaconf import MISSING
from typing import Annotated

import typer
from enum import Enum
from pathlib import Path
from omegaconf import OmegaConf

from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.defaultConfigs.train_config import TrainConfig
from tomocpt.defaultConfigs.network_config import NetworkConfig


def convert_config_to_dict(config_obj):
    """Helper function to convert config object to dict with proper enum handling"""
    config_dict = {}
    for key, value in config_obj.__dict__.items():
        # Convert enums to their string values
        if isinstance(value, Enum):
            config_dict[key] = value.name
        else:
            if value is NotImplemented:
                value = MISSING
            if isinstance(value, Path):
                value = str(value)
            config_dict[key] = value

    return config_dict


def init(output_path: Annotated[Path, typer.Option(help="Path to save config.yaml file")] = Path.cwd() / "config.yaml") -> None:
    """
    Function to create a template config file for running tomoCPT

    Args:
        output_path (Path): Path where the config file should be saved.
            Defaults to 'config.yaml' in the current working directory.

    Returns:
        None: Creates a config file at the specified location
        :param config:
    """
    # Convert path to Path object if string is provided
    if isinstance(output_path, str):
        output_path = Path(output_path)

    # Create config instances
    prepare_conf = PrepdataConfig()
    train_conf = TrainConfig()
    network_conf = NetworkConfig()
    infer_conf = InferConfig()
    # Prepare the config dictionary with proper enum handling and correct section names
    config_dict = {
        "prep": convert_config_to_dict(prepare_conf),
        "train": convert_config_to_dict(train_conf),
        "network": convert_config_to_dict(network_conf),
        "infer": convert_config_to_dict(infer_conf)
    }

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config_dict, output_path)
    # # Write the combined config to file
    # with open(output_path, 'w') as f:
    #     yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Config file created at: {output_path}")


if __name__ == '__main__':
    init(Path("/tmp/config.yaml"))
