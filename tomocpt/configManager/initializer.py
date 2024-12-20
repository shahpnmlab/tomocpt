from dataclasses import is_dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict
from omegaconf import MISSING
from typing_extensions import get_type_hints


def is_annotated_field(field_type: Any) -> bool:
    """Check if a field is annotated using typing.Annotated"""
    return str(field_type).startswith("typing.Annotated[")


def get_annotated_fields(config_obj: Any) -> Dict[str, Any]:
    """Get all annotated fields from a dataclass"""
    if not is_dataclass(config_obj):
        return {}

    type_hints = get_type_hints(type(config_obj), include_extras=True)
    return {
        field.name: type_hints[field.name]
        for field in fields(config_obj)
        if field.name in type_hints and is_annotated_field(type_hints[field.name])
    }


def convert_config_to_dict(config_obj: Any) -> dict:
    """
    Convert config object to dict, only including annotated fields
    with proper enum handling
    """
    if not is_dataclass(config_obj):
        return {}

    config_dict = {}
    annotated_fields = get_annotated_fields(config_obj)

    for field_name in annotated_fields:
        value = getattr(config_obj, field_name)

        # Handle different types of values
        if isinstance(value, Enum):
            config_dict[field_name] = value.name
        elif value is NotImplemented:
            config_dict[field_name] = MISSING
        elif isinstance(value, Path):
            config_dict[field_name] = str(value)
        elif is_dataclass(value):
            # Recursively handle nested dataclasses
            nested_dict = convert_config_to_dict(value)
            if nested_dict:  # Only include if there are annotated fields
                config_dict[field_name] = nested_dict
        else:
            config_dict[field_name] = value

    return config_dict


def initialize_config(output_path: Path = Path.cwd() / "config.yaml") -> None:
    """
    Function to create a template config file for running tomoCPT,
    only including annotated fields

    Args:
        output_path (Path): Path where the config file should be saved.
            Defaults to 'config.yaml' in the current working directory.
    """
    # Convert path to Path object if string is provided
    if isinstance(output_path, str):
        output_path = Path(output_path)

    # Create config instances
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.defaultConfigs.infer_config import InferConfig
    from tomocpt.defaultConfigs.train_config import TrainConfig
    from tomocpt.defaultConfigs.network_config import NetworkConfig

    prepare_conf = PrepdataConfig()
    train_conf = TrainConfig()
    network_conf = NetworkConfig()
    infer_conf = InferConfig()

    # Prepare the config dictionary with only annotated fields
    config_dict = {
        "prep": convert_config_to_dict(prepare_conf),
        "train": convert_config_to_dict(train_conf),
        "network": convert_config_to_dict(network_conf),
        "infer": convert_config_to_dict(infer_conf)
    }

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save using OmegaConf
    from omegaconf import OmegaConf
    OmegaConf.save(config_dict, output_path)

    print(f"Config file created at: {output_path}")


if __name__ == '__main__':
    initialize_config(Path("/tmp/config.yaml"))