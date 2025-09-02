from dataclasses import is_dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict
from omegaconf import MISSING, DictConfig
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


def convert_config_to_dict(config_obj: Any, ignore_fields: set = None) -> dict:
    """
    Convert config object to dict, including annotated fields and recursively
    handling nested dataclasses.
    """
    if not is_dataclass(config_obj):
        return {}

    config_dict = {}
    local_ignore_fields = ignore_fields if ignore_fields is not None else set()

    for field in fields(config_obj):
        if field.name in local_ignore_fields:
            continue

        field_name = field.name
        value = getattr(config_obj, field_name)

        # Recursively handle nested dataclasses (don't pass ignore_fields)
        if is_dataclass(value):
            nested_dict = convert_config_to_dict(value)
            if nested_dict:
                config_dict[field_name] = nested_dict
        # Only include annotated fields
        elif is_annotated_field(field.type):
            if isinstance(value, Enum):
                config_dict[field_name] = value.name
            elif value is NotImplemented or value is MISSING:
                config_dict[field_name] = "???"
            elif isinstance(value, Path):
                config_dict[field_name] = str(value)
            else:
                config_dict[field_name] = value

    return config_dict


def init(
    output_path: Path = Path.cwd() / "config.yaml", config: DictConfig = None
) -> None:
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
    from tomocpt.mainConfig import MainConfig

    main_conf = MainConfig()

    # Build config dict, separating shared fields
    main_conf_dict = {}
    shared_fields = {f.name for f in fields(main_conf.shared)}

    # Handle sections, ignoring shared fields in sub-configs
    for config_name in ["shared", "prepData", "train", "infer"]:
        config_obj = getattr(main_conf, config_name)
        ignore = shared_fields if config_name != "shared" else None
        section_dict = convert_config_to_dict(config_obj, ignore_fields=ignore)
        if section_dict:
            main_conf_dict[config_name] = section_dict

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save using OmegaConf
    from omegaconf import OmegaConf

    OmegaConf.save(main_conf_dict, output_path)

    print(f"Config file created at: {output_path}")

if __name__ == "__main__":
    init(Path("/tmp/config.yaml"))
