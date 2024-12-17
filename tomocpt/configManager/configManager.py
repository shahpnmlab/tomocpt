import copy
import sys
from dataclasses import is_dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Tuple, Set
from typing_extensions import Annotated
from pathlib import Path
import inspect
import typer
from omegaconf import OmegaConf, DictConfig, MISSING
from enum import Enum
from functools import wraps


def _normalize_path(path: str) -> str:
    """Normalize parameter path to dot notation"""
    return path.replace("__", ".")

typer.main.get_command_name = lambda name: _normalize_path(name) #.replace("_", "-")


class ConfigurationError(Exception):
    """Raised when there are conflicts in configuration settings"""
    pass


class MergePreference(str, Enum):
    COMMAND = "command"
    CONFIG_FILE = "configFile"


class ConfigManager:
    def __init__(self):
        self.registered_configs = {}

    def _get_default_value(self, config_obj: Any, path: str) -> Any:
        """Get the default value for a given parameter path"""
        parts = path.split('.')
        curr_obj = config_obj
        try:
            for part in parts[:-1]:
                curr_obj = getattr(curr_obj, part)
            return getattr(curr_obj, parts[-1])
        except AttributeError:
            return None

    def _get_valid_paths(self, config_obj: Any, prefix: str = '') -> Set[str]:
        paths = set()
        for field in fields(type(config_obj)):
            field_value = getattr(config_obj, field.name)
            current_path = f"{prefix}.{field.name}" if prefix else field.name
            paths.add(current_path)

            if is_dataclass(type(field_value)):
                paths.update(self._get_valid_paths(field_value, current_path))
        return paths

    def _process_nested_dataclass(self, obj: Any, prefix: str = '') -> List[
        Tuple[str, Any, typer.models.OptionInfo, Any]]:
        """Recursively process dataclasses but only create CLI params for Annotated fields"""
        params = []

        for field in fields(type(obj)):
            field_type = field.type
            field_value = getattr(obj, field.name)
            current_path = f"{prefix}{field.name}" if prefix else field.name

            if str(field.type).startswith("typing.Annotated["):
                field_type = field.type.__args__[0]
                if is_dataclass(field_type):
                    if field_value is MISSING:
                        field_value = field_type()
                    new_prefix = f"{prefix}{field.name}__" if prefix else f"{field.name}__"
                    nested_params = self._process_nested_dataclass(field_value, new_prefix)
                    params.extend(nested_params)
                elif any(isinstance(m, typer.models.OptionInfo) for m in field.type.__metadata__):
                    for metadata in field.type.__metadata__:
                        if isinstance(metadata, typer.models.OptionInfo):
                            if field_value is MISSING:
                                new_option = typer.Option(
                                    default=...,
                                    help=metadata.help,
                                    **{k: v for k, v in metadata.__dict__.items()
                                       if k not in ['default', 'help', 'param_decls']}
                                )
                                params.append((current_path, field_type, new_option, MISSING))
                            else:
                                params.append((current_path, field_type, metadata, field_value))
            elif is_dataclass(type(field_value)):
                # Add nested paths for hydra-style access without creating CLI params
                new_prefix = f"{prefix}{field.name}__" if prefix else f"{field.name}__"
                nested_params = self._process_nested_dataclass(field_value, new_prefix)
                params.extend(nested_params)

        return params

    def _get_annotated_options(self, config_obj: Any) -> Tuple[List[tuple], List[tuple], Set[str]]:
        """Returns (required_params, optional_params, valid_paths)"""
        if config_obj is None:
            return [], [], set()

        all_params = self._process_nested_dataclass(config_obj)
        # print([p[0] for p in all_params])
        # breakpoint()
        required_params = []
        optional_params = []
        valid_paths = self._get_valid_paths(config_obj)

        for param_name, field_type, option, field_value in all_params:
            if field_value is MISSING:
                required_params.append((param_name, field_type, option))
            else:
                optional_params.append((param_name, field_type, option, field_value))

        return required_params, optional_params, valid_paths

    def create_cli_parameters(self, config_obj: Any) -> Tuple[List[inspect.Parameter], Set[str]]:
        if config_obj is None:
            return [], set([])
        required_params, optional_params, valid_paths = self._get_annotated_options(config_obj)

        # Create required parameters (without defaults)
        required_parameters = [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,  # Make required params positional
                annotation=Annotated[type_, option]
            )
            for name, type_, option in required_params
        ]

        # Create optional parameters (with defaults)
        optional_parameters = [
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,  # Keep optional params as keyword-only
                annotation=Annotated[type_, option],
                default=default
            )
            for name, type_, option, default in optional_params
        ]

        return [*required_parameters, *optional_parameters], valid_paths


class ConfigurableApp:
    def __init__(self):
        self.app = typer.Typer(pretty_exceptions_show_locals=False,
                               pretty_exceptions_enable=False,
                               pretty_exceptions_short=True,
                               no_args_is_help=True,
                               add_completion=False)
        self.config_manager = ConfigManager()

    def _get_typed_value(self, key: str, value: str, curr_obj: Any, attr: str) -> Any:
        """Get properly typed value from type hints for config parameters"""
        field_type = None

        # Get the field from the dataclass
        for field in fields(type(curr_obj)):
            if field.name == attr:
                if str(field.type).startswith("typing.Optional"):
                    if value.lower() == "none":
                        return None
                    field_type = field.type.__args__[0]  # Get the inner type of Optional
                elif str(field.type).startswith("typing.Annotated"):
                    field_type = field.type.__args__[0]  # Get the actual type from Annotated
                else:
                    field_type = field.type
                break

        if field_type is None:
            raise ConfigurationError(f"Could not determine type for parameter '{key}'")

        try:
            return field_type(value)
        except ValueError:
            raise ConfigurationError(f"Could not convert value '{value}' to type {field_type} for parameter '{key}'")

    def command(self, *typer_args, config: Any = None, **typer_kwargs):
        def decorator(func: Callable):
            # Get config file value if present
            config_file_path = None
            for i, arg in enumerate(sys.argv):
                if arg == '--config_file':
                    if i + 1 < len(sys.argv):
                        config_file_path = sys.argv[i + 1]
                elif arg.startswith('--config_file='):
                    config_file_path = arg.split('=', 1)[1]

            # Load config file if present
            yaml_config = None
            yaml_updated_config = None
            if config_file_path:
                yaml_config = OmegaConf.load(config_file_path)

                if config is not None:
                    yaml_updated_config = copy.deepcopy(config)
                    merge_config(yaml_config, yaml_updated_config) #This hides the required parameters from help if they are defined in the yaml

            original_sig = inspect.signature(func)
            func_params = [p for p in original_sig.parameters.values() if p.name != 'config']
            cli_params, paths = self.config_manager.create_cli_parameters(config)
            # Create parameter mappings
            config_param_map = {param.name: param for param in cli_params}
            config_param_types = {
                param.name: param.annotation.__args__[0] if str(param.annotation).startswith("typing.Annotated")
                else param.annotation for param in cli_params
            }

            # Validate function parameters
            for func_param in func_params:
                if func_param.name in config_param_map:
                    config_type = config_param_types[func_param.name]
                    func_type = func_param.annotation if func_param.annotation != inspect.Parameter.empty else type(
                        None)
                    if func_type != config_type:
                        raise ConfigurationError(
                            f"Type mismatch for '{func_param.name}': function expects {func_type}, config provides {config_type}"
                        )

                    config_param = config_param_map[func_param.name]
                    config_default = config_param.default
                    func_default = func_param.default

                    if func_default != inspect.Parameter.empty and config_default != inspect.Parameter.empty:
                        if func_default != config_default:
                            raise ConfigurationError(
                                f"Default value mismatch for '{func_param.name}': function:{func_default}, config:{config_default}"
                            )
                    elif func_default != inspect.Parameter.empty:
                        raise ConfigurationError(
                            f"Parameter '{func_param.name}' required in config but has default {func_default} in function"
                        )

            extra_params = [
                inspect.Parameter(
                    'config_file',
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[Optional[Path], typer.Option(help="Config file")],
                    default=None
                ),
                inspect.Parameter(
                    'config_merge_preference',
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[Optional[MergePreference], typer.Option()],
                    default=None
                ),
                inspect.Parameter(
                    'config_args',
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=Annotated[Optional[List[str]], typer.Argument(help="Hydra-style config modifications")]
                )
            ]

            # Update cli_params with values from config file if present
            if yaml_config:
                updated_cli_params = []
                for param in cli_params:
                    param_path = _normalize_path(param.name).split('.')
                    curr_config = yaml_updated_config
                    found_value = True

                    # Navigate the yaml config structure
                    for p in param_path:
                        if hasattr(curr_config, p):
                            curr_config = getattr(curr_config, p)
                        else:
                            found_value = False
                            break
                    if found_value:
                        # Create new parameter with default from yaml
                        if curr_config == MISSING:
                            kwargs = {}
                        else:
                            kwargs = dict(default=curr_config)
                        new_param = inspect.Parameter(
                            param.name,
                            param.kind,
                            annotation=param.annotation,
                            **kwargs
                        )
                        updated_cli_params.append(new_param)
                    else:
                        updated_cli_params.append(param)
                cli_params = updated_cli_params

            # Organize parameters
            required_params = []
            optional_params = []
            processed_params = set()

            for param in func_params:
                if param.name in config_param_map:
                    config_param = config_param_map[param.name]
                    (required_params if config_param.default == inspect.Parameter.empty else optional_params).append(
                        config_param)
                    processed_params.add(param.name)
                else:
                    (required_params if param.default == inspect.Parameter.empty and param.name != "kwargs"
                     else optional_params).append(param)

            for param in cli_params:
                if param.name not in processed_params:
                    (required_params if param.default == inspect.Parameter.empty else optional_params).append(param)

            optional_params.extend(extra_params)

            @wraps(func)
            def wrapped(*args, config_args: Optional[List[str]] = None, config_file: Optional[Path] = None,
                        config_merge_preference: Optional[MergePreference] = None, **kwargs):
                cli_params = {}
                for k, v in kwargs.items():
                    if k in config_param_map:
                        cli_params[k] = v
                        path = k.split('__')
                        curr_obj = config
                        for p in path[:-1]:
                            curr_obj = getattr(curr_obj, p)
                        setattr(curr_obj, path[-1], v)

                if config_args:
                    for arg in config_args:
                        if '=' not in arg:
                            raise typer.BadParameter(f"Invalid arg {arg}. Missing '='?")
                        key, value = arg.split('=', 1)
                        path = key.split('.')
                        curr_obj = config

                        for p in path[:-1]:
                            if not hasattr(curr_obj, p):
                                raise ConfigurationError(f"Invalid path: {key}")
                            curr_obj = getattr(curr_obj, p)

                        if not hasattr(curr_obj, path[-1]):
                            raise ConfigurationError(f"Invalid parameter: {key}")

                        # Get field type from dataclass
                        field_type = None
                        for field in fields(type(curr_obj)):
                            if field.name == path[-1]:
                                if str(field.type).startswith("typing.Optional"):
                                    if value.lower() == "none":
                                        setattr(curr_obj, path[-1], None)
                                        continue
                                    field_type = field.type.__args__[0]
                                elif str(field.type).startswith("typing.Annotated"):
                                    field_type = field.type.__args__[0]
                                else:
                                    field_type = field.type
                                break

                        if field_type is None:
                            raise ConfigurationError(f"Cannot determine type for '{key}'")

                        try:
                            if isinstance(field_type, type) and issubclass(field_type, Enum):
                                typed_value = field_type(value)
                            elif str(field_type).startswith("typing.Tuple"):
                                tuple_types = field_type.__args__
                                cleaned_value = value.strip("()[]{}").split(",")
                                if len(cleaned_value) != len(tuple_types):
                                    raise ConfigurationError(
                                        f"Tuple length mismatch for {key}: expected {len(tuple_types)}, got {len(cleaned_value)}"
                                    )
                                typed_value = tuple(t(v.strip()) for t, v in zip(tuple_types, cleaned_value))
                            else:
                                typed_value = field_type(value)
                            setattr(curr_obj, path[-1], typed_value)
                        except ValueError as e:
                            raise ConfigurationError(
                                f"Cannot convert '{value}' to {field_type} for '{key}': {str(e)}"
                            )

                # if config_file:
                #     yaml_config = OmegaConf.load(config_file)
                #     self._update_from_yaml(config, yaml_config, config_merge_preference)

                func_kwargs = {k: v for k, v in kwargs.items() if k not in config_param_map}
                if 'config' in original_sig.parameters:
                    func_kwargs['config'] = config

                return func(*args, **func_kwargs)

            wrapped.__signature__ = inspect.Signature(parameters=[*required_params, *optional_params])
            return self.app.command(*typer_args, **typer_kwargs)(wrapped)

        return decorator

    def _update_from_yaml(self, config_obj: Any, yaml_config: DictConfig,
                          merge_preference: Optional[MergePreference]) -> None:
        """Update config object from YAML config"""

        # Track updates to detect conflicts
        yaml_updates = {}

        def _update_value(obj: Any, key: str, value: Any, path: str = ''):
            if hasattr(obj, key):
                current_value = getattr(obj, key)
                full_path = f"{path}.{key}" if path else key

                # If the attribute exists and is either a simple type or None
                if isinstance(current_value, (int, float, str, bool)) or current_value is None:
                    # Check if value would be changed
                    try:
                        typed_value = type(current_value)(value) if current_value is not None else value
                        if current_value is not None and current_value != typed_value:
                            if merge_preference is None:
                                raise ConfigurationError(
                                    f"Conflict detected for parameter '{full_path}': "
                                    f"Current value: {current_value}, YAML value: {value}. "
                                    "Please specify merge_preference to resolve conflicts."
                                )
                            if merge_preference == MergePreference.CONFIG_FILE:
                                yaml_updates[full_path] = (current_value, typed_value)
                                setattr(obj, key, typed_value)
                                print(f"Updated {full_path}: {current_value} -> {typed_value}")
                            else:  # MergePreference.COMMAND
                                print(f"Keeping current value for {full_path}: {current_value} (YAML value: {value})")
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert {value} to {type(current_value)} for {full_path}")
                elif is_dataclass(type(current_value)):
                    if hasattr(value, 'items'):  # Handles both dict and DictConfig
                        for k, v in value.items():
                            _update_value(current_value, k, v, full_path)

        # Handle flat config structure in YAML
        for section, values in yaml_config.items():
            if hasattr(values, 'items'):  # Handles both dict and DictConfig
                # Try to find matching section in config
                if hasattr(config_obj, section):
                    section_obj = getattr(config_obj, section)
                    if is_dataclass(type(section_obj)):
                        for key, value in values.items():
                            _update_value(section_obj, key, value, section)
                else:
                    # Try to match flattened keys
                    for key, value in values.items():
                        # Handle nested keys like 'n_epochs' in 'train' section
                        if hasattr(config_obj, 'train') and hasattr(getattr(config_obj, 'train'), key):
                            _update_value(getattr(config_obj, 'train'), key, value, 'train')
            else:
                # Handle top-level values
                _update_value(config_obj, section, values)

        # Report all updates if any occurred
        if yaml_updates and merge_preference == MergePreference.CONFIG_FILE:
            print("\nSummary of updates from YAML:")
            for path, (old_val, new_val) in yaml_updates.items():
                print(f"  {path}: {old_val} -> {new_val}")

    def run(self):
        self.app()

    def register_command(self, func, config):  # TODO: can we have this as a class method in app?
        @self.command(config=config)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


def merge_config(yaml_config: DictConfig, config: Any) -> Any:
    """
    Merge YAML config into a dataclass config object.
    """
    if not is_dataclass(config):
        return config

    # For each field in the dataclass
    for field in fields(config):
        field_name = field.name
        current_value = getattr(config, field_name)

        # Find the field in yaml_config, checking both top level and nested
        yaml_value = None
        if field_name in yaml_config:
            yaml_value = yaml_config[field_name]
        else:
            # Search in all top-level dictionaries
            for section in yaml_config.values():
                if isinstance(section, DictConfig) and field_name in section:
                    yaml_value = section[field_name]
                    break

        if yaml_value is not None:
            if is_dataclass(current_value) and isinstance(yaml_value, DictConfig):
                # Recursively merge nested dataclass
                setattr(config, field_name, merge_config(yaml_value, current_value))
            else:
                # Direct update for non-dataclass fields
                setattr(config, field_name, yaml_value)
    return config




def update_config(target: Any, source: Any):
    """
    Recursively updates the fields of a target dataclass with values from the source dataclass.
    Handles omegaconf.MISSING by keeping the non-MISSING value.
    """
    if not (is_dataclass(target) and is_dataclass(source)):
        raise ValueError("Both target and source must be dataclass instances of the same type.")

    for field in fields(target):
        field_name = field.name
        target_value = getattr(target, field_name)
        source_value = getattr(source, field_name)

        # Handle MISSING values
        if source_value == MISSING and target_value != MISSING:
            continue  # Keep the target value
        elif target_value == MISSING and source_value != MISSING:
            setattr(target, field_name, source_value)
            continue

        # Check if the field is a nested dataclass
        if is_dataclass(target_value) and is_dataclass(source_value):
            update_config(target_value, source_value)
        else:
            # Update the field value in the target dataclass
            setattr(target, field_name, source_value)


def compare_dataclasses(obj1: Any, obj2: Any, path: str = "") -> List[Tuple[str, Any, Any]]:
    """
    Compare two dataclass instances and return a list of differences with their full path names.

    Args:
        obj1: First dataclass instance
        obj2: Second dataclass instance
        path: Current path in the nested structure (used recursively)

    Returns:
        List of tuples containing (full_path, value1, value2) for all different values
    """
    if not is_dataclass(obj1) or not is_dataclass(obj2):
        raise ValueError("Both objects must be instances of dataclasses")

    if obj1.__class__ != obj2.__class__:
        raise ValueError("Objects must be instances of the same dataclass")

    differences = []

    for field in fields(obj1):
        field_name = field.name
        value1 = getattr(obj1, field_name)
        value2 = getattr(obj2, field_name)

        current_path = f"{path}.{field_name}" if path else field_name

        # If the field is itself a dataclass, recurse
        if is_dataclass(value1) and is_dataclass(value2):
            nested_diffs = compare_dataclasses(value1, value2, current_path)
            differences.extend(nested_diffs)
        # If the field is a list or tuple, compare elements
        elif isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
            if len(value1) != len(value2):
                differences.append((current_path, value1, value2))
            else:
                for i, (item1, item2) in enumerate(zip(value1, value2)):
                    if is_dataclass(item1) and is_dataclass(item2):
                        nested_diffs = compare_dataclasses(item1, item2, f"{current_path}[{i}]")
                        differences.extend(nested_diffs)
                    elif item1 != item2:
                        differences.append((f"{current_path}[{i}]", item1, item2))
        # For basic types, direct comparison
        elif value1 != value2:
            differences.append((current_path, value1, value2))

    return differences


def update_config_with_changed_values(target: Any, originaConfig:Any, configAfterCli:Any) -> Any:
    """
    Apply the differences found between two dataclasses to a target dataclass instance.

    Args:
        target: The dataclass instance to update
        differences: List of differences from compare_dataclasses

    Returns:
        Updated copy of the target instance
    """

    differences = compare_dataclasses(originaConfig, configAfterCli)

    # Create a deep copy to avoid modifying the original
    result = copy.deepcopy(target)

    for path, _, new_value in differences:
        # Split the path into parts
        parts = path.split('.')

        # Handle array indices
        current_obj = result
        for i, part in enumerate(parts[:-1]):  # All parts except the last one
            # Check if this part contains an array index
            if '[' in part:
                # Split into name and index
                name, idx = part.split('[')
                idx = int(idx.rstrip(']'))

                # Get the array/list
                if name:  # If there's a name before the index
                    current_obj = getattr(current_obj, name)
                current_obj = current_obj[idx]
            else:
                current_obj = getattr(current_obj, part)

        # Handle the final part
        last_part = parts[-1]
        if '[' in last_part:
            name, idx = last_part.split('[')
            idx = int(idx.rstrip(']'))
            if name:
                current_obj = getattr(current_obj, name)
            current_obj[idx] = new_value
        else:
            setattr(current_obj, last_part, new_value)

    return result

def create_app() -> ConfigurableApp:
    return ConfigurableApp()