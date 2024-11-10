# config_manager.py
import operator
import os.path
from dataclasses import is_dataclass, fields
from enum import Enum
from functools import wraps, reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, MutableMapping, Tuple
import inspect

import typer
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Annotated


class MergePreference(str, Enum):
    COMMAND = "command"
    CONFIG_FILE = "configFile"


T = TypeVar('T')


class ConfigurableApp:
    """Factory class for creating Typer apps with Hydra config management"""

    def __init__(self, base_config_class: Type[Any], config_store_name: str = "base_config",
                 set_global_config_fn: Optional[Callable[[DictConfig], None]]=None):

        self.base_config_class = base_config_class
        self.config_store_name = config_store_name
        self.set_global_config_fn = set_global_config_fn
        self.return_config = False
        self.app = typer.Typer()

        # Register the config with Hydra
        cs = ConfigStore.instance()
        cs.store(name=config_store_name, node=base_config_class)

    def _extract_override_values(self, overrides: List[str]) -> Dict[str, str]:
        values = {}
        for override in overrides:
            key, value = override.split('=', 1)
            values[key.strip()] = value.strip()
        return values

    def _extract_config_values(self, config: DictConfig) -> Dict[str, Any]:
        values = {}

        def extract_recursive(cfg: Union[Dict, DictConfig], prefix: str = ""):
            for key, value in cfg.items():
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, (dict, DictConfig)):
                    extract_recursive(value, f"{full_key}.")
                else:
                    values[full_key] = value

        extract_recursive(config)
        return values

    def _find_conflicts(self, override_values: Dict[str, str], config: DictConfig) -> List[str]:
        conflicts = []
        config_dict = self._extract_config_values(config)

        for key, override_value in override_values.items():
            if key in config_dict:
                config_value = config_dict[key]
                if str(override_value) != str(config_value):
                    conflicts.append(f"{key}: command={override_value}, config={config_value}")

        return conflicts

    def _flatten_dict(self, dictionary: dict, parent_key: str = '', separator: str = '/') -> dict:
        """Flatten a nested dictionary with path information."""
        items = []
        for key, value in dictionary.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(self._flatten_dict(value, new_key, separator=separator).items())
            else:
                items.append((new_key, (key, value)))
        return dict(items)

    def _get_from_dict(self, data_dict: dict, map_list: list) -> Any:
        """Get a value from a nested dictionary using a list of keys."""
        return reduce(operator.getitem, map_list, data_dict)

    def _set_in_dict(self, data_dict: dict, map_list: list, value: Any) -> None:
        """Set a value in a nested dictionary using a list of keys."""
        self._get_from_dict(data_dict, map_list[:-1])[map_list[-1]] = value

    def _handle_config_overrides(self, func_kwargs: dict, config: DictConfig, default_config:DictConfig) \
                                                                        -> Tuple[Dict[str, Any], DictConfig]:
        """
        Handle overrides from command-line arguments that match config paths.
        Args:
            func_kwargs: Dictionary of function arguments
            config: The configuration object
            default_config: the by default configuration to check it command lines are as by default
        Returns:
            Updated configuration object
        """
        # Convert config to dict for flattening
        config_dict = OmegaConf.to_container(config, resolve=True)
        flat_config = self._flatten_dict(config_dict)
        flat_default_config = self._flatten_dict(default_config)
        # Check for overlapping parameters
        for path, (key, conf_value) in flat_config.items():
            cli_key = path.replace("/", "_")

            if cli_key in func_kwargs:
                if func_kwargs[cli_key] is not None and func_kwargs[cli_key] != flat_default_config[path][1]:
                    cli_value = func_kwargs[cli_key]
                    if str(conf_value) != str(cli_value):  # Only update if values are different
                        print(f"Overriding config conf_value '{path}': {conf_value} → {cli_value}")
                        self._set_in_dict(config_dict, path.split("/"), cli_value)
                else:
                    func_kwargs[cli_key] = conf_value


        # Convert back to DictConfig
        return func_kwargs, OmegaConf.create(config_dict)

    def add_config_options(self, func: Callable, is_command:bool = False) -> Callable:
        """Add configuration options to a function"""

        def run_command(*args, **kwargs):
            # Extract config-related parameters

            config_file = kwargs.pop('config_file', None)
            config_args = kwargs.pop('config_args', [])
            if config_args and is_command:
                config_args = config_args[1:]
            config_merge_preference = kwargs.pop('config_merge_preference', None)
            # Process the configuration
            base_cfg, cfg = self.process_config(config_file, config_args, config_merge_preference)
            # breakpoint()
            # Handle automatic overrides from command-line arguments
            kwargs, cfg = self._handle_config_overrides(kwargs, cfg, base_cfg)

            if self.set_global_config_fn is not None:
                self.set_global_config_fn(cfg)
            # Call the original function
            if self.return_config:
                return func(*args, config=cfg, **kwargs)
            else:
                assert self.set_global_config_fn is not None, ("Error, if you do not provide and set_global_config_fn,"
                                                               " then you have to add a config:DictConfig argument to "
                                                               "your function ")
                return func(*args, **kwargs)

        # Get original signature and parameters
        sig = inspect.signature(func)
        orig_params = []
        for name, p in sig.parameters.items():
            if p.name != 'config':
                if not all([isinstance(x, typer.models.OptionInfo) for x in p.annotation.__metadata__]):
                    raise typer.BadParameter(
                        f"Parameter {name} was not provided as typer.Option. typer.Argument is not supported"
                    )
                orig_params.append(p)
            else:
                self.return_config = True

        # Add the config parameters to the function
        params = [
            inspect.Parameter(
                'config_file',
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Annotated[Optional[Path], typer.Option(help="External YAML config file")]
            ),
            inspect.Parameter(
                'config_args',
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Annotated[Optional[List[str]], typer.Argument(help="Hydra-style config modifications")]
            ),
            inspect.Parameter(
                'config_merge_preference',
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Annotated[Optional[MergePreference], typer.Option(help="How to handle config conflicts")]
            )
        ]

        # Create new signature with correct parameter order
        run_command.__signature__ = sig.replace(parameters=[*orig_params, *params])
        run_command.__name__ = func.__name__
        run_command.__doc__ = func.__doc__

        return run_command
    def process_config(
            self,
            config_file: Optional[Path],
            config_args: List[str],
            config_merge_preference: Optional[MergePreference]
    ) -> Tuple[DictConfig, DictConfig]:
        with hydra.initialize(version_base=None, config_path=None):
            base_cfg = hydra.compose(config_name=self.config_store_name)

            file_cfg = None
            if config_file is not None:
                if not os.path.isfile(config_file):
                    raise typer.BadParameter(f"Config file not found: {config_file}")
                file_cfg = OmegaConf.load(config_file)

            cmd_cfg = None
            if config_args:
                cmd_cfg = hydra.compose(config_name=self.config_store_name, overrides=config_args)

            if config_args and file_cfg:
                override_values = self._extract_override_values(config_args)
                conflicts = self._find_conflicts(override_values, file_cfg)

                if conflicts:
                    if config_merge_preference is None:
                        conflict_msg = "\n  ".join(conflicts)
                        raise typer.BadParameter(
                            f"Conflicts found between config file and command arguments:\n  {conflict_msg}\n"
                            f"Use --config-merge-preference to specify preference (command/configFile)"
                        )
                    else:
                        print(f"\nResolving conflicts with preference: {config_merge_preference}")
                        print("Conflicts found:")
                        for conflict in conflicts:
                            print(f"  {conflict}")

            cfg = base_cfg.copy()
            if config_merge_preference == MergePreference.COMMAND:
                if file_cfg:
                    cfg = OmegaConf.merge(cfg, file_cfg)
                if cmd_cfg:
                    cfg = OmegaConf.merge(cfg, cmd_cfg)
            else:
                if cmd_cfg:
                    cfg = OmegaConf.merge(cfg, cmd_cfg)
                if file_cfg:
                    cfg = OmegaConf.merge(cfg, file_cfg)

            return base_cfg, cfg

    def command(self, *args, **kwargs):
        """Decorator for adding commands with config support"""

        def decorator(func: Callable):
            # Add config options to the function
            wrapped = self.add_config_options(func, is_command=True)
            # Register with Typer
            return self.app.command(*args, **kwargs)(wrapped)

        return decorator

    def run_with_config(self, func: Callable):
        """Run a single function with config support"""
        wrapped = self.add_config_options(func, is_command=False)
        typer.run(wrapped)

    def run(self):
        """Run the app with subcommands"""
        self.app()

def dictconfig_to_dataclass(config: DictConfig, target_class: Type[T]) -> T:
    """
    Convert an OmegaConf DictConfig to a nested dataclass structure.

    Args:
        config: OmegaConf DictConfig object
        target_class: The target dataclass type

    Returns:
        An instance of the target dataclass
    """
    if not is_dataclass(target_class):
        raise ValueError(f"{target_class.__name__} is not a dataclass")

    # Convert DictConfig to regular dict
    config_dict = OmegaConf.to_container(config, resolve=True)
    if config_dict is None:
        raise ValueError("Failed to convert DictConfig to dict")

    # Process each field in the dataclass
    init_kwargs = {}
    for field in fields(target_class):
        field_value = config_dict.get(field.name)
        if field_value is None:
            continue

        # Handle nested dataclass
        if is_dataclass(field.type):
            if isinstance(field_value, dict):
                init_kwargs[field.name] = dictconfig_to_dataclass(
                    OmegaConf.create(field_value),
                    field.type
                )
        # Handle list of dataclasses
        elif hasattr(field.type, "__origin__") and field.type.__origin__ is list:
            if len(field.type.__args__) > 0 and is_dataclass(field.type.__args__[0]):
                nested_type = field.type.__args__[0]
                if isinstance(field_value, list):
                    init_kwargs[field.name] = [
                        dictconfig_to_dataclass(OmegaConf.create(item), nested_type)
                        if isinstance(item, dict)
                        else item
                        for item in field_value
                    ]
                else:
                    init_kwargs[field.name] = field_value
        else:
            init_kwargs[field.name] = field_value

    return target_class(**init_kwargs)


def update_dataclass_from_config(dc_instance: Any, config: DictConfig) -> None:
    """
    Recursively updates a dataclass instance with values from a DictConfig.

    Args:
        dc_instance: The dataclass instance to update
        config: DictConfig containing the update values

    Example:
        @dataclass
        class NestedConfig:
            value: int

        @dataclass
        class MainConfig:
            name: str
            nested: NestedConfig

        config = OmegaConf.create({
            "name": "new_name",
            "nested": {"value": 42}
        })

        instance = MainConfig(name="old_name", nested=NestedConfig(value=0))
        update_dataclass_from_config(instance, config)
    """
    if not is_dataclass(dc_instance):
        raise ValueError("First argument must be a dataclass instance")

    if not isinstance(config, DictConfig):
        raise ValueError("Second argument must be a DictConfig")

    # Get all fields of the dataclass
    dc_fields = {field.name: field for field in fields(dc_instance)}

    # Iterate through the config keys
    for key, value in config.items():
        if key not in dc_fields:
            continue

        if isinstance(value, DictConfig):
            # If the value is a nested DictConfig and the corresponding
            # dataclass field is also a dataclass, recurse
            current_value = getattr(dc_instance, key)
            if is_dataclass(current_value):
                update_dataclass_from_config(current_value, value)
        else:
            # Update the simple value
            setattr(dc_instance, key, value)



# Convenience function
def create_configurable_app(
        base_config_class: Type[Any],
        config_store_name: str = "base_config",
        set_global_config_fn: Optional[Callable[[DictConfig], None]]=None
) -> ConfigurableApp:
    return ConfigurableApp(base_config_class, config_store_name, set_global_config_fn=set_global_config_fn)



