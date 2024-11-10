# config_manager.py
import os.path
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
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

    def __init__(self, base_config_class: Type[Any], config_store_name: str = "base_config"):
        self.base_config_class = base_config_class
        self.config_store_name = config_store_name
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

    def add_config_options(self, func: Callable) -> Callable:
        """Add configuration options to a function"""

        def run_command(*args, **kwargs):
            # Extract config-related parameters
            config_file = kwargs.pop('config_file', None)
            config_args = kwargs.pop('config_args', [])

            config_merge_preference = kwargs.pop('config_merge_preference', None)

            # Process the configuration
            cfg = self.process_config(config_file, config_args, config_merge_preference)

            # Call the original function
            return func(*args, config=cfg, **kwargs)

        # Get original signature and parameters
        sig = inspect.signature(func)
        orig_params = []
        #We need to make sure that only typer.Option are provided by the user, as typer argument is used for config
        for name, p in sig.parameters.items():
            # print(name, p)
            if p.name != 'config':
                if not all([isinstance(x, typer.models.OptionInfo) for x in p.annotation.__metadata__]):
                    raise typer.BadParameter(f"Paramenter {name} was not provided as typer.Option. typer.Argument is not supported")
                orig_params.append(p)
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
                annotation=Annotated[Optional[List[str]], typer.Argument(help="Hydra-style config modifications ")]
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
    ) -> DictConfig:
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

            cfg = base_cfg
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

            return cfg

    def command(self, *args, **kwargs):
        """Decorator for adding commands with config support"""

        def decorator(func: Callable):
            # Add config options to the function
            wrapped = self.add_config_options(func)
            # Register with Typer
            return self.app.command(*args, **kwargs)(wrapped)

        return decorator

    def run_with_config(self, func: Callable):
        """Run a single function with config support"""
        wrapped = self.add_config_options(func)
        typer.run(wrapped)

    def run(self):
        """Run the app with subcommands"""
        self.app()


# Convenience function
def create_configurable_app(
        base_config_class: Type[Any],
        config_store_name: str = "base_config"
) -> ConfigurableApp:
    return ConfigurableApp(base_config_class, config_store_name)