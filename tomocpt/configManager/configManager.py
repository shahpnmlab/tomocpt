import copy
import sys
import warnings
from dataclasses import is_dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Tuple, Set, get_args, get_origin
from typing_extensions import Annotated
from pathlib import Path
import inspect
import typer
from omegaconf import OmegaConf, DictConfig, MISSING, MissingMandatoryValue
from enum import Enum
from functools import wraps


def _normalize_path(path: str) -> str:
    """Normalize parameter path to dot notation"""
    return path.replace("__", ".")


typer.main.get_command_name = lambda name: _normalize_path(name)  # .replace("_", "-")


class ConfigurationError(Exception):
    """Raised when there are conflicts in configuration settings"""

    pass


class MergePreference(str, Enum):
    COMMAND = "command"
    CONFIG_FILE = "configFile"


# --------------------------------------------------------------------------------------
# typed values for key=value overrides and YAML (slim CLI, 9 Nov 2026)
# --------------------------------------------------------------------------------------

_TRUE = {"true", "1", "yes", "y", "on"}
_FALSE = {"false", "0", "no", "n", "off"}
_NONE = {"none", "null"}


def _unwrap_annotated(tp: Any) -> Any:
    while get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    return tp


def parse_value(field_type: Any, value: Any, key: str = "") -> Any:
    """``value`` (a command-line string or a YAML value) converted to ``field_type``.

    Booleans accept true/false/1/0/yes/no/on/off; ``none``/``null`` clear an Optional field; Enums
    take their value; Paths, tuples (``a,b`` or a list), ints (``3``, not ``3.5``) and floats as
    expected. Raises ``ConfigurationError`` when the value does not fit."""
    tp = _unwrap_annotated(field_type)
    origin = get_origin(tp)

    def fail(why: str = ""):
        raise ConfigurationError(f"Cannot use {value!r} for '{key}' (expected {tp}){': ' + why if why else ''}")

    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if value is None or (isinstance(value, str) and value.strip().lower() in _NONE):
            if len(args) < len(get_args(tp)):
                return None
            fail("not optional")
        errors = []
        for a in args:
            try:
                return parse_value(a, value, key)
            except ConfigurationError as e:
                errors.append(str(e))
        fail("; ".join(errors))
    if tp is Any or tp is None:
        return value
    if tp is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        fail()
    if isinstance(tp, type) and issubclass(tp, Enum):
        if isinstance(value, tp):
            return value
        for member in tp:
            if str(member.value) == str(value) or member.name == str(value):
                return member
        fail(f"one of {[m.value for m in tp]}")
    if isinstance(tp, type) and issubclass(tp, Path):
        return value if isinstance(value, Path) else Path(str(value))
    if origin in (tuple, Tuple):
        items = list(value) if isinstance(value, (list, tuple)) else \
            [v for v in str(value).strip("()[]{}").split(",") if v.strip() != ""]
        types = get_args(tp)
        if len(types) == 2 and types[1] is Ellipsis:
            types = (types[0],) * len(items)
        if types and len(types) != len(items):
            fail(f"{len(types)} items")
        return tuple(parse_value(t, v.strip() if isinstance(v, str) else v, key) for t, v in zip(types, items)) \
            if types else tuple(items)
    if origin in (list, List):
        items = list(value) if isinstance(value, (list, tuple)) else [v for v in str(value).split(",") if v.strip()]
        (t,) = get_args(tp) or (Any,)
        return [parse_value(t, v.strip() if isinstance(v, str) else v, key) for v in items]
    if tp is int:
        if isinstance(value, bool):
            fail()
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        try:
            return int(str(value).strip())
        except ValueError:
            fail()
    if tp is float:
        if isinstance(value, bool):
            fail()
        try:
            return float(value)
        except (TypeError, ValueError):
            fail()
    if tp is str:
        return value if isinstance(value, str) else str(value)
    if isinstance(tp, type) and isinstance(value, tp):
        return value
    try:
        return tp(value)
    except Exception:
        fail()


def field_type_of(obj: Any, name: str) -> Any:
    for f in fields(type(obj)):
        if f.name == name:
            return f.type
    raise ConfigurationError(f"Invalid parameter: {name}")


def coerce_config_types(config: Any, prefix: str = "") -> None:
    """Convert every field value of ``config`` (recursively) to its declared type in place. YAML
    values arrive raw (``"32"`` for an Enum, ``"flips"``); a value that does not fit is left as it
    is with a warning (the run fails later where it is used, as before)."""
    if not is_dataclass(config):
        return
    for f in fields(type(config)):
        v = getattr(config, f.name, MISSING)
        if is_dataclass(v):
            coerce_config_types(v, f"{prefix}{f.name}.")
            continue
        if v is MISSING or isinstance(v, str) and v == "???":
            continue
        try:
            new = parse_value(f.type, v, prefix + f.name)
        except ConfigurationError as e:
            if v is not None:
                warnings.warn(str(e))
            continue
        if new is not v:
            setattr(config, f.name, new)


class ConfigManager:
    def __init__(self):
        self.registered_configs = {}

    def _get_default_value(self, config_obj: Any, path: str) -> Any:
        """Get the default value for a given parameter path"""
        parts = path.split(".")
        curr_obj = config_obj
        try:
            for part in parts[:-1]:
                curr_obj = getattr(curr_obj, part)
            return getattr(curr_obj, parts[-1])
        except AttributeError:
            return None

    def _get_valid_paths(self, config_obj: Any, prefix: str = "") -> Set[str]:
        paths = set()
        for field in fields(type(config_obj)):
            field_value = getattr(config_obj, field.name)
            current_path = f"{prefix}.{field.name}" if prefix else field.name
            paths.add(current_path)

            if is_dataclass(type(field_value)):
                paths.update(self._get_valid_paths(field_value, current_path))
        return paths

    def _process_nested_dataclass(
        self, obj: Any, prefix: str = ""
    ) -> List[Tuple[str, Any, typer.models.OptionInfo, Any]]:
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
                    new_prefix = (
                        f"{prefix}{field.name}__" if prefix else f"{field.name}__"
                    )
                    nested_params = self._process_nested_dataclass(
                        field_value, new_prefix
                    )
                    params.extend(nested_params)
                elif any(
                    isinstance(m, typer.models.OptionInfo)
                    for m in field.type.__metadata__
                ):
                    for metadata in field.type.__metadata__:
                        if isinstance(metadata, typer.models.OptionInfo):
                            if field_value is MISSING:
                                new_option = typer.Option(
                                    default=...,
                                    help=metadata.help,
                                    **{
                                        k: v
                                        for k, v in metadata.__dict__.items()
                                        if k not in ["default", "help", "param_decls"]
                                    },
                                )
                                params.append(
                                    (current_path, field_type, new_option, MISSING)
                                )
                            else:
                                params.append(
                                    (current_path, field_type, metadata, field_value)
                                )
            elif is_dataclass(type(field_value)):
                # Add nested paths for hydra-style access without creating CLI params
                new_prefix = f"{prefix}{field.name}__" if prefix else f"{field.name}__"
                nested_params = self._process_nested_dataclass(field_value, new_prefix)
                params.extend(nested_params)

        return params

    def _get_annotated_options(
        self, config_obj: Any
    ) -> Tuple[List[tuple], List[tuple], Set[str]]:
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

    def create_cli_parameters(
        self, config_obj: Any, flags: Optional[Set[str]] = None
    ) -> Tuple[List[inspect.Parameter], Set[str]]:
        """``flags``: dotted field names that become ``--flags`` (required fields always do);
        ``None`` = every annotated field (the pre-9 Nov 2026 behaviour)."""
        if config_obj is None:
            return [], set([])
        required_params, optional_params, valid_paths = self._get_annotated_options(
            config_obj
        )
        if flags is not None:
            optional_params = [p for p in optional_params if _normalize_path(p[0]) in flags]

        # Create required parameters (without defaults)
        required_parameters = [
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,  # Make required params positional
                annotation=Annotated[type_, option],
            )
            for name, type_, option in required_params
        ]

        # Create optional parameters (with defaults)
        optional_parameters = [
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,  # Keep optional params as keyword-only
                annotation=Annotated[type_, option],
                default=default,
            )
            for name, type_, option, default in optional_params
        ]

        return [*required_parameters, *optional_parameters], valid_paths


class ConfigurableApp:
    def __init__(self):
        self.app = typer.Typer(
            pretty_exceptions_show_locals=False,
            pretty_exceptions_enable=False,
            pretty_exceptions_short=True,
            no_args_is_help=True,
            add_completion=False,
        )
        self.config_manager = ConfigManager()
        #: command name -> (section config, its everyday flags or None)
        self._sections: Dict[str, Tuple[Any, Optional[Set[str]]]] = {}

    def _get_typed_value(self, key: str, value: str, curr_obj: Any, attr: str) -> Any:
        """Properly typed value of ``key=value`` for field ``attr`` of ``curr_obj``."""
        return parse_value(field_type_of(curr_obj, attr), value, key)

    def command(
        self,
        *typer_args,
        mainConfig: Any = None,
        config_key: str,
        create_cli_args=True,
        **typer_kwargs,
    ):
        def decorator(func: Callable):
            # Get mainConfig file value if present
            config_file_path = None
            for i, arg in enumerate(sys.argv):
                if arg == "--config_file":
                    if i + 1 < len(sys.argv):
                        config_file_path = sys.argv[i + 1]
                elif arg.startswith("--config_file="):
                    config_file_path = arg.split("=", 1)[1]

            # Load mainConfig file if present
            yaml_config = None
            config_updated_with_yaml = None
            if config_file_path:
                yaml_config = OmegaConf.load(config_file_path)
                if mainConfig is not None:
                    config_updated_with_yaml, unmatched_yaml = merge_config(
                        yaml_config, mainConfig
                    )
                    # Propagate values from shared config to other configs
                    if hasattr(config_updated_with_yaml, "shared"):
                        shared_conf = getattr(config_updated_with_yaml, "shared")
                        for field_info in fields(config_updated_with_yaml):
                            if field_info.name == "shared":
                                continue
                            specific_conf = getattr(
                                config_updated_with_yaml, field_info.name
                            )
                            if not is_dataclass(specific_conf):
                                continue
                            for shared_field in fields(shared_conf):
                                if hasattr(specific_conf, shared_field.name):
                                    spec_val = getattr(
                                        specific_conf, shared_field.name
                                    )
                                    shared_val = getattr(shared_conf, shared_field.name)
                                    if spec_val is None and shared_val is not None:
                                        setattr(
                                            specific_conf, shared_field.name, shared_val
                                        )

            # Select only the configuration that is relevant for the command
            if config_key is not None:
                config = getattr(mainConfig, config_key)
            else:
                config = mainConfig
            original_sig = inspect.signature(func)
            func_params = [
                p for p in original_sig.parameters.values() if p.name != "config"
            ]
            flags = None
            if create_cli_args and config_key is not None:
                from tomocpt.configManager.cli_options import CLI_FLAGS

                flags = set(CLI_FLAGS[config_key]) if config_key in CLI_FLAGS else None
            if create_cli_args:
                cli_params, paths = self.config_manager.create_cli_parameters(config, flags)
            else:
                cli_params, paths = [], []
            self._sections[typer_kwargs.get("name") or func.__name__] = (config, flags)

            # Create parameter mappings
            config_param_map = {param.name: param for param in cli_params}
            config_param_types = {
                param.name: (
                    param.annotation.__args__[0]
                    if str(param.annotation).startswith("typing.Annotated")
                    else param.annotation
                )
                for param in cli_params
            }
            # Validate function parameters
            for func_param in func_params:
                if func_param.name in config_param_map:
                    config_type = config_param_types[func_param.name]
                    func_type = (
                        func_param.annotation
                        if func_param.annotation != inspect.Parameter.empty
                        else type(None)
                    )
                    if func_type != config_type:
                        raise ConfigurationError(
                            f"Type mismatch for '{func_param.name}': function expects {func_type}, config provides {config_type}"
                        )

                    config_param = config_param_map[func_param.name]
                    config_default = config_param.default
                    func_default = func_param.default

                    if (
                        func_default != inspect.Parameter.empty
                        and config_default != inspect.Parameter.empty
                    ):
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
                    "config_file",
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[
                        Optional[Path], typer.Option(help="Config file")
                    ],
                    default=None,
                ),
                inspect.Parameter(
                    "config_merge_preference",
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[Optional[MergePreference], typer.Option()],
                    default=None,
                ),
                inspect.Parameter(
                    "config_args",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=Annotated[
                        Optional[List[str]],
                        typer.Argument(help="Hydra-style config modifications"),
                    ],
                ),
            ]

            # Update cli_params with values from config file if present
            if yaml_config:
                updated_cli_params = []
                for param in cli_params:
                    param_path = _normalize_path(param.name).split(".")
                    curr_config = config_updated_with_yaml
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
                            **kwargs,
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
                    (
                        required_params
                        if config_param.default == inspect.Parameter.empty
                        else optional_params
                    ).append(config_param)
                    processed_params.add(param.name)
                else:
                    (
                        required_params
                        if param.default == inspect.Parameter.empty
                        and param.name != "kwargs"
                        else optional_params
                    ).append(param)

            for param in cli_params:
                if param.name not in processed_params:
                    (
                        required_params
                        if param.default == inspect.Parameter.empty
                        else optional_params
                    ).append(param)

            optional_params.extend(extra_params)

            @wraps(func)
            def wrapped(
                *args,
                config_args: Optional[List[str]] = None,
                config_file: Optional[Path] = None,
                config_merge_preference: Optional[MergePreference] = None,
                **kwargs,
            ):
                cli_params = {}
                for k, v in kwargs.items():
                    if k in config_param_map:
                        cli_params[k] = v
                        path = k.split("__")
                        curr_obj = config
                        for p in path[:-1]:
                            curr_obj = getattr(curr_obj, p)
                        setattr(curr_obj, path[-1], v)

                # YAML values of fields without a flag arrive raw: give them their types
                coerce_config_types(config)

                if config_args:
                    for arg in config_args:
                        if "=" not in arg:
                            raise typer.BadParameter(f"Invalid arg {arg}. Missing '='?")
                        key, value = arg.split("=", 1)
                        path = key.split(".")
                        curr_obj = config

                        for p in path[:-1]:
                            if not hasattr(curr_obj, p) or not is_dataclass(getattr(curr_obj, p)):
                                raise ConfigurationError(f"Invalid path: {key}")
                            curr_obj = getattr(curr_obj, p)

                        if not is_dataclass(curr_obj) or path[-1] not in {f.name for f in fields(type(curr_obj))}:
                            raise ConfigurationError(f"Invalid parameter: {key}")
                        setattr(curr_obj, path[-1], self._get_typed_value(key, value, curr_obj, path[-1]))

                # if config_file:
                #     yaml_config = OmegaConf.load(config_file)
                #     self._update_from_yaml(config, yaml_config, config_merge_preference)

                func_kwargs = {
                    k: v for k, v in kwargs.items() if k not in config_param_map
                }
                if "config" in original_sig.parameters:
                    func_kwargs["config"] = config

                # Checkpoint loads compare against this snapshot of the YAML+CLI config, not the
                # live global that each load updates (Gate 3c). Scoped to the command.
                previous_snapshot = get_cli_snapshot()
                if mainConfig is not None:
                    set_cli_snapshot(mainConfig)
                try:
                    return func(*args, **func_kwargs)
                finally:
                    _restore_cli_snapshot(previous_snapshot)

            wrapped.__signature__ = inspect.Signature(
                parameters=[*required_params, *optional_params]
            )
            if flags is not None and "epilog" not in typer_kwargs:
                from tomocpt.configManager.cli_options import HELP_EPILOG

                typer_kwargs["epilog"] = HELP_EPILOG
            return self.app.command(*typer_args, **typer_kwargs)(wrapped)

        return decorator

    def run(self, args: Optional[List[str]] = None):
        argv = list(sys.argv[1:] if args is None else args)
        self.app(args=self._rewrite_old_flags(argv))

    def _rewrite_old_flags(self, argv: List[str]) -> List[str]:
        """Long flags of settings that are no longer flags (slim CLI) become ``key=value``
        overrides, with a deprecation warning, so existing scripts keep working."""
        cmd_index = next((i for i, a in enumerate(argv) if not a.startswith("-")), None)
        if cmd_index is None or argv[cmd_index] not in self._sections:
            return argv
        config, flags = self._sections[argv[cmd_index]]
        if flags is None or config is None:
            return argv
        keep = {"help", "config_file", "config_merge_preference"} | set(flags)
        out, overrides = argv[:cmd_index + 1], []
        i = cmd_index + 1
        while i < len(argv):
            tok = argv[i]
            if tok == "--":
                out.extend(argv[i:])
                break
            if not tok.startswith("--"):
                out.append(tok)
                i += 1
                continue
            name, eq, inline = tok[2:].partition("=")
            negated = name.startswith("no-")
            base = name[3:] if negated else name
            target = self._field_target(config, base)
            if base in keep or target is None:
                out.append(tok)
                i += 1
                continue
            obj, attr = target
            is_bool = _unwrap_annotated(field_type_of(obj, attr)) is bool
            if negated:
                value = "false"
            elif eq:
                value = inline
            elif is_bool:
                value = "true"
            elif i + 1 < len(argv):
                i += 1
                value = argv[i]
            else:
                out.append(tok)
                i += 1
                continue
            overrides.append(f"{base}={value}")
            msg = f"--{name} is no longer a command-line flag; use {base}={value} (or the YAML config)"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            print(f"Warning: {msg}", file=sys.stderr)
            i += 1
        return out + overrides

    @staticmethod
    def _field_target(config: Any, dotted: str):
        obj = config
        parts = dotted.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p, None)
            if not is_dataclass(obj):
                return None
        if not is_dataclass(obj) or parts[-1] not in {f.name for f in fields(type(obj))}:
            return None
        if is_dataclass(getattr(obj, parts[-1], None)):
            return None
        return obj, parts[-1]

    def register_command(
        self, func, mainConfig, config_key, create_cli_args=True
    ):  # TODO: can we have this as a class method in app?
        @self.command(
            mainConfig=mainConfig,
            config_key=config_key,
            create_cli_args=create_cli_args,
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


def merge_config(yaml_config: DictConfig, config: Any) -> Tuple[Any, Dict]:
    """
    Merge YAML config into a dataclass config object and return unmatched sections.
    """
    if not is_dataclass(config):
        return config, {}

    # Track used keys to identify unmatched sections
    used_keys = set()

    # For each field in the dataclass
    for field in fields(config):
        field_name = field.name
        current_value = getattr(config, field_name)

        # Find the field in yaml_config, checking both top level and nested
        yaml_value = None
        yaml_key = None

        try:
            if field_name in yaml_config:
                yaml_value = yaml_config[field_name]
                yaml_key = field_name
            else:
                # Search in all top-level dictionaries
                for key, section in yaml_config.items():
                    if (
                        isinstance(section, dict) and field_name in section
                    ):  # Changed DictConfig to dict
                        yaml_value = section[field_name]
                        yaml_key = key
                        break

            if yaml_value is not None:
                used_keys.add(yaml_key)
                if is_dataclass(current_value) and isinstance(
                    yaml_value, (dict, DictConfig)
                ):  # Accept both dict and DictConfig
                    # Convert yaml_value to regular dict if it's a DictConfig
                    if isinstance(yaml_value, DictConfig):
                        yaml_value = OmegaConf.to_container(yaml_value, resolve=True)
                    # Recursively merge nested dataclass
                    merged_value, unmatched_nested = merge_config(
                        yaml_value, current_value
                    )
                    setattr(config, field_name, merged_value)
                    # If there are unmatched nested sections, add them to used_keys
                    if unmatched_nested:
                        used_keys.add(yaml_key)
                else:
                    # Direct update for non-dataclass fields
                    setattr(config, field_name, yaml_value)
        except MissingMandatoryValue:
            # Skip fields with missing mandatory values
            continue

    # Collect unmatched sections
    unmatched = {}
    for key, value in yaml_config.items():
        if key not in used_keys:
            if isinstance(value, DictConfig):
                value = OmegaConf.to_container(value, resolve=True)
            unmatched[key] = value

    return config, unmatched


_ABSENT = object()

_cli_snapshot: Optional[Any] = None


def set_cli_snapshot(config: Any) -> None:
    """Store a deep copy of the config as set by YAML and the command line (Gate 3c)."""
    global _cli_snapshot
    _cli_snapshot = copy.deepcopy(config)


def get_cli_snapshot() -> Optional[Any]:
    """The config as set by YAML and the command line, or ``None`` outside a CLI command."""
    return _cli_snapshot


def clear_cli_snapshot() -> None:
    global _cli_snapshot
    _cli_snapshot = None


def _restore_cli_snapshot(snapshot: Optional[Any]) -> None:
    global _cli_snapshot
    _cli_snapshot = snapshot


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """Return ``getattr(obj, name)``, or ``default`` if the attribute is absent.

    Use it to read config objects unpickled from checkpoints that predate a field
    (see ``backfill_missing_fields``). For a dataclass field, only the instance's
    own value counts: a field absent from the instance returns ``default``, not the
    class-level default that plain ``getattr`` would fall back to.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        instance_dict = getattr(obj, "__dict__", None)
        if instance_dict is not None and name in {f.name for f in fields(obj)}:
            return instance_dict.get(name, default)
    return getattr(obj, name, default)


def backfill_missing_fields(target: Any, defaults: Any) -> Any:
    """Add fields present on ``defaults`` but absent from ``target``, recursively.

    Checkpoints pickle the whole config object. When a config field is added
    later, unpickling an older checkpoint produces an instance of the *current*
    dataclass that is simply missing that attribute - dataclass unpickling
    restores ``__dict__`` and never runs ``__init__``, so no default is applied.
    Anything that then walks the field list with a bare ``getattr`` raises
    ``AttributeError``.

    A checkpoint written before a field existed has no opinion about it, so the
    current default is the correct value to fill in. This keeps old checkpoints
    loadable as config grows, rather than only fixing the one field that
    happened to be added most recently.

    Parameters
    ----------
    target : Any
        Dataclass instance to backfill, modified in place.
    defaults : Any
        Dataclass instance of the same type supplying default values.

    Returns
    -------
    Any
        ``target``, for convenience.
    """
    if not (is_dataclass(target) and is_dataclass(defaults)):
        return target

    for field in fields(type(defaults)):
        default_value = getattr(defaults, field.name, None)
        # Check the instance, not ``hasattr``: a field with a plain default also exists
        # as a class attribute, so ``hasattr`` is true even when the instance lacks it.
        existing = safe_getattr(target, field.name, _ABSENT)
        if existing is _ABSENT:
            setattr(target, field.name, copy.deepcopy(default_value))
        elif is_dataclass(existing) and is_dataclass(default_value):
            backfill_missing_fields(existing, default_value)
    return target


def update_config(target: Any, source: Any):
    """
    Recursively updates the fields of a target dataclass with values from the source dataclass.
    Handles omegaconf.MISSING by keeping the non-MISSING value.
    """
    if not (is_dataclass(target) and is_dataclass(source)):
        raise ValueError(
            "Both target and source must be dataclass instances of the same type."
        )

    for field in fields(target):
        field_name = field.name
        target_value = getattr(target, field_name)
        source_value = safe_getattr(source, field_name, _ABSENT)
        if source_value is _ABSENT:
            # Source predates this field (e.g. an older checkpoint's config) and
            # so has no opinion on it. Keep the target's value.
            continue

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


def compare_dataclasses(
    obj1: Any, obj2: Any, path: str = ""
) -> List[Tuple[str, Any, Any]]:
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
        value1 = safe_getattr(obj1, field_name, _ABSENT)
        value2 = safe_getattr(obj2, field_name, _ABSENT)
        if value1 is _ABSENT and value2 is _ABSENT:
            continue

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
                        nested_diffs = compare_dataclasses(
                            item1, item2, f"{current_path}[{i}]"
                        )
                        differences.extend(nested_diffs)
                    elif item1 != item2:
                        differences.append((f"{current_path}[{i}]", item1, item2))
        # For basic types, direct comparison
        elif value1 != value2:
            differences.append((current_path, value1, value2))

    return differences


def update_config_with_changed_values(
    target: Any, originaConfig: Any, configAfterCli: Any
) -> Any:
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
        if new_value is _ABSENT:
            # The field is absent from configAfterCli (an older config): nothing to apply.
            continue
        # Split the path into parts
        parts = path.split(".")

        # Handle array indices
        current_obj = result
        for i, part in enumerate(parts[:-1]):  # All parts except the last one
            # Check if this part contains an array index
            if "[" in part:
                # Split into name and index
                name, idx = part.split("[")
                idx = int(idx.rstrip("]"))

                # Get the array/list
                if name:  # If there's a name before the index
                    current_obj = safe_getattr(current_obj, name, _ABSENT)
                    if current_obj is _ABSENT:
                        break
                current_obj = current_obj[idx]
            else:
                current_obj = safe_getattr(current_obj, part, _ABSENT)
                if current_obj is _ABSENT:
                    break
        if current_obj is _ABSENT:
            # The target predates this section: there is no object to update.
            continue

        # Handle the final part
        last_part = parts[-1]
        if "[" in last_part:
            name, idx = last_part.split("[")
            idx = int(idx.rstrip("]"))
            if name:
                current_obj = safe_getattr(current_obj, name, _ABSENT)
                if current_obj is _ABSENT:
                    continue
            current_obj[idx] = new_value
        else:
            setattr(current_obj, last_part, new_value)

    return result


def create_app() -> ConfigurableApp:
    return ConfigurableApp()
