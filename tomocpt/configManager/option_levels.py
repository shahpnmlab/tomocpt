"""Option levels of the config tree (Gate 10, human decision 22 Nov 2026).

Every setting of ``MainConfig`` under ``shared``, ``prepData``, ``train`` and ``infer`` has a level:

* ``basic``: the everyday command-line flags of ``cli_options.CLI_FLAGS`` plus required fields
  (no default). ``CLI_FLAGS`` is the single source of this set.
* ``advanced``: every other setting that changes what is trained or predicted (the default).
* ``diagnostic``: settings that only produce extra output for inspection (``DIAGNOSTIC``); they
  never change training or the picks.
* ``internal``: fields that are not user settings (``INTERNAL_FIELDS``: unused or fixed values),
  and the copies of the ``shared`` fields in the other sections (set them under ``shared``).
  ``tomocpt init`` does not write them and ``docs/options.md`` leaves them out.

``tomocpt init --level <level>`` writes every field up to that level (``initializer``);
``python -m tomocpt.configManager.option_docs`` writes the reference ``docs/options.md``.
"""
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple, Union, get_args, get_origin, get_type_hints

import typer
from omegaconf import MISSING

LEVELS = ("basic", "advanced", "diagnostic")
INTERNAL = "internal"
SECTIONS = ("shared", "prepData", "train", "infer")

#: Settings that only add output for inspection (dotted, relative to the section).
DIAGNOSTIC: Dict[str, Tuple[str, ...]] = {
    "prepData": (
        "write_filament_masks",  # binary tube masks next to the labels, never used as targets
    ),
    "train": (
        "snapshot_chunks",  # chunk predictions saved after every validation (train dashboard)
        "cv_class_weights",  # class weights of the mean F-beta in the CV report only
    ),
    "infer": (),
}

#: Fields that are not user settings: unused, or fixed by the code (dotted, relative to the section).
INTERNAL_FIELDS: Dict[str, Tuple[str, ...]] = {
    "prepData": (
        "USE_CUDA_FOR_DATA",  # unused
    ),
    "train": (
        "optimizer._target_",  # RAdam is the tested optimizer; schedule_free requires it
        "network.NOISE2NOISE_RAND_MASK_SIZE",  # unused
        "network.SwinUNETR.in_channels",  # one tomogram
        "network.SwinUNETR.out_channels",  # pipeline v2 sets it from the class table
        "network.UNET.in_channels",  # one tomogram
        "N_GPUS",  # unused (every visible GPU is used)
        "USE_CUDA_FOR_DATA",  # unused
        "CHUNK_SIZE",  # 64 throughout (network input, Fourier augmentation, prediction patches)
        "RANDOM_FRACTION_TO_SAMPLE_TRAIN",  # unused
    ),
    "infer": (
        "N_CPUS_IF_NO_GPU",  # unused by predict
    ),
}


@dataclass(frozen=True)
class Option:
    """One setting: ``section.path`` as used in the YAML and in ``key=value`` overrides."""

    section: str
    path: str
    level: str
    type: Any
    default: Any
    help: str

    @property
    def key(self) -> str:
        return f"{self.section}.{self.path}"


@lru_cache(maxsize=1)
def _defaults():
    from tomocpt.mainConfig import MainConfig

    return MainConfig()


@lru_cache(maxsize=1)
def _shared_names() -> frozenset:
    return frozenset(f.name for f in fields(type(_defaults().shared)))


def _resolve(section: str, dotted: str):
    """(owner dataclass instance, its ``dataclasses.Field``) of a leaf; ``KeyError`` otherwise."""
    if section not in SECTIONS:
        raise KeyError(f"Unknown config section '{section}' (one of {SECTIONS})")
    obj = getattr(_defaults(), section)
    *head, last = dotted.split(".")
    for part in head:
        names = {f.name for f in fields(type(obj))}
        if part not in names or not is_dataclass(getattr(obj, part)):
            raise KeyError(f"Unknown option '{section}.{dotted}'")
        obj = getattr(obj, part)
    for f in fields(type(obj)):
        if f.name == last:
            if is_dataclass(getattr(obj, last)):
                raise KeyError(f"'{section}.{dotted}' is a sub-section, not an option")
            return obj, f
    raise KeyError(f"Unknown option '{section}.{dotted}'")


def _option_info(owner: Any, name: str):
    hint = get_type_hints(type(owner), include_extras=True).get(name)
    if hint is None or get_origin(hint) is not Annotated:
        return None
    return next((m for m in get_args(hint)[1:] if isinstance(m, typer.models.OptionInfo)), None)


def unwrap_type(tp: Any) -> Any:
    """The type inside ``Annotated[...]``."""
    while get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    return tp


def option_level(section: str, dotted_field: str) -> str:
    """Level of ``section.dotted_field``: one of ``LEVELS`` or ``"internal"``.

    Raises ``KeyError`` for an unknown section or field, or for a sub-section."""
    from tomocpt.configManager.cli_options import CLI_FLAGS

    owner, f = _resolve(section, dotted_field)
    if section == "shared":
        return "basic"
    if "." not in dotted_field and dotted_field in _shared_names():
        return INTERNAL  # copied from `shared` (MainConfig.__post_init__); set it under `shared`
    if dotted_field in INTERNAL_FIELDS.get(section, ()):
        return INTERNAL
    if dotted_field in CLI_FLAGS.get(section, ()) or getattr(owner, f.name) is MISSING:
        return "basic"
    if dotted_field in DIAGNOSTIC.get(section, ()):
        return "diagnostic"
    return "advanced"


def option_help(section: str, dotted_field: str) -> str:
    """Help text of ``section.dotted_field`` (empty when the field has none)."""
    owner, f = _resolve(section, dotted_field)
    info = _option_info(owner, f.name)
    return (info.help or "") if info is not None else ""


def subsection_help(owner: Any, name: str) -> str:
    """Help text of the sub-section field ``name`` of the dataclass instance ``owner`` (may be empty)."""
    info = _option_info(owner, name)
    return (info.help or "") if info is not None else ""


def level_rank(level: str) -> int:
    """0, 1, 2 for the levels; 3 for ``internal``."""
    return LEVELS.index(level) if level in LEVELS else len(LEVELS)


def iter_options(section: str) -> List[Option]:
    """Every leaf of ``section`` in dataclass order (internal fields included)."""
    out: List[Option] = []

    def walk(obj, prefix):
        for f in fields(type(obj)):
            v = getattr(obj, f.name)
            if is_dataclass(v):
                walk(v, f"{prefix}{f.name}.")
                continue
            path = prefix + f.name
            out.append(Option(section, path, option_level(section, path), unwrap_type(f.type), v,
                              option_help(section, path)))

    walk(getattr(_defaults(), section), "")
    return out


def describe_type(tp: Any) -> str:
    """Short human-readable type: ``float``, ``optional path``, ``one of: star, imod``, ..."""
    tp = unwrap_type(tp)
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        inner = " or ".join(describe_type(a) for a in args)
        return f"optional {inner}" if len(args) < len(get_args(tp)) else inner
    if isinstance(tp, type) and issubclass(tp, Enum):
        return "one of: " + ", ".join(str(m.value) for m in tp)
    if isinstance(tp, type) and issubclass(tp, Path):
        return "path"
    if origin in (tuple, Tuple):
        args = get_args(tp)
        if args and args[-1] is not Ellipsis:
            return f"{len(args)} x {describe_type(args[0])}" if len(set(args)) == 1 else \
                "(" + ", ".join(describe_type(a) for a in args) + ")"
        return "tuple"
    if origin in (list, List):
        return "list"
    if tp is Any:
        return "any"
    return getattr(tp, "__name__", str(tp))
