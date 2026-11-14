"""Shared helpers of the Gate 8 performance harness: config loading, the global ``mainConfig``
context, provenance (commit, config hash, environment)."""
import contextlib
import copy
import hashlib
import json
import subprocess
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

REPO = Path(__file__).resolve().parents[2]

#: Synthetic class diameters (Å), cycled for C classes: the CZII set first.
SYNTHETIC_DIAMETERS = (130.0, 180.0, 260.0, 300.0, 120.0, 270.0)


def _assign_inplace(target, source):
    """Copy every field of ``source`` onto ``target`` recursively, keeping nested object identity
    (legacy modules hold references to the nested sections of the global config)."""
    for f in fields(type(source)):
        if not hasattr(source, f.name):
            continue
        sv = getattr(source, f.name)
        tv = getattr(target, f.name, None)
        if is_dataclass(sv) and is_dataclass(tv) and type(sv) is type(tv):
            _assign_inplace(tv, sv)
        else:
            setattr(target, f.name, copy.deepcopy(sv))


@contextlib.contextmanager
def applied_config(cfg):
    """``cfg`` applied to the global ``mainConfig`` (and ``tomocpt.constants`` restored) for the block."""
    from tomocpt import constants
    from tomocpt.mainConfig import mainConfig

    saved = copy.deepcopy(mainConfig)
    saved_constants = {k: copy.deepcopy(v) for k, v in vars(constants).items() if not k.startswith("__")}
    try:
        _assign_inplace(mainConfig, cfg)
        yield mainConfig
    finally:
        _assign_inplace(mainConfig, saved)
        for k in [k for k in vars(constants) if not k.startswith("__") and k not in saved_constants]:
            delattr(constants, k)
        for k, v in saved_constants.items():
            setattr(constants, k, v)


def _parse_value(text: str, current):
    low = str(text).strip()
    if isinstance(current, bool) or low.lower() in ("true", "false"):
        return low.lower() in ("true", "1", "yes")
    if low.lower() in ("none", "null"):
        return None
    if hasattr(current, "value") and type(current) is not str:  # Enum
        return type(current)(low)
    for cast in (int, float):
        try:
            return cast(low)
        except ValueError:
            pass
    return low


def apply_overrides(cfg, overrides: Iterable[str]):
    """``section.field[.sub]=value`` overrides on a ``MainConfig`` (e.g.
    ``train.network.SwinUNETR.use_checkpoint=false``)."""
    for item in overrides or ():
        key, sep, value = str(item).partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not key=value")
        parts = key.strip().split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise ValueError(f"override {item!r}: unknown field {key}")
        setattr(obj, parts[-1], _parse_value(value, getattr(obj, parts[-1])))
    return cfg


def load_config(config: Union[None, str, Path, object] = None, overrides: Iterable[str] = ()):
    """A fresh ``MainConfig``: the defaults, a deep copy of a given ``MainConfig``, or a YAML file
    merged as ``tomocpt train --config_file`` does (shared fields propagated); then ``overrides``."""
    from tomocpt.mainConfig import MainConfig

    if config is None:
        cfg = MainConfig()
    elif is_dataclass(config):
        cfg = copy.deepcopy(config)
    else:
        from omegaconf import OmegaConf
        from tomocpt.configManager.configManager import merge_config

        cfg = MainConfig()
        cfg, _ = merge_config(OmegaConf.load(str(config)), cfg)
        shared = cfg.shared
        for f in fields(cfg):
            if f.name == "shared":
                continue
            section = getattr(cfg, f.name)
            for sf in fields(shared):
                if hasattr(section, sf.name) and getattr(section, sf.name) is None \
                        and getattr(shared, sf.name) is not None:
                    setattr(section, sf.name, getattr(shared, sf.name))
        _coerce_enums(cfg, MainConfig())
    return apply_overrides(cfg, overrides)


def _coerce_enums(cfg, defaults):
    """YAML gives plain strings where the dataclass default is an Enum (the CLI converts them;
    ``merge_config`` alone does not)."""
    from enum import Enum

    for f in fields(cfg):
        v, d = getattr(cfg, f.name, None), getattr(defaults, f.name, None)
        if is_dataclass(v) and is_dataclass(d):
            _coerce_enums(v, d)
        elif isinstance(d, Enum) and not isinstance(v, Enum) and v is not None:
            setattr(cfg, f.name, type(d)(str(v)))


def synthetic_class_table(cfg, n_classes: int):
    """Pipeline v2, multi-class, ``n_classes`` particle classes (diameters cycled from
    :data:`SYNTHETIC_DIAMETERS`)."""
    diam = [SYNTHETIC_DIAMETERS[i % len(SYNTHETIC_DIAMETERS)] for i in range(int(n_classes))]
    cfg.prepData.multiclass = True
    cfg.prepData.class_diameters = ",".join(f"{i + 1}:{d:g}" for i, d in enumerate(diam))
    cfg.prepData.class_names = ",".join(f"{i + 1}:class{i + 1}" for i in range(int(n_classes)))
    cfg.train.data_pipeline_version = 2
    return diam


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _plain(obj):
    if is_dataclass(obj):
        return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
    if hasattr(obj, "value") and not isinstance(obj, (str, int, float)):
        return str(obj.value)
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj if isinstance(obj, (str, int, float, bool, type(None))) else str(obj)


#: Fields left out of the config hash (paths and run bookkeeping, not what is computed).
_HASH_SKIP = ("model_dir", "chunks_dir", "training_data_dir", "raw_data_dir", "predictions_dir",
              "experiment_name", "launch_tensorboard")


def config_hash(cfg) -> str:
    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k not in _HASH_SKIP}
        return d
    blob = json.dumps(strip(_plain(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def environment() -> dict:
    import torch

    env = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("monai", "kornia", "torchio", "pytorch_lightning"):
        try:
            env[name] = __import__(name).__version__
        except Exception:
            pass
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def write_json(path: Optional[Union[str, Path]], data: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=1, default=str) + "\n")
