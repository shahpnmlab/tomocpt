"""G1-C7 (config) and G1-C9 (``tomocpt init`` round trip).

Note on option spelling: the legacy CLI keeps underscores in option names
(``--output_path``, ``--config_file``); ``--output-path`` / ``--config-file`` (as written in
the Gate 1 spec and the README) are rejected by ``v2-gate-0-base``. The YAML is merged only
when ``--config_file`` appears in ``sys.argv`` (``ConfigurableApp.command``). These tests use
the spelling the code accepts.
"""

import copy
import os
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


def _field_paths(obj, prefix=""):
    """Every field path of a dataclass tree (nested and leaves)."""
    out = []
    for f in fields(type(obj)):
        path = f"{prefix}.{f.name}" if prefix else f.name
        out.append(path)
        v = getattr(obj, f.name)
        if is_dataclass(v):
            out.extend(_field_paths(v, path))
    return out


def _get(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _has(obj, path):
    for part in path.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def _delete(obj, path):
    *head, last = path.split(".")
    for part in head:
        obj = getattr(obj, part)
    delattr(obj, last)


# --- G1-C7 -----------------------------------------------------------------------------

def test_G1_C7_sections_instantiate_alone():
    from tomocpt.defaultConfigs.infer_config import InferConfig
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.defaultConfigs.train_config import TrainConfig

    for cls in (PrepdataConfig, TrainConfig, InferConfig):
        cfg = cls()
        assert is_dataclass(cfg)


def test_G1_C7_main_config_sections_are_isolated():
    from tomocpt.mainConfig import MainConfig

    cfg = MainConfig()
    assert not hasattr(cfg.train, "prepData")
    for section in ("prepData", "train", "infer"):
        for other in ("prepData", "train", "infer", "shared"):
            assert not hasattr(getattr(cfg, section), other), f"{section}.{other}"


def test_G1_C7_update_config_keeps_target_defaults_for_missing_source_fields():
    from tomocpt.configManager.configManager import update_config
    from tomocpt.mainConfig import MainConfig

    defaults = MainConfig()
    target = MainConfig()
    source = MainConfig()
    source.train.n_epochs = 3
    source.train.network.SwinUNETR.drop_rate = 0.1
    for path in ("batch_size", "network.SwinUNETR.feature_size", "augmentation", "optimizer.lr"):
        _delete(source.train, path)

    update_config(target.train, source=source.train)

    assert target.train.n_epochs == 3
    assert target.train.network.SwinUNETR.drop_rate == 0.1
    assert target.train.batch_size == defaults.train.batch_size
    assert target.train.network.SwinUNETR.feature_size == defaults.train.network.SwinUNETR.feature_size
    assert target.train.augmentation == defaults.train.augmentation
    assert target.train.optimizer.lr == defaults.train.optimizer.lr


def test_G1_C7_backfill_adds_every_field_a_legacy_checkpoint_lacks(legacy_ckpt):
    from tomocpt.configManager.configManager import backfill_missing_fields
    from tomocpt.mainConfig import MainConfig

    defaults = MainConfig()
    paths = _field_paths(defaults)

    # the real legacy checkpoint config: after backfill it has every current field
    legacy = torch.load(legacy_ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]["config"]
    backfill_missing_fields(legacy, defaults)
    missing = [p for p in paths if not _has(legacy, p)]
    assert not missing, missing

    # the mechanism, for every field: a config that lacks it gets the current default back
    for path in paths:
        cfg = MainConfig()
        _delete(cfg, path)
        if _has(cfg, path):
            # a plain class-level default stays visible after deleting the instance attribute,
            # so an old checkpoint cannot lack it; only default_factory fields can go missing
            continue
        backfill_missing_fields(cfg, defaults)
        assert _has(cfg, path), path
        assert _get(cfg, path) == _get(defaults, path), path


def test_G1_C7_safe_getattr():
    from tomocpt.configManager.configManager import safe_getattr
    from tomocpt.mainConfig import MainConfig

    cfg = MainConfig()
    sentinel = object()
    assert safe_getattr(cfg.train, "n_epochs", sentinel) == cfg.train.n_epochs
    assert safe_getattr(cfg.train, "field_added_by_a_later_gate", sentinel) is sentinel
    assert safe_getattr(cfg.train, "field_added_by_a_later_gate", None) is None

    old = copy.deepcopy(cfg)  # as an old checkpoint unpickles: the field is simply absent
    delattr(old.train, "batch_size")
    assert safe_getattr(old.train, "batch_size", 7) == 7
    assert safe_getattr(old.train, "n_epochs", 7) == cfg.train.n_epochs


# --- G1-C9 -----------------------------------------------------------------------------

def _run_init(tmp_path) -> Path:
    import tomocpt

    out = tmp_path / "x.yaml"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(tomocpt.__file__).resolve().parents[1])  # same code as the tests
    env["CUDA_VISIBLE_DEVICES"] = ""
    res = subprocess.run([sys.executable, "-m", "tomocpt.main", "init", "--output_path", str(out)],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    assert out.is_file()
    return out


def _load_with_config_file(monkeypatch, yaml_path, extra_args=()):
    """Run a ``predict``-shaped command of the real CLI machinery with ``--config_file`` and
    return the infer config the command receives."""
    from typer.testing import CliRunner

    from tomocpt.configManager.configManager import create_app
    from tomocpt.mainConfig import MainConfig

    args = ["predict", "--config_file", str(yaml_path), *extra_args]
    monkeypatch.setattr(sys, "argv", ["tomocpt", *args])  # the YAML is read at registration
    main_config = MainConfig()
    captured = {}

    def predict(config=None):
        captured["infer"] = config

    def train(config=None):
        captured["train"] = config

    app = create_app()
    app.register_command(predict, main_config, "infer")
    app.register_command(train, main_config, "train")
    res = CliRunner().invoke(app.app, args)
    assert res.exit_code == 0, (res.output, res.exception)
    return captured["infer"], main_config


def test_G1_C9_init_writes_a_config_that_loads_back(tmp_path, monkeypatch):
    from tomocpt.defaultConfigs.infer_config import InferConfig, OutputFormat

    yaml_path = _run_init(tmp_path)
    loaded = OmegaConf.load(yaml_path)
    for section in ("shared", "prepData", "train", "infer"):
        assert section in loaded, section

    # unchanged template: loads; required (???) fields come from the command line
    infer, _ = _load_with_config_file(monkeypatch, yaml_path,
                                      ["--length", "150", "--predictions_dir", str(tmp_path / "p")])
    defaults = InferConfig()
    assert infer.length == 150
    assert infer.confidence_threshold == defaults.confidence_threshold
    assert infer.predictions_coord_format == OutputFormat.relion_31
    assert infer.use_tta == defaults.use_tta

    # edited template: the edited values reach the command
    data = OmegaConf.to_container(loaded)
    data["infer"]["length"] = 123.0
    data["infer"]["predictions_dir"] = str(tmp_path / "preds")
    data["infer"]["confidence_threshold"] = 0.77
    edited = tmp_path / "edited.yaml"
    OmegaConf.save(OmegaConf.create(data), edited)
    infer, _ = _load_with_config_file(monkeypatch, edited)
    assert infer.length == 123.0
    assert infer.confidence_threshold == pytest.approx(0.77)
    assert str(infer.predictions_dir) == str(tmp_path / "preds")
    assert infer.use_tta == defaults.use_tta
