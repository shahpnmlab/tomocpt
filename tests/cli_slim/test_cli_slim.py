"""Slim CLI (human decision 9 Nov 2026: "curtail the prep/train/infer options and move them into
configs and dot overrides").

* ``--help`` of ``prepare_data`` / ``train`` / ``predict`` lists only the everyday flags of
  ``tomocpt.configManager.cli_options.CLI_FLAGS`` (plus required fields and ``--config_file``).
* Every other setting is reachable as ``section.field=value`` and from the YAML, with the field's
  type: bools (``false`` is False), ``none`` for Optional, Enums, Paths, tuples.
* YAML values of non-flag fields arrive typed (an Enum field holds the Enum, not the string).
* Old long flags of removed options still work through ``ConfigurableApp.run`` (rewritten to
  ``key=value``, with a deprecation warning), so existing scripts keep running.
"""
import sys
from enum import Enum
from pathlib import Path

import pytest
from typer.testing import CliRunner

TESTS_DIR = Path(__file__).resolve().parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from utils.tiny import applied_main_config  # noqa: E402

SECTIONS = {"prepare_data": "prepData", "train": "train", "predict": "infer"}


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tomocpt"])
    with applied_main_config(None):
        yield


def _app(seen, argv=("tomocpt",), monkeypatch=None):
    """App with stub commands named as the real ones; each stub records its config."""
    from tomocpt.configManager.configManager import create_app
    from tomocpt.mainConfig import mainConfig

    if monkeypatch is not None:
        monkeypatch.setattr(sys, "argv", list(argv))

    def prepare_data(config=None):
        seen["prepare_data"] = config

    def train(config=None):
        seen["train"] = config

    def predict(config=None):
        seen["predict"] = config

    app = create_app()
    app.register_command(prepare_data, mainConfig, "prepData")
    app.register_command(train, mainConfig, "train")
    app.register_command(predict, mainConfig, "infer")
    return app


def _help_options(app, cmd):
    # Rich truncates option names to the terminal width; give it room (ruling 9 Nov 2026).
    out = CliRunner().invoke(app.app, [cmd, "--help"], terminal_width=250,
                             env={"COLUMNS": "250", "TERMINAL_WIDTH": "250"}).output
    opts = set()
    for line in out.splitlines():
        for tok in line.replace("│", " ").split():
            if tok.startswith("--"):
                opts.add(tok.split("=")[0].rstrip(","))
    return out, opts


def _required_fields(section_obj, prefix=""):
    from dataclasses import fields, is_dataclass

    from omegaconf import MISSING

    out = set()
    for f in fields(type(section_obj)):
        v = getattr(section_obj, f.name)
        if is_dataclass(v):
            out |= _required_fields(v, f"{prefix}{f.name}.")
        elif v is MISSING:
            out.add(prefix + f.name)
    return out


# --------------------------------------------------------------------------------------
# help
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", list(SECTIONS))
def test_help_lists_only_everyday_flags(cmd):
    from tomocpt.configManager.cli_options import CLI_FLAGS
    from tomocpt.mainConfig import MainConfig

    app = _app({})
    out, opts = _help_options(app, cmd)
    section = SECTIONS[cmd]
    allowed = set(CLI_FLAGS[section]) | _required_fields(getattr(MainConfig(), section))
    allowed_opts = {f"--{n}" for n in allowed} | {f"--no-{n}" for n in allowed}
    extra = opts - allowed_opts - {"--config_file", "--config_merge_preference", "--help"}
    assert not extra, f"{cmd} --help shows non-everyday flags: {sorted(extra)}"
    assert {f"--{n}" for n in CLI_FLAGS[section]} <= opts
    assert len(CLI_FLAGS[section]) <= 16
    assert "key=value" in out, "the help must say how to set the other settings"


def test_whitelist_names_are_config_fields():
    from tomocpt.configManager.cli_options import CLI_FLAGS
    from tomocpt.mainConfig import MainConfig

    m = MainConfig()
    for section, names in CLI_FLAGS.items():
        for name in names:
            obj = getattr(m, section)
            for part in name.split("."):
                assert hasattr(obj, part), f"{section}.{name}"
                obj = getattr(obj, part)


def test_train_help_drops_the_long_tail():
    _, opts = _help_options(_app({}), "train")
    for gone in ("--augmentation.affine_p", "--ssl.dense_weight", "--discriminative.temperature",
                 "--particles.class_head", "--early_stop_patience"):
        assert gone not in opts


# --------------------------------------------------------------------------------------
# dot overrides
# --------------------------------------------------------------------------------------

BASE = ["train", "--model_dir", "/tmp/m", "--chunks_dir", "/tmp/c"]


@pytest.mark.parametrize("arg, path, expected", [
    ("particles.loss_checkpoint=false", "particles.loss_checkpoint", False),
    ("particles.loss_checkpoint=False", "particles.loss_checkpoint", False),
    ("particles.loss_checkpoint=0", "particles.loss_checkpoint", False),
    ("launch_tensorboard=true", "launch_tensorboard", True),
    ("particles.class_weights=none", "particles.class_weights", None),
    ("particles.class_weights=1:1.0,2:2.0", "particles.class_weights", "1:1.0,2:2.0"),
    ("val_stems=a,b", "val_stems", "a,b"),
    ("val_stems=null", "val_stems", None),
    ("optimizer.lr=1e-3", "optimizer.lr", 1e-3),
    ("n_epochs=3", "n_epochs", 3),
    ("particles.cross_class_neg_weight=2", "particles.cross_class_neg_weight", 2.0),
])
def test_dot_override_types(arg, path, expected):
    seen = {}
    r = CliRunner().invoke(_app(seen).app, BASE + [arg])
    assert r.exit_code == 0, (r.output, r.exception)
    obj = seen["train"]
    for p in path.split("."):
        obj = getattr(obj, p)
    assert obj == expected and type(obj) is type(expected), (obj, type(obj))


def test_dot_override_enum_and_path():
    seen = {}
    r = CliRunner().invoke(_app(seen).app, BASE + ["augmentation.rotation_mode=flips", "resume_from=/tmp/x.ckpt"])
    assert r.exit_code == 0, (r.output, r.exception)
    cfg = seen["train"]
    assert isinstance(cfg.augmentation.rotation_mode, Enum) and cfg.augmentation.rotation_mode.value == "flips"
    assert cfg.resume_from == Path("/tmp/x.ckpt")


@pytest.mark.parametrize("arg", ["particles.loss_checkpoint=maybe", "no_such_field=1", "particles.nope=1",
                                 "n_epochs=three"])
def test_bad_dot_override_is_an_error(arg):
    r = CliRunner().invoke(_app({}).app, BASE + [arg])
    assert r.exit_code != 0


# --------------------------------------------------------------------------------------
# YAML
# --------------------------------------------------------------------------------------

def test_yaml_values_of_non_flag_fields_arrive_typed(tmp_path, monkeypatch):
    y = tmp_path / "c.yaml"
    y.write_text(
        "train:\n"
        "  model_dir: /tmp/m\n"
        "  chunks_dir: /tmp/c\n"
        "  val_stems: a,b\n"
        "  particles:\n"
        "    loss_checkpoint: false\n"
        "    class_head: softmax\n"
        "  augmentation:\n"
        "    rotation_mode: flips\n"
        "  network:\n"
        "    TORCH_FLOAT_PRECISION: '32'\n"
    )
    seen = {}
    argv = ["tomocpt", "train", "--config_file", str(y)]
    app = _app(seen, argv, monkeypatch)
    r = CliRunner().invoke(app.app, argv[1:] + ["optimizer.lr=0.002"])
    assert r.exit_code == 0, (r.output, r.exception)
    cfg = seen["train"]
    assert cfg.val_stems == "a,b"
    assert cfg.particles.loss_checkpoint is False
    assert cfg.particles.class_head == "softmax"
    assert isinstance(cfg.augmentation.rotation_mode, Enum) and cfg.augmentation.rotation_mode.value == "flips"
    assert isinstance(cfg.network.TORCH_FLOAT_PRECISION, Enum)
    assert cfg.model_dir == Path("/tmp/m")
    assert cfg.optimizer.lr == 0.002


# --------------------------------------------------------------------------------------
# old long flags
# --------------------------------------------------------------------------------------

def _run(app, args):
    with pytest.raises(SystemExit) as e:
        app.run(args)
    return e.value.code


def test_old_long_flags_still_work_with_a_warning(caplog):
    seen = {}
    app = _app(seen)
    with pytest.warns(DeprecationWarning, match="augmentation.affine_p=0.5"):
        code = _run(app, BASE + ["--augmentation.affine_p", "0.5", "--particles.class_head=softmax",
                                 "--early_stop_patience", "7"])
    assert code in (0, None)
    cfg = seen["train"]
    assert cfg.augmentation.affine_p == 0.5
    assert cfg.particles.class_head == "softmax"
    assert cfg.early_stop_patience == 7


def test_old_boolean_flags_still_work():
    seen = {}
    app = _app(seen)
    with pytest.warns(DeprecationWarning):
        code = _run(app, ["predict", "--predictions_dir", "/tmp/p", "--no-cross_class_nms", "--invert_contrast"])
    assert code in (0, None)
    assert seen["predict"].cross_class_nms is False
    assert seen["predict"].invert_contrast is True


def test_everyday_flags_are_not_rewritten():
    seen = {}
    app = _app(seen)
    code = _run(app, BASE + ["--batch_size", "7", "--n_epochs", "2"])
    assert code in (0, None)
    assert seen["train"].batch_size == 7 and seen["train"].n_epochs == 2
