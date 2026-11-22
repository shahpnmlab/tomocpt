"""Gate 10, section 2: ``tomocpt init --level`` and the help epilog.

Criteria:
* C8   the file written at every level loads with ``--config_file`` for prepare_data, train and
       predict and gives the default config (round trip); a partial (basic) file loads with the
       omitted fields at their defaults.
* C9   a partial file does not leak a nested value into a same-named field of the parent section.
* C10  levels nest: advanced has every basic field, diagnostic every non-internal field; internal
       fields are never written; a level writes nothing above it.
* C11  ``init`` accepts ``key=value`` overrides; they appear in the file (even above the level).
* C12  deterministic output.
* C13  valid YAML with help strings containing ``:``, ``#``, quotes and newlines.
* C14  each field is preceded by a comment with its help (and the allowed values of an enum);
       level headers; the top header points to --config_file, key=value and the docs.
* C15  ``HELP_EPILOG`` names ``tomocpt init --level`` and the docs; ``init --help`` shows
       ``--level``; prepare_data / train / predict get no new flag.
"""
import re
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest
import typer
from omegaconf import MISSING, OmegaConf
from typer.testing import CliRunner

SECTIONS = ("shared", "prepData", "train", "infer")
COMMANDS = {"prepare_data": "prepData", "train": "train", "predict": "infer"}
REQUIRED_ARGS = {
    "prepare_data": ["--raw_data_dir", "/d/t", "--particle_length_ang", "150", "--coordinate_files", "/d/c.star"],
    "train": ["--model_dir", "/d/m"],
    "predict": ["--predictions_dir", "/d/p"],
}
REQUIRED_VALUES = {
    "prepData": {"raw_data_dir": "/d/t", "particle_length_ang": "150", "coordinate_files": "/d/c.star"},
    "train": {"model_dir": "/d/m"},
    "infer": {"predictions_dir": "/d/p"},
}


def _leaves(obj, prefix=""):
    out = []
    for f in fields(type(obj)):
        v = getattr(obj, f.name)
        if is_dataclass(v):
            out.extend(_leaves(v, f"{prefix}{f.name}."))
        else:
            out.append(prefix + f.name)
    return out


def _yaml_leaves(d, prefix=""):
    out = set()
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _yaml_leaves(v, f"{prefix}{k}.")
        else:
            out.add(prefix + k)
    return out


def _get(obj, dotted):
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def _set(obj, dotted, value):
    *head, last = dotted.split(".")
    for p in head:
        obj = getattr(obj, p)
    setattr(obj, last, value)


def _write(tmp_path, level, name=None):
    from tomocpt.configManager.initializer import init

    out = tmp_path / (name or f"{level}.yaml")
    init(output_path=out, level=level)
    assert out.is_file()
    return out


def _load(path):
    return OmegaConf.to_container(OmegaConf.load(path))


def _stub_app(main_config):
    from tomocpt.configManager.configManager import create_app

    seen = {}

    def prepare_data(config=None):
        seen["prepare_data"] = config

    def train(config=None):
        seen["train"] = config

    def predict(config=None):
        seen["predict"] = config

    app = create_app()
    app.register_command(prepare_data, main_config, "prepData")
    app.register_command(train, main_config, "train")
    app.register_command(predict, main_config, "infer")
    return app, seen


def _run_with_file(monkeypatch, cmd, path, extra=()):
    from tomocpt.mainConfig import MainConfig

    args = [cmd, "--config_file", str(path), *REQUIRED_ARGS[cmd], *extra]
    monkeypatch.setattr(sys, "argv", ["tomocpt", *args])  # the YAML is read at registration
    main_config = MainConfig()
    app, seen = _stub_app(main_config)
    res = CliRunner().invoke(app.app, args)
    assert res.exit_code == 0, (res.output, res.exception)
    return seen[cmd]


def _expected(section):
    """The default section with the required values given on the command line."""
    from pathlib import Path

    from tomocpt.configManager.configManager import parse_value
    from tomocpt.mainConfig import MainConfig

    sec = getattr(MainConfig(), section)
    for name, value in REQUIRED_VALUES[section].items():
        f = next(f for f in fields(type(sec)) if f.name == name)
        _set(sec, name, parse_value(f.type, value, name))
    return sec


def _diff(a, b):
    from tomocpt.configManager.configManager import compare_dataclasses

    return compare_dataclasses(a, b)


# --- C8 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["basic", "advanced", "diagnostic"])
@pytest.mark.parametrize("cmd", list(COMMANDS))
def test_G10_C8_round_trip(tmp_path, monkeypatch, level, cmd):
    path = _write(tmp_path, level)
    got = _run_with_file(monkeypatch, cmd, path)
    expected = _expected(COMMANDS[cmd])
    assert not _diff(got, expected), _diff(got, expected)


def test_G10_C8_partial_file_loads_with_defaults(tmp_path, monkeypatch):
    p = tmp_path / "partial.yaml"
    p.write_text("train:\n  n_epochs: 7\n  particles:\n    class_head: softmax\n")
    got = _run_with_file(monkeypatch, "train", p)
    expected = _expected("train")
    expected.n_epochs = 7
    expected.particles.class_head = "softmax"
    assert not _diff(got, expected), _diff(got, expected)


# --- C9 ---------------------------------------------------------------------------------

def test_G10_C9_partial_file_does_not_leak_nested_keys(tmp_path, monkeypatch):
    # train.temperature (distillation) and train.discriminative.temperature share a name
    p = tmp_path / "partial.yaml"
    p.write_text("train:\n  discriminative:\n    temperature: 0.2\n")
    got = _run_with_file(monkeypatch, "train", p)
    assert got.discriminative.temperature == pytest.approx(0.2)
    assert got.temperature == _expected("train").temperature


# --- C10 --------------------------------------------------------------------------------

def _non_internal(levels):
    from tomocpt.configManager.option_levels import option_level
    from tomocpt.mainConfig import MainConfig

    m = MainConfig()
    return {f"{s}.{p}" for s in SECTIONS for p in _leaves(getattr(m, s)) if option_level(s, p) in levels}


def test_G10_C10_levels_nest(tmp_path):
    files = {lv: _yaml_leaves(_load(_write(tmp_path, lv))) for lv in ("basic", "advanced", "diagnostic")}
    basic = _non_internal({"basic"})
    advanced = _non_internal({"basic", "advanced"})
    everything = _non_internal({"basic", "advanced", "diagnostic"})
    assert files["basic"] == basic
    assert files["advanced"] == advanced
    assert files["diagnostic"] == everything
    assert basic <= files["advanced"] <= files["diagnostic"]
    internal = {f"{s}.{p}" for s in SECTIONS for p in _leaves(getattr(__import__("tomocpt.mainConfig", fromlist=["x"]).MainConfig(), s))} - everything
    assert "train.optimizer._target_" in internal
    assert not (files["diagnostic"] & internal)


def test_G10_C10_default_level_is_basic(tmp_path):
    from tomocpt.configManager.initializer import init

    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    init(output_path=a)
    init(output_path=b, level="basic")
    assert a.read_text() == b.read_text()


# --- C11 --------------------------------------------------------------------------------

def _init_app():
    from tomocpt.configManager.configManager import create_app
    from tomocpt.configManager.initializer import init
    from tomocpt.mainConfig import MainConfig

    app = create_app()
    app.register_command(init, MainConfig(), None, False)
    return app


def test_G10_C11_overrides_appear_in_the_file(tmp_path):
    out = tmp_path / "o.yaml"
    res = CliRunner().invoke(_init_app().app, ["init", "--output_path", str(out), "--level", "basic",
                                               "train.n_epochs=3", "train.particles.class_head=softmax",
                                               "infer.calibration_beta=2.5"])
    assert res.exit_code == 0, (res.output, res.exception)
    d = _load(out)
    assert d["train"]["n_epochs"] == 3
    assert d["train"]["particles"]["class_head"] == "softmax"  # advanced field, written as it differs
    assert d["infer"]["calibration_beta"] == 2.5
    assert "particles" not in _load(_write(tmp_path, "basic", "plain.yaml"))["train"]


def test_G10_C11_cli_level_option(tmp_path):
    out = tmp_path / "d.yaml"
    res = CliRunner().invoke(_init_app().app, ["init", "--output_path", str(out), "--level", "diagnostic"])
    assert res.exit_code == 0, (res.output, res.exception)
    assert _yaml_leaves(_load(out)) == _non_internal({"basic", "advanced", "diagnostic"})
    bad = CliRunner().invoke(_init_app().app, ["init", "--output_path", str(tmp_path / "x.yaml"), "--level", "expert"])
    assert bad.exit_code != 0


# --- C12 --------------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["basic", "advanced", "diagnostic"])
def test_G10_C12_deterministic(tmp_path, level):
    assert _write(tmp_path, level, "a.yaml").read_text() == _write(tmp_path, level, "b.yaml").read_text()


# --- C13 --------------------------------------------------------------------------------

def _option_info(cls, name):
    hint = get_type_hints(cls, include_extras=True)[name]
    return next(m for m in get_args(hint)[1:] if isinstance(m, typer.models.OptionInfo))


def test_G10_C13_yaml_valid_with_awkward_help(tmp_path, monkeypatch):
    from tomocpt.defaultConfigs.infer_config import InferConfig

    awkward = "a: b # c 'd' \"e\"\n- f: [g] {h}\n  ---\n% & * ! | >"
    monkeypatch.setattr(_option_info(InferConfig, "length"), "help", awkward)
    monkeypatch.setattr(_option_info(InferConfig, "confidence_threshold"), "help", "x" * 350 + ": # y")
    path = _write(tmp_path, "basic")
    d = _load(path)
    assert d["infer"]["length"] is None
    assert d["infer"]["confidence_threshold"] == 0.3
    assert "a: b # c 'd'" in path.read_text()


# --- C14 --------------------------------------------------------------------------------

def _comment_blocks(text):
    """``{dotted key: comment text above it}`` from the written YAML."""
    blocks, stack, pending = {}, [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            pending = []
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped.startswith("#"):
            pending.append(stripped.lstrip("#").strip())
            continue
        key = stripped.split(":", 1)[0]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([k for _, k in stack] + [key])
        blocks[dotted] = " ".join(pending)
        if stripped.endswith(":"):
            stack.append((indent, key))
        pending = []
    return blocks


def _norm(s):
    return " ".join(s.split())


def test_G10_C14_comments_hold_help_and_choices(tmp_path):
    from tomocpt.configManager.option_levels import option_help

    text = _write(tmp_path, "diagnostic").read_text()
    blocks = _comment_blocks(text)
    leaves = _yaml_leaves(_load(tmp_path / "diagnostic.yaml"))
    for dotted in leaves:
        section, path = dotted.split(".", 1)
        assert dotted in blocks, dotted
        assert _norm(option_help(section, path)) in _norm(blocks[dotted]), dotted
    assert "star" in blocks["prepData.coordinate_file_type"] and "imod" in blocks["prepData.coordinate_file_type"]
    for v in ("warp", "relion_31", "relion_50", "imod"):
        assert v in blocks["infer.predictions_coord_format"]


def test_G10_C14_level_headers_and_top_header(tmp_path):
    text = _write(tmp_path, "diagnostic").read_text()
    top = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        top.append(line)
    head = "\n".join(top)
    assert top, "no header comment"
    for needle in ("--config_file", "key=value", "docs/options.md", "docs/tuning.md", "docs/examples",
                   "--level"):
        assert needle in head, needle
    train_part = text.split("\ntrain:", 1)[1].split("\ninfer:", 1)[0]
    i_basic = train_part.index("# --- basic ---")
    i_adv = train_part.index("# --- advanced ---")
    i_diag = train_part.index("# --- diagnostic ---")
    assert i_basic < i_adv < i_diag
    assert train_part.index("\n  n_epochs:") < i_adv  # basic field under the basic header
    basic_text = _write(tmp_path, "basic", "b.yaml").read_text()
    assert "# --- advanced ---" not in basic_text and "# --- diagnostic ---" not in basic_text


# --- C15 --------------------------------------------------------------------------------

def test_G10_C15_epilog_points_to_levels_and_docs():
    from tomocpt.configManager.cli_options import HELP_EPILOG

    assert "tomocpt init --level advanced" in HELP_EPILOG
    assert "docs/options.md" in HELP_EPILOG and "docs/tuning.md" in HELP_EPILOG
    assert "key=value" in HELP_EPILOG


def test_G10_C15_init_help_shows_level():
    out = CliRunner().invoke(_init_app().app, ["init", "--help"], terminal_width=200,
                             env={"COLUMNS": "200"}).output
    assert "--level" in out
    for lv in ("basic", "advanced", "diagnostic"):
        assert lv in out
