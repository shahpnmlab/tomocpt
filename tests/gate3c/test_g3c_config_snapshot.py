"""Gate 3c: CLI config snapshot (spec ``03c_AGENT_CONFIG_SNAPSHOT.md``, G3c-C1..C5).

The snapshot API (``set_cli_snapshot``, ``get_cli_snapshot``, ``clear_cli_snapshot``) is imported
inside the tests, so the module imports on code without it.
"""

import copy

import pytest
import yaml

from utils.tiny import applied_main_config, tiny_main_config

#: Train settings of the ``ckpts`` fixture (conftest.py); kept in sync there.
A_TRAIN = dict(n_epochs=3, batch_size=4)
B_TRAIN = dict(n_epochs=9, batch_size=6)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _api():
    from tomocpt.configManager.configManager import (
        clear_cli_snapshot, get_cli_snapshot, set_cli_snapshot)
    return set_cli_snapshot, get_cli_snapshot, clear_cli_snapshot


def _defaults():
    from tomocpt.mainConfig import MainConfig
    return MainConfig()


def _load(path):
    """Load a picking checkpoint strictly on CPU; return a deep copy of its config."""
    from tomocpt.networks.pickingModel import BasePickingModel

    model = BasePickingModel.load_from_checkpoint(path, map_location="cpu", strict=True)
    return copy.deepcopy(model.hparams.config)


def _diff(a, b):
    from tomocpt.configManager.configManager import compare_dataclasses
    return compare_dataclasses(a, b)


def _class_names(cfg):
    from tomocpt.labels.multiclass import class_table_from_config
    return [c.name for c in class_table_from_config(cfg.prepData)]


def _assert_B_own(cfg):
    assert cfg.train.n_epochs == B_TRAIN["n_epochs"]
    assert cfg.train.batch_size == B_TRAIN["batch_size"]
    assert cfg.train.data_pipeline_version == 2
    assert _class_names(cfg) == ["A", "B"]


def _assert_A_own(cfg):
    assert cfg.train.n_epochs == A_TRAIN["n_epochs"]
    assert cfg.train.batch_size == A_TRAIN["batch_size"]
    assert cfg.train.data_pipeline_version == 1


# --------------------------------------------------------------------------------------
# snapshot API
# --------------------------------------------------------------------------------------

def test_G3c_api_set_get_clear_deep_copy():
    set_snap, get_snap, clear_snap = _api()
    clear_snap()
    assert get_snap() is None

    cfg = _defaults()
    cfg.train.n_epochs = 42
    set_snap(cfg)
    snap = get_snap()
    assert snap is not None and snap is not cfg
    assert snap.train is not cfg.train
    assert snap.train.n_epochs == 42
    cfg.train.n_epochs = 7  # the stored copy does not follow the original
    assert get_snap().train.n_epochs == 42
    assert [d[0] for d in _diff(get_snap(), _defaults())] == ["train.n_epochs"]

    clear_snap()
    assert get_snap() is None


# --------------------------------------------------------------------------------------
# G3c-C1: two loads in one process do not leak; order does not matter
# --------------------------------------------------------------------------------------

def _two_orders(p1, p2):
    """Load p1 then p2, and (with fresh globals) p2 then p1. Returns the four configs."""
    with applied_main_config(None):
        c1_first = _load(p1)
        c2_second = _load(p2)
    with applied_main_config(None):
        c2_first = _load(p2)
        c1_second = _load(p1)
    return c1_first, c2_second, c2_first, c1_second


def test_G3c_C1_sequential_loads_keep_own_settings(ckpts):
    from tomocpt.mainConfig import mainConfig

    set_snap, _, _ = _api()
    set_snap(_defaults())  # nothing changed on the command line

    a1, b2, b1, a2 = _two_orders(ckpts["A"], ckpts["B"])
    for cfg in (b1, b2):
        _assert_B_own(cfg)
    for cfg in (a1, a2):
        _assert_A_own(cfg)
    assert _diff(b1, b2) == [], "B loaded after A differs from B loaded first"
    assert _diff(a1, a2) == [], "A loaded after B differs from A loaded first"

    # The injection into the global train section stays (legacy code reads mainConfig.train).
    _load(ckpts["B"])
    assert mainConfig.train.n_epochs == B_TRAIN["n_epochs"]
    assert mainConfig.train.data_pipeline_version == 2


def test_G3c_C1_legacy_and_v2_checkpoint_in_one_process(ckpts):
    set_snap, _, _ = _api()
    set_snap(_defaults())

    l1, b2, b1, l2 = _two_orders(ckpts["legacy"], ckpts["B"])
    _assert_B_own(b1)
    _assert_B_own(b2)
    assert _diff(b1, b2) == []
    assert _diff(l1, l2) == [], "legacy checkpoint loaded after B differs from loaded first"
    assert l1.train.data_pipeline_version == 1


# --------------------------------------------------------------------------------------
# G3c-C2: a command-line change (in the snapshot) still overrides the checkpoint
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_G3c_C2_cli_change_overrides_on_both_loads(ckpts, order):
    set_snap, _, _ = _api()
    snap = _defaults()
    snap.train.n_epochs = 42
    set_snap(snap)
    # The live global holds no CLI change; only the snapshot does.
    with applied_main_config(_defaults()):
        first = _load(ckpts[order[0]])
        second = _load(ckpts[order[1]])
    for cfg, name in ((first, order[0]), (second, order[1])):
        assert cfg.train.n_epochs == 42, f"CLI value lost for {name}"
        own = A_TRAIN if name == "A" else B_TRAIN
        assert cfg.train.batch_size == own["batch_size"], f"{name} lost its own batch_size"
        assert cfg.train.data_pipeline_version == (2 if name == "B" else 1)


# --------------------------------------------------------------------------------------
# G3c-C3: without a snapshot, the legacy computation (live mainConfig = CLI state)
# --------------------------------------------------------------------------------------

def test_G3c_C3_no_snapshot_uses_live_global(ckpts, tmp_path):
    import torch
    from tomocpt.configManager.configManager import (
        backfill_missing_fields, update_config_with_changed_values)
    from tomocpt.mainConfig import _mainConfigNoChanges

    _, get_snap, clear_snap = _api()
    clear_snap()
    assert get_snap() is None

    live = _defaults()
    live.train.n_epochs = 42
    live.train.optimizer.lr = 1e-5
    for name in ("A", "B", "legacy"):
        with applied_main_config(live) as g:
            raw = torch.load(ckpts[name], map_location="cpu", weights_only=False)
            expected = update_config_with_changed_values(
                target=backfill_missing_fields(raw["hyper_parameters"]["config"], _mainConfigNoChanges),
                originaConfig=_mainConfigNoChanges, configAfterCli=copy.deepcopy(g))
            got = _load(ckpts[name])
        assert got.train.n_epochs == 42 and got.train.optimizer.lr == 1e-5
        assert _diff(got, expected) == [], name


# --------------------------------------------------------------------------------------
# G3c-C4: config=None stores a copy of the global, not the global itself
# --------------------------------------------------------------------------------------

def test_G3c_C4_config_none_is_a_copy(tmp_path):
    from tomocpt.mainConfig import mainConfig
    from tomocpt.networks.pickingModel import BasePickingModel

    with applied_main_config(tiny_main_config(tmp_path)):
        before = copy.deepcopy(mainConfig)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        cfg = model.hparams.config
        assert cfg is not mainConfig
        assert cfg.train is not mainConfig.train
        assert cfg.train.network is not mainConfig.train.network
        assert _diff(cfg, before) == []

        mainConfig.train.n_epochs = 77
        mainConfig.train.batch_size = 13
        mainConfig.train.network.SwinUNETR.feature_size = 48
        mainConfig.prepData.desired_particle_pixel_size = 31
        assert _diff(model.hparams.config, before) == []


# --------------------------------------------------------------------------------------
# G3c-C5: the real Typer wrapper sets the snapshot (YAML + CLI applied, before the command)
# --------------------------------------------------------------------------------------

def test_G3c_C5_cli_wrapper_sets_snapshot(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tomocpt.configManager.configManager import create_app
    from tomocpt.mainConfig import mainConfig

    _, get_snap, clear_snap = _api()
    clear_snap()

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump({"train": {
        "n_epochs": 17, "batch_size": 5,
        # required train options (MISSING defaults), supplied by the YAML
        "model_dir": str(tmp_path / "model"), "chunks_dir": str(tmp_path / "chunks")}}))
    args = ["g3cstub", "--config_file", str(cfg_file), "--batch_size", "7", "optimizer.lr=0.0001"]

    seen = {}

    def g3cstub(config=None):
        snap = get_snap()
        seen["snap"] = copy.deepcopy(snap)
        seen["is_global"] = snap is mainConfig
        seen["live"] = copy.deepcopy(mainConfig)

    def g3cother(config=None):  # a second command, so the first is invoked by name
        pass

    # The wrapper reads --config_file from sys.argv when the command is registered.
    monkeypatch.setattr("sys.argv", ["tomocpt", *args])
    app = create_app()
    app.register_command(g3cstub, mainConfig, "train")
    app.register_command(g3cother, mainConfig, "train")
    result = CliRunner().invoke(app.app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"

    assert "snap" in seen, "stub command did not run"
    snap = seen["snap"]
    assert snap is not None, "the CLI wrapper did not set the snapshot before the command ran"
    assert not seen["is_global"], "the snapshot must be a copy, not the global mainConfig"
    assert snap.train.n_epochs == 17          # from the YAML
    assert snap.train.batch_size == 7         # CLI option over the YAML
    assert snap.train.optimizer.lr == 1e-4    # hydra-style override
    assert str(snap.train.model_dir) == str(tmp_path / "model")
    assert _diff(snap, seen["live"]) == [], "snapshot differs from the YAML+CLI-applied config"


def test_G3c_C5_snapshot_does_not_outlive_the_command(tmp_path, monkeypatch):
    """The snapshot is scoped to the command call: after it returns (or raises), the previous
    state (here: no snapshot) is back. Otherwise an in-process CLI invocation (as
    ``tests/gate1/test_g1_config_cli.py`` does, with a non-global config) leaves a stale
    snapshot that later checkpoint loads in the same process would use instead of the live
    ``mainConfig`` ("All earlier tests pass unchanged")."""
    from typer.testing import CliRunner

    from tomocpt.configManager.configManager import create_app
    from tomocpt.mainConfig import MainConfig

    _, get_snap, clear_snap = _api()
    clear_snap()
    args = ["g3cstub", "--predictions_dir", str(tmp_path / "p")]
    seen = {}

    def g3cstub(config=None):
        seen["during"] = get_snap() is not None

    def g3cboom(config=None):
        seen["boom"] = get_snap() is not None
        raise RuntimeError("stub failure")

    monkeypatch.setattr("sys.argv", ["tomocpt", *args])
    app = create_app()
    app.register_command(g3cstub, MainConfig(), "infer")
    app.register_command(g3cboom, MainConfig(), "infer")

    result = CliRunner().invoke(app.app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert seen["during"] is True
    assert get_snap() is None, "snapshot leaked after the command returned"

    result = CliRunner().invoke(app.app, ["g3cboom", "--predictions_dir", str(tmp_path / "p")])
    assert isinstance(result.exception, RuntimeError)
    assert seen["boom"] is True
    assert get_snap() is None, "snapshot leaked after the command raised"
