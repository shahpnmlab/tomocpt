"""G4c-C4: checkpoints with different ``train`` settings load in one process (TODO 3a.4)."""

import copy
import json
from pathlib import Path

import pytest
import torch

from gate4 import g4_helpers as H4
from gate4b import g4b_helpers as HB
from gate4c import g4c_helpers as H
from gate4c.conftest import ABC_CLASSES


def _w(model):
    return float(model.hparams.config.train.discriminative.contrastive_weight)


def _has_head(model):
    return any(n.startswith("contrastive_head.") for n, _ in model.named_parameters())


@pytest.mark.parametrize("order", [("contrastive", "plain"), ("plain", "contrastive")])
def test_G4c_C4_load_member_isolated_in_either_order(plain_and_contrastive, order):
    from tomocpt.configManager.configManager import get_cli_snapshot
    from tomocpt.mainConfig import mainConfig
    from tomocpt.predict.ensemble import load_member_isolated

    before_train, before_snap = H.state_of(mainConfig.train), get_cli_snapshot()
    for name in order:
        ck = plain_and_contrastive[name]
        model = load_member_isolated(ck["path"], torch.device("cpu"))
        assert _w(model) == pytest.approx(ck["contrastive_weight"]), name
        assert _has_head(model) == (ck["contrastive_weight"] > 0), name
    assert H.state_of(mainConfig.train) == before_train
    assert get_cli_snapshot() is before_snap


def test_G4c_C4_load_member_isolated_with_a_cli_snapshot(plain_and_contrastive):
    """A CLI snapshot (as in ``tomocpt train`` / ``predict``) must not leak either; it is restored."""
    from tomocpt.configManager.configManager import clear_cli_snapshot, get_cli_snapshot, set_cli_snapshot
    from tomocpt.mainConfig import mainConfig
    from tomocpt.predict.ensemble import load_member_isolated

    cli = copy.deepcopy(mainConfig)
    cli.train.n_epochs = 3
    set_cli_snapshot(cli)
    try:
        snap = get_cli_snapshot()
        for name in ("contrastive", "plain", "contrastive"):
            ck = plain_and_contrastive[name]
            model = load_member_isolated(ck["path"], torch.device("cpu"))
            assert _w(model) == pytest.approx(ck["contrastive_weight"])
        assert get_cli_snapshot() is snap
    finally:
        clear_cli_snapshot()


def test_G4c_C4_load_ensemble_with_mixed_members(tmp_path, plain_and_contrastive):
    from tomocpt.mainConfig import mainConfig
    from tomocpt.predict.ensemble import load_ensemble

    before = H.state_of(mainConfig.train)
    ens = HB.write_ensemble(tmp_path / "e.json", [plain_and_contrastive["contrastive"]["path"],
                                                  plain_and_contrastive["plain"]["path"]])
    model = load_ensemble(ens)
    assert [_w(m) for m in model.members] == pytest.approx([0.1, 0.0])
    assert H.state_of(mainConfig.train) == before


def test_G4c_C4_build_ensemble_with_mixed_members(tmp_path, monkeypatch, plain_and_contrastive):
    from tomocpt.defaultConfigs.infer_config import InferConfig
    from tomocpt.mainConfig import mainConfig
    from tomocpt.predict.ensemble import build_ensemble

    before = H.state_of(mainConfig.train)
    ckpts = []
    for k, name in enumerate(("contrastive", "plain")):
        v = HB.blob_val_tomogram(tmp_path / "tomos", f"TS_{k}", ABC_CLASSES, shift=2.0 * k)
        HB.write_fold_layout(tmp_path / f"chunks{k}", ABC_CLASSES, [v["entry"]], folds=2, fold=k)
        ckpts.append(HB.fold_ckpt(plain_and_contrastive[name]["path"], tmp_path / "ck" / f"f{k}.ckpt",
                                  tmp_path / f"chunks{k}"))
    cfg = InferConfig()
    cfg.use_cuda = False
    cfg.predictions_dir = tmp_path / "pred"
    cfg.calibration_min_particles = 1
    with H4.inprocess_dask(monkeypatch), H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=False)):
        rec = build_ensemble([str(c) for c in ckpts], tmp_path / "ens.json", cfg)
    assert len(rec["checkpoints"]) == 2
    assert H.state_of(mainConfig.train) == before
