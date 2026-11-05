"""G4g-C1..C4: ReduceLROnPlateau matched to early stopping."""

import pytest
import torch

from gate3 import g3_helpers as G3
from utils.tiny import applied_main_config

CLASSES = ((1, "A", 150.0), (2, "B", 300.0))

# val_loss_smooth per epoch of the 3b EMA pilot (CZII fold 0, 5 Nov 2026), epochs 0..58
EMA_CURVE = [0.851619, 0.607239, 0.43488, 0.313249, 0.227106, 0.166001, 0.122509, 0.091622, 0.069679,
             0.054107, 0.043068, 0.035179, 0.029521, 0.025473, 0.022599, 0.02055, 0.019094, 0.01804,
             0.017256, 0.01668, 0.016236, 0.015931, 0.015721, 0.015544, 0.015395, 0.015289, 0.015195,
             0.015126, 0.015041, 0.014964, 0.01491, 0.014874, 0.014853, 0.014839, 0.014827, 0.014802,
             0.014798, 0.014788, 0.014775, 0.014759, 0.014741, 0.014714, 0.01469, 0.0147, 0.014685,
             0.014679, 0.014675, 0.014654, 0.01463, 0.014627, 0.014618, 0.014626, 0.014642, 0.014659,
             0.014681, 0.014689, 0.014682, 0.014684, 0.014694]


def _cfg(tmp_path, **train):
    cfg = G3.v2_config(tmp_path, CLASSES, cond=False)
    for k, v in train.items():
        setattr(cfg.train, k, v)
    return cfg


def _model(train_continue=None):
    from tomocpt.networks.pickingModel import BasePickingModel

    torch.manual_seed(0)
    return BasePickingModel(lr=4e-4, train_continue=train_continue)


def _conf(tmp_path, train_continue=None, **train):
    with applied_main_config(_cfg(tmp_path, **train)):
        return _model(train_continue).configure_optimizers()


def test_G4g_C1_defaults_legacy_scheduler(tmp_path):
    from tomocpt.defaultConfigs.train_config import TrainConfig

    t = TrainConfig()
    assert t.lr_plateau_match_early_stop is False and t.lr_plateau_patience == 5
    conf = _conf(tmp_path)
    s = conf["lr_scheduler"]
    assert isinstance(s, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert conf["monitor"] == "val_loss_smooth"
    assert s.patience == 9 and s.factor == pytest.approx(0.5) and s.cooldown == 1
    assert s.threshold == pytest.approx(1e-4) and s.threshold_mode == "rel" and s.mode == "min"


def test_G4g_C2_matched_scheduler(tmp_path):
    conf = _conf(tmp_path, lr_plateau_match_early_stop=True, early_stop_min_delta=2e-4,
                 early_stop_patience=10, lr_plateau_patience=4)
    s = conf["lr_scheduler"]
    assert isinstance(s, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert conf["monitor"] == "val_loss_smooth"
    assert s.patience == 4 and s.factor == pytest.approx(0.5) and s.cooldown == 1
    assert s.threshold == pytest.approx(2e-4) and s.threshold_mode == "abs" and s.mode == "min"


def test_G4g_C2_matched_keeps_fine_tune_groups(tmp_path):
    ref = _conf(tmp_path, train_continue="finetune")["optimizer"]
    got = _conf(tmp_path, train_continue="finetune", lr_plateau_match_early_stop=True)["optimizer"]
    assert sorted(g["lr"] for g in got.param_groups) == pytest.approx(sorted(g["lr"] for g in ref.param_groups))
    assert len(ref.param_groups) > 1


def test_G4g_C3_patience_not_below_early_stop_raises(tmp_path):
    with pytest.raises(ValueError):
        _conf(tmp_path, lr_plateau_match_early_stop=True, lr_plateau_patience=10, early_stop_patience=10)


def test_G4g_C3_schedule_free_raises(tmp_path):
    with pytest.raises(ValueError):
        _conf(tmp_path, lr_plateau_match_early_stop=True, schedule_free=True)


def _replay(scheduler):
    """Steps ``scheduler`` and Lightning's EarlyStopping (min_delta 1e-4, patience 10) through
    EMA_CURVE; returns (epochs at which the LR dropped, epoch at which early stopping fired)."""
    from pytorch_lightning.callbacks import EarlyStopping

    es = EarlyStopping(monitor="val_loss_smooth", min_delta=1e-4, patience=10, mode="min")
    opt = scheduler.optimizer
    drops, lr = [], opt.param_groups[0]["lr"]
    for epoch, v in enumerate(EMA_CURVE):
        scheduler.step(v)
        if opt.param_groups[0]["lr"] < lr:
            drops.append(epoch)
            lr = opt.param_groups[0]["lr"]
        stop, _ = es._evaluate_stopping_criteria(torch.tensor(v))
        if stop:
            return drops, epoch
    return drops, None


def test_G4g_C4_matched_halves_before_early_stop_legacy_never(tmp_path):
    legacy = _conf(tmp_path, early_stop_min_delta=1e-4)["lr_scheduler"]
    drops, stop = _replay(legacy)
    assert stop is not None and drops == []
    matched = _conf(tmp_path, early_stop_min_delta=1e-4, lr_plateau_match_early_stop=True,
                    lr_plateau_patience=5)["lr_scheduler"]
    drops, stop = _replay(matched)
    assert stop is not None and len(drops) >= 1 and drops[0] < stop
