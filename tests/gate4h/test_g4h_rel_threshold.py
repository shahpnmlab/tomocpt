"""G4h-C1..C5: relative improvement threshold for early stopping and the matched LR plateau."""

import pytest
import torch

from gate3 import g3_helpers as G3
from gate4g.test_g4g_lr_plateau import EMA_CURVE
from utils.tiny import applied_main_config

CLASSES = ((1, "A", 150.0), (2, "B", 300.0))


def _cfg(tmp_path, **train):
    cfg = G3.v2_config(tmp_path, CLASSES, cond=False)
    for k, v in train.items():
        setattr(cfg.train, k, v)
    return cfg


def _model():
    from tomocpt.networks.pickingModel import BasePickingModel

    torch.manual_seed(0)
    return BasePickingModel(lr=4e-4, train_continue=None)


def _build(tmp_path, **train):
    """(early-stopping callbacks, configure_optimizers dict)."""
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    from tomocpt.training.helpers import _setup_callbacks

    mc = _cfg(tmp_path, **train)
    with applied_main_config(mc):
        model = _model()
        cbs = _setup_callbacks(mc.train, model, ModelCheckpoint(monitor="val_loss"))
        conf = model.configure_optimizers()
    return [c for c in cbs if isinstance(c, EarlyStopping)], conf


def test_G4h_C1_default_absolute(tmp_path):
    from tomocpt.defaultConfigs.train_config import TrainConfig
    from tomocpt.training.callbacks import RelativeEarlyStopping

    assert TrainConfig().early_stop_rel_delta is None
    es, conf = _build(tmp_path, early_stop_min_delta=2e-4, lr_plateau_match_early_stop=True)
    assert len(es) == 1 and not isinstance(es[0], RelativeEarlyStopping)
    assert abs(es[0].min_delta) == pytest.approx(2e-4)
    s = conf["lr_scheduler"]
    assert s.threshold == pytest.approx(2e-4) and s.threshold_mode == "abs"


def test_G4h_C2_relative_callback_and_scheduler(tmp_path):
    from tomocpt.training.callbacks import RelativeEarlyStopping

    es, conf = _build(tmp_path, early_stop_rel_delta=0.007, early_stop_patience=12,
                      lr_plateau_match_early_stop=True, lr_plateau_patience=5)
    assert len(es) == 1 and isinstance(es[0], RelativeEarlyStopping)
    assert es[0].rel_delta == pytest.approx(0.007) and es[0].patience == 12
    assert es[0].monitor == "val_loss_smooth"
    s = conf["lr_scheduler"]
    assert s.threshold == pytest.approx(0.007) and s.threshold_mode == "rel" and s.patience == 5


def test_G4h_C2_relative_without_match_keeps_legacy_scheduler(tmp_path):
    _, conf = _build(tmp_path, early_stop_rel_delta=0.007)
    s = conf["lr_scheduler"]
    assert s.patience == 9 and s.threshold == pytest.approx(1e-4) and s.threshold_mode == "rel"


@pytest.mark.parametrize("bad", [0.0, -0.01, 1.0, 2.0])
def test_G4h_C3_invalid_values_raise(tmp_path, bad):
    with pytest.raises(ValueError):
        _build(tmp_path, early_stop_rel_delta=bad)


def _replay(scheduler, es, curve):
    opt = scheduler.optimizer
    drops, lr = [], opt.param_groups[0]["lr"]
    for epoch, v in enumerate(curve):
        scheduler.step(v)
        if opt.param_groups[0]["lr"] < lr:
            drops.append(epoch)
            lr = opt.param_groups[0]["lr"]
        stop, _ = es._evaluate_stopping_criteria(torch.tensor(v))
        if stop:
            return drops, epoch
    return drops, None


@pytest.mark.parametrize("scale", [1.0, 100.0, 0.01])
def test_G4h_C4_replay_and_scale_invariance(tmp_path, scale):
    es, conf = _build(tmp_path, early_stop_rel_delta=0.007, early_stop_patience=10,
                      lr_plateau_match_early_stop=True, lr_plateau_patience=5)
    drops, stop = _replay(conf["lr_scheduler"], es[0], [v * scale for v in EMA_CURVE])
    assert drops == [41, 48] and stop == 52
    assert float(es[0].best_score) == pytest.approx(0.01469 * scale, rel=1e-5)


def test_G4h_C5_state_dict_round_trip():
    from tomocpt.training.callbacks import RelativeEarlyStopping

    a = RelativeEarlyStopping(monitor="val_loss_smooth", rel_delta=0.01, patience=3)
    for v in (1.0, 0.95, 0.949):
        a._evaluate_stopping_criteria(torch.tensor(v))
    assert float(a.best_score) == pytest.approx(0.95) and a.wait_count == 1
    b = RelativeEarlyStopping(monitor="val_loss_smooth", rel_delta=0.01, patience=3)
    b.load_state_dict(a.state_dict())
    stop, _ = b._evaluate_stopping_criteria(torch.tensor(0.9406))  # 0.95 * 0.99 = 0.9405: not enough
    assert not stop and b.wait_count == 2
    stop, _ = b._evaluate_stopping_criteria(torch.tensor(0.94))
    assert b.wait_count == 0 and float(b.best_score) == pytest.approx(0.94)
