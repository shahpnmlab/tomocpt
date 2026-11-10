"""Gate 8g (human request 10 Nov 2026): learning rate and EMA decay follow the effective batch size.

``train.lr_scaling``: ``none`` (opt-in phase default) | ``sqrt`` (Adam-family rule;
Krizhevsky 2014, Malladi et al. 2022) | ``linear`` (Goyal et al. 2017, SGD). The reference is the
recipe's tuning point, ``train.lr_reference_batch`` (16). With ``k = effective batch /
reference`` (effective = ``batch_size x devices``): ``lr -> lr * sqrt(k)`` or ``lr * k``, and
``ema_decay -> ema_decay ** k`` (Busbridge et al. 2023), so the EMA spans the same number of
samples. ``k = 1`` changes nothing, bit for bit. Epochs are not changed (Goyal: same epochs).
"""
import math

import pytest

from gate4f import g4f_helpers as H4
from utils.tiny import applied_main_config


def test_G8g_C1_rules():
    from tomocpt.training.hparam_scaling import scaled_ema_decay, scaled_lr

    assert scaled_lr(4e-4, 4.0, "none") == 4e-4
    assert scaled_lr(4e-4, 4.0, "sqrt") == pytest.approx(8e-4, rel=1e-12)
    assert scaled_lr(4e-4, 4.0, "linear") == pytest.approx(16e-4, rel=1e-12)
    assert scaled_lr(4e-4, 0.25, "sqrt") == pytest.approx(2e-4, rel=1e-12)
    assert scaled_ema_decay(0.999, 4.0, "none") == 0.999
    for rule in ("sqrt", "linear"):
        assert scaled_ema_decay(0.999, 4.0, rule) == pytest.approx(0.999 ** 4, rel=1e-12)
        assert scaled_lr(4e-4, 1.0, rule) == 4e-4          # k = 1: exactly unchanged
        assert scaled_ema_decay(0.999, 1.0, rule) == 0.999
    with pytest.raises(ValueError, match="lr_scaling"):
        scaled_lr(4e-4, 2.0, "cubic")


def test_G8g_C2_scale_factor():
    from tomocpt.training.hparam_scaling import scale_factor

    assert scale_factor(batch_size=16, devices=1, reference_batch=16) == 1.0
    assert scale_factor(batch_size=16, devices=2, reference_batch=16) == 2.0   # DDP: per-GPU batch x GPUs
    assert scale_factor(batch_size=4, devices=1, reference_batch=16) == 0.25
    with pytest.raises(ValueError):
        scale_factor(batch_size=16, devices=1, reference_batch=0)


def test_G8g_C3_defaults():
    from tomocpt.defaultConfigs.train_config import OptimizerConfig, TrainConfig

    t = TrainConfig()
    assert t.lr_scaling == "none" and t.lr_reference_batch == 16 and OptimizerConfig().lr == 4e-4
    assert not hasattr(OptimizerConfig(), "lr_scaling")  # optimizer fields are passed to the optimizer


def _callbacks_ema(mc, model):
    from pytorch_lightning.callbacks import ModelCheckpoint

    from tomocpt.training.averaging import WeightAveraging
    from tomocpt.training.helpers import _setup_callbacks

    cbs = _setup_callbacks(mc.train, model, ModelCheckpoint(monitor="val_loss"))
    return [c for c in cbs if isinstance(c, WeightAveraging)][0].ema_decay


@pytest.mark.parametrize("rule,batch,devices", [("none", 64, 1), ("sqrt", 64, 1), ("linear", 8, 1),
                                                ("sqrt", 16, 2), ("sqrt", 16, 1)])
def test_G8g_C4_training_uses_the_scaled_values(tmp_path, monkeypatch, rule, batch, devices):
    import tomocpt.training.hparam_scaling as S
    from tomocpt.training.helpers import _create_model

    monkeypatch.setattr(S, "training_devices", lambda config: devices)
    cfg = H4.v2_cfg(tmp_path, weight_averaging="ema", ema_decay=0.999, batch_size=batch)
    cfg.train.lr_scaling = rule
    k = batch * devices / 16
    want_lr = {"none": 4e-4, "sqrt": 4e-4 * math.sqrt(k), "linear": 4e-4 * k}[rule]
    want_ema = 0.999 if rule == "none" else 0.999 ** k
    with applied_main_config(cfg) as mc:
        model, _, _ = _create_model(mc.train)
        assert model.lr == pytest.approx(want_lr, rel=1e-12)
        assert _callbacks_ema(mc, model) == pytest.approx(want_ema, rel=1e-12)
        assert mc.train.n_epochs == cfg.train.n_epochs
        if k == 1:
            assert model.lr == 4e-4 and _callbacks_ema(mc, model) == 0.999
