"""G9-C1 (LR part), G9-C2: linear LR warmup (``train.lr_warmup_steps``)."""

import pytest
import torch

from gate3 import g3_helpers as G3
from utils.tiny import applied_main_config

CLASSES = ((1, "A", 150.0), (2, "B", 300.0))
N = 10


def _cfg(tmp_path, **train):
    cfg = G3.v2_config(tmp_path, CLASSES, cond=False)
    for k, v in train.items():
        setattr(cfg.train, k, v)
    return cfg


def _model_conf(tmp_path, train_continue=None, **train):
    from tomocpt.networks.pickingModel import BasePickingModel

    with applied_main_config(_cfg(tmp_path, **train)):
        torch.manual_seed(0)
        model = BasePickingModel(lr=4e-4, train_continue=train_continue)
        return model, model.configure_optimizers()


def _set_step(monkeypatch, model, step):
    monkeypatch.setattr(type(model), "global_step", property(lambda self: step))


def _stepped_lrs(monkeypatch, model, opt, step):
    """Group lrs seen inside ``optimizer.step`` at ``global_step = step``; checks they are restored."""
    seen = []
    orig = opt.step

    def recorder(closure=None):
        seen.append([g["lr"] for g in opt.param_groups])
        if closure is not None:
            closure()

    opt.step = recorder
    _set_step(monkeypatch, model, step)
    nominal = [g["lr"] for g in opt.param_groups]
    try:
        model.optimizer_step(0, 0, opt, lambda: None)
    finally:
        opt.step = orig
    assert [g["lr"] for g in opt.param_groups] == nominal  # nominal values restored after the step
    assert len(seen) == 1
    return seen[0], nominal


def test_G9_C1_default_is_off():
    from tomocpt.defaultConfigs.train_config import TrainConfig

    assert TrainConfig().lr_warmup_steps == 0


@pytest.mark.parametrize("train_continue", [None, "finetune"])
def test_G9_C1_default_lr_unchanged(tmp_path, monkeypatch, train_continue):
    model, conf = _model_conf(tmp_path, train_continue=train_continue)
    for step in (0, 3, 100):
        seen, nominal = _stepped_lrs(monkeypatch, model, conf["optimizer"], step)
        assert seen == nominal


def test_G9_C2_factor_helper():
    from tomocpt.networks.baseModel import linear_warmup_factor

    assert linear_warmup_factor(0, 0) == 1.0
    assert linear_warmup_factor(5, 0) == 1.0
    assert linear_warmup_factor(0, N) == pytest.approx(1 / N)
    assert linear_warmup_factor(N // 2, N) == pytest.approx((N // 2 + 1) / N)
    assert linear_warmup_factor(N - 1, N) == 1.0
    assert linear_warmup_factor(N, N) == 1.0
    assert linear_warmup_factor(N + 5, N) == 1.0


@pytest.mark.parametrize("train_continue", [None, "finetune"])
def test_G9_C2_warmup_values(tmp_path, monkeypatch, train_continue):
    model, conf = _model_conf(tmp_path, train_continue=train_continue, lr_warmup_steps=N)
    opt = conf["optimizer"]
    base = [g["lr"] for g in opt.param_groups]
    if train_continue is not None:
        assert len(set(base)) > 1  # differential LR groups (not vacuous)
    for step, factor in ((0, 1 / N), (N // 2, (N // 2 + 1) / N), (N, 1.0), (N + 5, 1.0)):
        seen, nominal = _stepped_lrs(monkeypatch, model, opt, step)
        assert nominal == base
        assert seen == pytest.approx([b * factor for b in base]), step


def test_G9_C2_warmup_follows_lr_scaling(tmp_path, monkeypatch):
    """The base of the warmup is the (lr_scaling-scaled) lr the model was built with."""
    model, conf = _model_conf(tmp_path, lr_warmup_steps=N)
    opt = conf["optimizer"]
    assert opt.param_groups[0]["lr"] == pytest.approx(model.lr)
    seen, _ = _stepped_lrs(monkeypatch, model, opt, 1)
    assert seen[0] == pytest.approx(model.lr * 2 / N)


def test_G9_C2_plateau_scheduler_still_reduces(tmp_path, monkeypatch):
    model, conf = _model_conf(tmp_path, lr_warmup_steps=N)
    opt, sched = conf["optimizer"], conf["lr_scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    base = opt.param_groups[0]["lr"]
    # during warmup, the scheduler sees the nominal lr
    _stepped_lrs(monkeypatch, model, opt, 2)
    assert opt.param_groups[0]["lr"] == base
    # after warmup: no improvement for patience + 2 epochs -> the lr halves
    _stepped_lrs(monkeypatch, model, opt, N + 1)
    for _ in range(sched.patience + 2):
        sched.step(1.0)
    reduced = opt.param_groups[0]["lr"]
    assert reduced == pytest.approx(base * 0.5)
    seen, _ = _stepped_lrs(monkeypatch, model, opt, N + 5)
    assert seen[0] == pytest.approx(reduced)
    # a reduction during warmup composes multiplicatively
    model2, conf2 = _model_conf(tmp_path, lr_warmup_steps=N)
    opt2, sched2 = conf2["optimizer"], conf2["lr_scheduler"]
    for _ in range(sched2.patience + 2):
        sched2.step(1.0)
    seen, _ = _stepped_lrs(monkeypatch, model2, opt2, 4)
    assert seen[0] == pytest.approx(base * 0.5 * 5 / N)


def test_G9_C2_resume_continues_from_global_step(tmp_path, monkeypatch):
    """No warmup state of its own: the factor depends only on the (restored) global step."""
    model, conf = _model_conf(tmp_path, lr_warmup_steps=N)
    opt = conf["optimizer"]
    sd = opt.state_dict()
    model2, conf2 = _model_conf(tmp_path, lr_warmup_steps=N)
    conf2["optimizer"].load_state_dict(sd)
    seen, _ = _stepped_lrs(monkeypatch, model2, conf2["optimizer"], 6)
    assert seen[0] == pytest.approx(opt.param_groups[0]["lr"] * 7 / N)


def test_G9_C2_schedule_free_raises(tmp_path):
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        _model_conf(tmp_path, lr_warmup_steps=N, schedule_free=True)


def test_G9_C2_lightning_fit_scales_the_step(tmp_path):
    """Through Lightning's LightningOptimizer: inside the step (``on_before_optimizer_step`` runs
    after the closure, inside ``optimizer.step``) the lr is warmed up; after the step it is nominal."""
    import pytorch_lightning as pl

    class Probe(pl.Callback):
        def __init__(self):
            self.lrs, self.inside = [], []

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            self.inside.append([g["lr"] for g in trainer.optimizers[0].param_groups])

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.lrs.append([g["lr"] for g in trainer.optimizers[0].param_groups])

    D_VOX = (12.0, 18.0)
    cfg = _cfg(tmp_path, lr_warmup_steps=N)
    with applied_main_config(cfg):
        from tomocpt.networks.pickingModel import BasePickingModel

        torch.manual_seed(0)
        model = BasePickingModel(lr=4e-4)
        probe = Probe()
        items = [G3.v2_item(s, D_VOX) for s in range(4)]
        trainer = G3.cpu_trainer(tmp_path / "tb", callbacks=[probe], max_steps=2, limit_train_batches=2,
                                 limit_val_batches=1)
        trainer.fit(model, datamodule=G3.datamodule(items))
    assert trainer.global_step == 2
    assert probe.lrs == [[4e-4], [4e-4]]
    assert probe.inside == [[pytest.approx(4e-4 / N)], [pytest.approx(4e-4 * 2 / N)]]
