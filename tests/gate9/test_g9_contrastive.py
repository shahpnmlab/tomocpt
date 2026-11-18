"""G9-C1 (contrastive part), G9-C3: contrastive weight ramp
(``train.discriminative.contrastive_warmup_steps``), and the bank age limit wired into the model."""

import pytest
import torch

from gate3 import g3_helpers as G3
from utils.tiny import applied_main_config

CLASSES = ((1, "A", 150.0), (2, "B", 300.0))
D_VOX = (12.0, 18.0)
W = 0.5


def _cfg(tmp_path, **disc):
    cfg = G3.v2_config(tmp_path, CLASSES, cond=False)
    cfg.train.discriminative.contrastive_weight = W
    for k, v in disc.items():
        setattr(cfg.train.discriminative, k, v)
    return cfg


def _model(tmp_path, **disc):
    from tomocpt.networks.pickingModel import BasePickingModel

    with applied_main_config(_cfg(tmp_path, **disc)):
        torch.manual_seed(0)
        return BasePickingModel(lr=4e-4)


def test_G9_C1_defaults():
    from tomocpt.defaultConfigs.discriminative_config import DiscriminativeConfig

    d = DiscriminativeConfig()
    assert d.contrastive_warmup_steps == 0
    assert d.bank_max_age == 0


def test_G9_C1_default_weight_is_constant(tmp_path):
    model = _model(tmp_path)
    for step in (0, 1, 7, 10_000):
        assert model.contrastive_weight_at(step) == W
    assert model.contrastive_bank.max_age == 0


def test_G9_C3_ramp_values(tmp_path):
    model = _model(tmp_path, contrastive_warmup_steps=10)
    assert model.contrastive_weight_at(0) == 0.0
    assert model.contrastive_weight_at(5) == pytest.approx(W * 0.5)
    assert model.contrastive_weight_at(9) == pytest.approx(W * 0.9)
    assert model.contrastive_weight_at(10) == pytest.approx(W)
    assert model.contrastive_weight_at(25) == pytest.approx(W)


def test_G9_C3_bank_max_age_reaches_the_bank(tmp_path):
    model = _model(tmp_path, bank_max_age=7)
    assert model.contrastive_bank.max_age == 7


class _GradProbe:
    """Lightning callback: records the contrastive head's gradients and bank fill before the step."""

    def __new__(cls):
        import pytorch_lightning as pl

        class Probe(pl.Callback):
            def __init__(self):
                self.grads, self.bank_count = None, None

            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                self.grads = [None if p.grad is None else p.grad.detach().clone()
                              for p in pl_module.contrastive_head.parameters()]
                self.bank_count = int(pl_module.contrastive_bank.count.sum())

        return Probe()


def _fit_one_step(tmp_path, **disc):
    cfg = _cfg(tmp_path, **disc)
    with applied_main_config(cfg):
        from tomocpt.networks.pickingModel import BasePickingModel

        torch.manual_seed(0)
        model = BasePickingModel(lr=4e-4)
        probe = _GradProbe()
        items = [G3.v2_item(s, D_VOX) for s in range(2)]
        trainer = G3.cpu_trainer(tmp_path / "tb", callbacks=[probe], max_steps=1, limit_val_batches=0)
        trainer.fit(model, datamodule=G3.datamodule(items))
    return probe


@pytest.mark.parametrize("ramp", [0, 10])
def test_G9_C3_head_gets_gradients_at_step_0(tmp_path, ramp):
    probe = _fit_one_step(tmp_path, contrastive_warmup_steps=ramp)
    assert probe.grads and all(g is not None for g in probe.grads)  # DDP: every head parameter has a grad
    total = sum(float(g.abs().sum()) for g in probe.grads)
    if ramp:
        assert total == 0.0  # weight 0 at step 0
    else:
        assert total > 0.0
    assert probe.bank_count > 0  # anchors still go into the bank during the ramp
