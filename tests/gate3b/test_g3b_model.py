"""G3b-C4: ``BasePickingModel.training_step`` passes ``train.particles.cross_class_neg_weight``
to the v2 loss; a checkpoint config without the field loads with k = 0; v1 is untouched."""

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as H
from utils.tiny import applied_main_config

D_VOX_ABC = [15.0, 30.0, 20.0]


def _items():
    return [H.v2_item(60, D_VOX_ABC, apix=10.0, available=[True, True, False]),
            H.v2_item(61, D_VOX_ABC, apix=10.0, available=[True, True, True])]


def _want(model, items, k):
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    batch = H.collate(items)
    with torch.no_grad():
        logits = model(batch["input_data"])
        return float(multiclass_picking_loss(torch.sigmoid(logits), batch["target_data"],
                                             batch["class_available"], batch["class_d_vox"],
                                             checkpoint=False, cross_class_neg_weight=k))


def test_G3b_C4_config_field_default():
    from tomocpt.defaultConfigs.particles_config import ParticlesConfig
    from tomocpt.mainConfig import MainConfig

    assert ParticlesConfig().cross_class_neg_weight == 0.0
    assert MainConfig().train.particles.cross_class_neg_weight == 0.0


@pytest.mark.parametrize("k", [None, 0.0, 1.5])
def test_G3b_C4_training_step_uses_config_k(tmp_path, k):
    """The loss of the first v2 ``training_step`` equals ``multiclass_picking_loss`` with
    ``cross_class_neg_weight = train.particles.cross_class_neg_weight`` (``None`` = field left at
    its default), rel 1e-5, and k = 1.5 differs from k = 0 by more than 1 %."""
    from tomocpt.networks.pickingModel import BasePickingModel

    cfg = H.v2_config(tmp_path, H.CLASSES_ABC)
    if k is not None:
        cfg.train.particles.cross_class_neg_weight = k
    items = _items()
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        want = _want(model, items, 0.0 if k is None else k)
        want0 = _want(model, items, 0.0)
        loss, _, _ = H.first_step_loss_and_grads(model, items, tmp_path / "tb")
    assert loss == pytest.approx(want, rel=1e-5)
    if k == 1.5:
        assert abs(want - want0) > 0.01 * want0


def test_G3b_C4_v1_training_step_ignores_k(tmp_path):
    """Pipeline v1 (no ``class_available`` in the batch) keeps ``BasePickingModel.loss`` even with
    ``cross_class_neg_weight`` set."""
    from tomocpt.networks.pickingModel import BasePickingModel
    from utils.tiny import tiny_main_config

    cfg = tiny_main_config(tmp_path)
    cfg.train.particles.cross_class_neg_weight = 2.0
    it = H.v2_item(70, [16.0], apix=10.0)
    items = [{"input_data": it["input_data"], "target_data": it["target_data"]},
             {"input_data": it["input_data"].flip(-1), "target_data": it["target_data"].flip(-1)}]
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        batch = H.collate(items)
        with torch.no_grad():
            want = float(model.loss(torch.sigmoid(model(batch["input_data"])), batch["target_data"]))
        loss, _, _ = H.first_step_loss_and_grads(model, items, tmp_path / "tb")
    assert loss == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("field", ["missing", "1.5"])
def test_G3b_C4_checkpoint_without_field_loads_with_k0(tmp_path, field):
    """A v2 checkpoint whose pickled config lacks ``cross_class_neg_weight`` (written before
    Gate 3b; the field is removed from the instance) loads with k = 0 and trains with k = 0.
    Control: the same checkpoint with the field set to 1.5 trains with 1.5."""
    from tomocpt.configManager.configManager import safe_getattr
    from tomocpt.networks.pickingModel import BasePickingModel

    src = H.build_v2_checkpoint(tmp_path / "src", H.CLASSES_ABC, cond=False)
    ckpt = torch.load(src["path"], map_location="cpu", weights_only=False)
    particles = ckpt["hyper_parameters"]["config"].train.particles
    particles.__dict__.pop("cross_class_neg_weight", None)
    if field == "1.5":
        particles.cross_class_neg_weight = 1.5
    else:
        assert "cross_class_neg_weight" not in vars(particles)
    path = tmp_path / "edited.ckpt"
    torch.save(ckpt, path)

    model = BasePickingModel.load_from_checkpoint(path, map_location="cpu")
    k = 0.0 if field == "missing" else 1.5
    assert float(safe_getattr(model.hparams.config.train.particles, "cross_class_neg_weight", -1.0)) == k
    items = _items()
    want = _want(model, items, k)
    loss, _, _ = H.first_step_loss_and_grads(model, items, tmp_path / "tb")
    assert loss == pytest.approx(want, rel=1e-5)
