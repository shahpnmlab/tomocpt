"""G4f-C1 / C5: centroid (core) weight in the v2 loss."""

import pytest
import torch
from torch import nn

from gate3 import g3_helpers as G3


def _batch(seed=0):
    torch.manual_seed(seed)
    items = [G3.v2_item(500 + i, [15.0, 30.0], apix=10.0) for i in range(2)]
    y = torch.stack([it["target_data"] for it in items])
    avail = torch.stack([it["class_available"] for it in items])
    d = torch.stack([it["class_d_vox"] for it in items])
    p = torch.rand_like(y)
    return p, y, avail, d


def test_G4f_C1_defaults():
    from tomocpt.defaultConfigs.particles_config import ParticlesConfig

    pc = ParticlesConfig()
    assert pc.core_weight == 0.0 and pc.core_power == 4.0 and pc.offset_weight == 0.0


@pytest.mark.parametrize("value_loss", ["huber", "focal"])
def test_G4f_C5_core_weight_zero_is_bit_identical(value_loss):
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    p, y, a, d = _batch()
    ref = multiclass_picking_loss(p, y, a, d, checkpoint=False, value_loss=value_loss)
    got = multiclass_picking_loss(p, y, a, d, checkpoint=False, value_loss=value_loss, core_weight=0.0)
    assert torch.equal(ref, got)


def test_G4f_C5_core_weight_by_hand():
    from tomocpt.training.multiclass_loss import _value_weights, multiclass_picking_loss

    p, y, a, d = _batch(1)
    k, pw = 3.0, 4.0
    base = multiclass_picking_loss(p, y, a, d, checkpoint=False)
    got = multiclass_picking_loss(p, y, a, d, checkpoint=False, core_weight=k, core_power=pw)
    w = _value_weights(y, a, 0.0)
    extra = (w * k * y ** pw * nn.functional.huber_loss(p, y, reduction="none")).flatten(2).mean(2)  # [B, C]
    want = base + extra.sum() / a.sum()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)
    # only the second channel is a particle channel: the first keeps its plain weight
    got1 = multiclass_picking_loss(p, y, a, d, checkpoint=False, core_weight=k, core_power=pw,
                                   core_channels=[False, True])
    want1 = base + extra[:, 1].sum() / a.sum()
    torch.testing.assert_close(got1, want1, rtol=1e-5, atol=1e-7)
