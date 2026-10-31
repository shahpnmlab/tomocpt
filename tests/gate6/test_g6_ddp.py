"""G6-C7 (DDP rule): one training step with a batch whose filament channel is unavailable gives
every parameter a gradient (not ``None``), with both class heads (Gate 3d).

Class table: 1 "ribo" (particle, 120 A), 5 "microtubule" (filament, 250 A, declared with
``prepData.filament_coords``). The batch (CLAUDE.md 5.1) has particle targets from
``render_targets`` and a tube target from ``render_tubes`` in the filament channel, with
``class_available = [True, False]`` for every sample.
"""

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as G3
from gate6 import g6_helpers as H
from utils.tiny import applied_main_config

PARTICLES = ((1, "ribo", 120.0),)
FILAMENTS = ((5, "microtubule", 250.0),)


def _items():
    from tomocpt.labels.filaments import render_tubes

    items = []
    for b in range(2):
        it = G3.v2_item(700 + b, [12.0, 25.0], apix=10.0, available=[True, False], n_per_class=0)
        poly = H.line_xyz((4.0, 20.0 + 5 * b, 30.0), (60.0, 40.0, 34.0), n=15)
        tube = render_tubes(H.segments_from_polyline(poly, 1, 25.0), (64, 64, 64), 2)
        pts = [[32.0, 12.0 + 4 * b, 50.0, 0, 12.0]]
        it["target_data"] = torch.from_numpy(np.maximum(G3.render(pts, 2), tube))
        items.append(it)
    return items


@pytest.mark.parametrize("class_head", ["sigmoid", "softmax"])
def test_G6_C7_filament_channel_unavailable_every_parameter_gets_a_gradient(tmp_path, class_head):
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.networks.pickingModel import BasePickingModel

    cfg = H.v2_config_with_filaments(tmp_path, PARTICLES, FILAMENTS, class_head=class_head)
    table = class_table_from_config(cfg.prepData)
    assert [str(getattr(r.kind, "value", r.kind)) for r in table] == ["particle", "filament"]
    items = _items()
    assert items[0]["target_data"][1].max() > 0.99, "the filament channel holds a tube target"
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        loss, grads, _ = G3.first_step_loss_and_grads(model, items, tmp_path / "tb")
    assert np.isfinite(loss)
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, f"parameters without a gradient: {missing[:10]} ({len(missing)} in all)"
