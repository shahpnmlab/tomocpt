"""Shared by the Gate 7 lock golden generator (``make_goldens_gate7.py``, run on the Gate 6 code)
and the G7-C1 / G7-C7 lock tests (run on the code under test).

``lock_record(root, case)`` builds a tiny pipeline-v2 ``BasePickingModel`` with deterministic
weights for one case of :data:`CASES`, and returns a JSON-able record:

* ``state_dict_keys``: sorted keys of the LightningModule ``state_dict`` (no new module or buffer);
* ``logits_sha256`` / ``logits_sum``: ``forward`` on a fixed input in eval mode (bit-exact hash);
* ``loss_hex`` / ``loss``: the first training-step loss of a one-batch CPU ``fit`` (fp32);
* ``grads_abs_sum``: float64 sum of ``|grad|`` over every parameter after that backward (CPU
  backward is not bit-reproducible between runs; compared with rtol 1e-5).

Class table: 1 "ribo" (particle, 120 A), 5 "mt" (filament, 250 A, declared with
``prepData.filament_coords``). Batch: two ``g3_helpers.v2_item`` samples with particle targets
and a straight tube in the filament channel (``render_tubes``), every class available.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np

PARTICLES = ((1, "ribo", 120.0),)
FILAMENTS = ((5, "mt", 250.0),)
#: name -> (class_head, scale_conditioning)
CASES = {"sigmoid_uncond": ("sigmoid", False), "softmax_cond": ("softmax", True)}


def seed_all(seed: int = 0):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _sha(t) -> str:
    a = np.ascontiguousarray(t.detach().cpu().numpy())
    return hashlib.sha256(a.tobytes()).hexdigest()


def lock_items():
    """Two pipeline-v2 samples (no batch dimension), particle + tube targets."""
    import torch
    from gate3 import g3_helpers as G3
    from gate6 import g6_helpers as H6

    from tomocpt.labels.filaments import render_tubes

    items = []
    for b in range(2):
        it = G3.v2_item(900 + b, [12.0, 25.0], apix=10.0, n_per_class=0)
        pts = [[30.3, 14.6 + 6 * b, 44.2, 0, 12.0], [44.7, 40.1, 15.5 - 3 * b, 0, 12.0]]
        poly = H6.line_xyz((4.0, 22.0 + 5 * b, 30.0), (60.0, 44.0, 36.0), n=15)
        tube = render_tubes(H6.segments_from_polyline(poly, 1, 25.0), (64, 64, 64), 2)
        it["target_data"] = torch.from_numpy(np.maximum(G3.render(pts, 2), tube))
        items.append(it)
    return items


def lock_config(root: Path, case: str):
    from gate6 import g6_helpers as H6

    head, cond = CASES[case]
    return H6.v2_config_with_filaments(root, PARTICLES, FILAMENTS, cond=cond, class_head=head)


def lock_record(root: Path, case: str, cfg=None) -> dict:
    """Record of one case (see module docstring). ``cfg`` overrides :func:`lock_config`
    (the tests pass a config with Gate 7 options set)."""
    import torch
    from gate3 import g3_helpers as G3
    from utils.tiny import applied_main_config

    from tomocpt.networks.pickingModel import BasePickingModel

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = lock_config(root, case) if cfg is None else cfg
    _, cond = CASES[case]
    with applied_main_config(cfg):
        seed_all(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        keys = sorted(model.state_dict().keys())
        x = G3.fixed_input(1, seed=21)
        model.eval()
        with torch.no_grad():
            logits = model(x, apix=G3.apix_tensor(10.0) if cond else None)
        model.train()
        seed_all(0)
        loss, grads, trainer = G3.first_step_loss_and_grads(model, lock_items(), root / "tb")
    gsum = sum(float(g.double().abs().sum()) for g in grads.values() if g is not None)
    return {
        "state_dict_keys": keys,
        "logits_sha256": _sha(logits),
        "logits_sum": float(logits.double().sum()),
        "loss": float(loss),
        "loss_hex": float(loss).hex(),
        "grads_abs_sum": gsum,
        "_model": model,
        "_grads": grads,
        "_trainer": trainer,
    }


def public(record: dict) -> dict:
    return {k: v for k, v in record.items() if not k.startswith("_")}
