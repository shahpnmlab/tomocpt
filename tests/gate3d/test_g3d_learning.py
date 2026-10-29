"""G3d-C7 (slow, CUDA): the Gate 3 C13 scenario (solid vs hollow spheres of the same size, plus a
small and a large class) trained with ``train.particles.class_head = "softmax"`` and
``cross_class_neg_weight = 0`` on seeds 0 and 1 (seed 1 fails with sigmoid at k = 0, Gate 3b README).

Reuses ``gate3.test_g3_learning`` unchanged (data, ``train_picker``, ``_c13_eval``, which predicts
through ``BasePickingModel.predict_step`` without ``classes``, so with the softmax over all four
channels). The config is set by wrapping ``gate3.g3_helpers.v2_config``, as Gate 3b does.
"""

import json

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as H
from gate3.test_g3_learning import C13_LR, C13_STEPS, _c13_data, _c13_eval, _need_cuda, c13_classes, train_picker

pytestmark = pytest.mark.slow

SEEDS = (0, 1)
F1_MIN = 0.9
CROSS_MAX = 0.05


def run_c13_softmax(tmp, seed, monkeypatch):
    base = H.v2_config

    def v2_config_softmax(*a, **kw):
        cfg = base(*a, **kw)
        cfg.train.particles.class_head = "softmax"
        cfg.train.particles.cross_class_neg_weight = 0.0
        return cfg

    # spy on the training-step loss calls: k = 0, and the probabilities are a softmax with an
    # implicit background (per-voxel sum over channels <= 1; a sigmoid head with four channels
    # near 0.5 at the start of training sums to about 2)
    import tomocpt.networks.pickingModel as pm

    loss_fn, seen = pm.multiclass_picking_loss, {"k": set(), "max_sum": 0.0, "n": 0}

    def spy(*a, **kw):
        if torch.is_grad_enabled():
            seen["k"].add(float(kw.get("cross_class_neg_weight", 0.0)))
            y_pred = a[0] if a else kw["y_pred"]
            seen["max_sum"] = max(seen["max_sum"], float(y_pred.detach().float().sum(1).max()))
            seen["n"] += 1
        return loss_fn(*a, **kw)

    monkeypatch.setattr(H, "v2_config", v2_config_softmax)
    monkeypatch.setattr(pm, "multiclass_picking_loss", spy)
    try:
        train, held = _c13_data(seed)
        model = train_picker(tmp, c13_classes(), False, train, C13_STEPS, seed, lr=C13_LR)
        res = _c13_eval(model, held)
    finally:
        monkeypatch.setattr(H, "v2_config", base)
        monkeypatch.setattr(pm, "multiclass_picking_loss", loss_fn)
    assert seen["n"] > 0 and seen["k"] == {0.0}, seen
    assert seen["max_sum"] <= 1.0 + 1e-3, seen
    print(f"\nG3D_RESULT C13 softmax seed={seed} " + json.dumps(res, sort_keys=True))
    return res


def test_G3d_C7_softmax_fixes_c13_without_k(tmp_path, monkeypatch):
    """Seeds 0 and 1: every class F1 >= 0.9 and mean cross-pick rate (over seeds and both
    directions, as Gate 3b) <= 0.05. Two tiny-model trainings of about 4.5 min each."""
    _need_cuda()
    results = {s: run_c13_softmax(tmp_path / f"s{s}", s, monkeypatch) for s in SEEDS}
    mean_cross = float(np.mean([v for r in results.values() for v in r["cross"].values()]))
    summary = {"mean_cross": mean_cross, "f1": {s: r["f1"] for s, r in results.items()}}
    print("\nG3D_SUMMARY " + json.dumps(summary, sort_keys=True))
    for s, r in results.items():
        assert all(v >= F1_MIN for v in r["f1"].values()), (s, summary)
    assert mean_cross <= CROSS_MAX, summary
