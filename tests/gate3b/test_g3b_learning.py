"""G3b-C5 (slow, CUDA): the Gate 3 C13 scenario (solid vs hollow spheres of the same size, plus a
small and a large class, scale conditioning off) trained with the recommended
``train.particles.cross_class_neg_weight`` and with k = 0 on seeds 0-2.

Reuses ``gate3.test_g3_learning`` unchanged (data, training through ``BasePickingModel.training_step``,
evaluation); k is set by wrapping ``gate3.g3_helpers.v2_config``, the config builder that
``train_picker`` calls.

Environment:
* ``TOMOCPT_G3B_K`` overrides the recommended k (calibration only).
* ``TOMOCPT_G3B_SKIP_K0=1`` uses the k = 0 numbers recorded at Gate 3 (README) instead of
  training the three k = 0 models (saves ~12 min; the comparison is then against recorded runs).
"""

import json
import os

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as H
from gate3.test_g3_learning import C13_LR, C13_STEPS, _c13_data, _c13_eval, _need_cuda, c13_classes, train_picker

pytestmark = pytest.mark.slow

#: recommended k (calibrated on the reference implementation, see README)
K_RECOMMENDED = 0.25
K = float(os.environ.get("TOMOCPT_G3B_K", K_RECOMMENDED))
SEEDS = (0, 1, 2)
FAILING_K0_SEED = 1

#: Gate 3 README, C13 seeds 0-2 at k = 0 (reference implementation, 1800 steps)
K0_RECORDED = {
    0: {"f1": {"small": 0.847, "solid": 1.0, "hollow": 1.0, "large": 1.0},
        "cross": {"solid_in_hollow": 0.0, "hollow_in_solid": 0.0}},
    1: {"f1": {"small": 0.857, "solid": 0.571, "hollow": 1.0, "large": 1.0},
        "cross": {"solid_in_hollow": 0.0, "hollow_in_solid": 1.0}},
    2: {"f1": {"small": 1.0, "solid": 0.842, "hollow": 1.0, "large": 1.0},
        "cross": {"solid_in_hollow": 0.0, "hollow_in_solid": 0.0}},
}


def c13_passes(res):
    """The Gate 3 C13 thresholds: F1 >= 0.8 for every class, cross-pick < 0.15 both ways."""
    return all(v >= 0.8 for v in res["f1"].values()) and all(v < 0.15 for v in res["cross"].values())


def mean_cross(results):
    return float(np.mean([v for r in results.values() for v in r["cross"].values()]))


def run_c13(tmp, seed, k, monkeypatch):
    base = H.v2_config

    def v2_config_k(*a, **kw):
        cfg = base(*a, **kw)
        cfg.train.particles.cross_class_neg_weight = k
        return cfg

    # spy: the k that training-step loss calls (grad enabled) receive
    import tomocpt.networks.pickingModel as pm

    loss_fn, seen = pm.multiclass_picking_loss, set()

    def spy(*a, **kw):
        if torch.is_grad_enabled():
            seen.add(float(kw.get("cross_class_neg_weight", 0.0)))
        return loss_fn(*a, **kw)

    monkeypatch.setattr(H, "v2_config", v2_config_k)
    monkeypatch.setattr(pm, "multiclass_picking_loss", spy)
    try:
        train, held = _c13_data(seed)
        model = train_picker(tmp, c13_classes(), False, train, C13_STEPS, seed, lr=C13_LR)
        res = _c13_eval(model, held)
    finally:
        monkeypatch.setattr(H, "v2_config", base)
        monkeypatch.setattr(pm, "multiclass_picking_loss", loss_fn)
    assert seen == {float(k)}, seen
    print(f"\nG3B_RESULT C13 k={k} seed={seed} " + json.dumps(res, sort_keys=True))
    return res


def test_G3b_C5_cross_class_negatives_fix_c13(tmp_path, monkeypatch):
    """With k = K_RECOMMENDED: mean cross-pick rate over seeds 0-2 (both directions) lower than
    with k = 0; the C13 thresholds hold on seed 1 (where k = 0 fails) and on at least 2 of 3 seeds.
    Calibration (scratch reference implementation, README): k = 0.25 and k = 1.0 both pass all three
    seeds with cross-pick 0.0 in both directions and every F1 >= 0.973 (thresholds 0.8 / 0.15); k = 0
    re-run on seed 1 fails again (hollow_in_solid 0.875, solid F1 0.593). Only seeds 0-2 were run, so
    the 9-of-10 rule is not demonstrated; the margin on the asserted quantities is large."""
    _need_cuda()
    assert K > 0
    with_k, without_k = {}, {}
    for seed in SEEDS:
        with_k[seed] = run_c13(tmp_path / f"k{K}_s{seed}", seed, K, monkeypatch)
        if os.environ.get("TOMOCPT_G3B_SKIP_K0") == "1":
            without_k[seed] = K0_RECORDED[seed]
        else:
            without_k[seed] = run_c13(tmp_path / f"k0_s{seed}", seed, 0.0, monkeypatch)
    summary = {"k": K, "mean_cross_k": mean_cross(with_k), "mean_cross_k0": mean_cross(without_k),
               "pass_k": {s: c13_passes(r) for s, r in with_k.items()},
               "pass_k0": {s: c13_passes(r) for s, r in without_k.items()}}
    print("\nG3B_SUMMARY " + json.dumps(summary, sort_keys=True))
    assert summary["mean_cross_k"] < summary["mean_cross_k0"], summary
    assert c13_passes(with_k[FAILING_K0_SEED]), (with_k[FAILING_K0_SEED], summary)
    assert sum(summary["pass_k"].values()) >= 2, summary
