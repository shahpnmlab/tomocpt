"""G5-C13 (one SSL v2 training step) and G5-C14 (SSL v2 -> Gate 3 picker transfer).

The v2 ``SelfSupervisedModel`` is trained on ``torch.utils.data.default_collate`` of
``make_training_pair`` dicts (README.md). The fits run once per session (``ssl_v2_fit``).
"""

import pytest
import torch

from gate3 import g3_helpers as H3
from gate5 import g5_helpers as H
from utils.tiny import applied_main_config


# --------------------------------------------------------------------------------------
# G5-C13
# --------------------------------------------------------------------------------------

def _check_every_parameter_has_a_gradient(fit):
    grads = fit["grads"]
    assert grads, "no backward pass was observed"
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, f"parameters without a gradient: {missing[:10]} ({len(missing)} in total)"
    bad = sorted(n for n, g in grads.items() if not torch.isfinite(g).all())
    assert not bad, f"non-finite gradients: {bad[:10]}"


def test_G5_C13_v2_step_all_tasks(ssl_v2_fit):
    from tomocpt.networks.swinunetr import ScaleConditionedSwinUNETR

    fit = ssl_v2_fit("default")
    model = fit["model"]
    _check_every_parameter_has_a_gradient(fit)
    assert isinstance(model.model, ScaleConditionedSwinUNETR)
    assert getattr(model, "rotation_head", None) is None, "SSL v2 has no rotation head"
    for cls in H.head_classes():
        assert any(isinstance(m, cls) for m in model.modules()), f"no {cls.__name__} in the v2 model"
    assert "val_loss_smooth" in fit["metrics"]


def test_G5_C13_disabled_tasks_have_no_modules(ssl_v2_fit):
    fit = ssl_v2_fit("recon_only")
    model = fit["model"]
    _check_every_parameter_has_a_gradient(fit)
    assert getattr(model, "rotation_head", None) is None
    for cls in H.head_classes():
        assert not any(isinstance(m, cls) for m in model.modules()), \
            f"{cls.__name__} exists although its loss weight is 0"
    assert "val_loss_smooth" in fit["metrics"]


# --------------------------------------------------------------------------------------
# G5-C14
# --------------------------------------------------------------------------------------

CLASSES = {1: ((1, "A", 150.0),),
           4: ((1, "A", 150.0), (2, "B", 300.0), (3, "C", 200.0), (4, "D", 120.0))}


@pytest.mark.parametrize("n_classes", [1, 4])
def test_G5_C14_ssl_v2_checkpoint_into_picker(tmp_path, ssl_v2_fit, n_classes):
    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.training.helpers import _create_model

    fit = ssl_v2_fit("default")
    ssl_sd = torch.load(fit["path"], map_location="cpu", weights_only=False)["state_dict"]

    cfg = H3.v2_config(tmp_path, CLASSES[n_classes], cond=True)
    with applied_main_config(cfg) as g:
        g.train.resume_from = None
        g.train.fine_tune_from = fit["path"]
        g.train.use_distillation = False
        model, resume, _ = _create_model(g.train)
    assert isinstance(model, BasePickingModel)
    assert resume is None
    pick_sd = model.model.state_dict()
    loaded = {"model." + k for k in pick_sd}

    # Backbone, decoder and cond all come from the checkpoint: nothing starts fresh ...
    fresh = sorted(k for k in loaded if k not in ssl_sd)
    assert not fresh, f"picker tensors not in the SSL checkpoint: {fresh[:10]}"
    # ... and only the SSL heads are left behind.
    dropped = set(ssl_sd) - loaded
    assert dropped == fit["head_keys"], (sorted(dropped - fit["head_keys"])[:10],
                                         sorted(fit["head_keys"] - dropped)[:10])
    assert any(k.startswith("cond.") for k in pick_sd)

    for k, v in pick_sd.items():
        src = ssl_sd["model." + k]
        if k.startswith("out."):
            # Gate 3 rule: the one-channel output is copied into every class channel.
            assert v.shape[0] == n_classes and src.shape[0] == 1, k
            for c in range(n_classes):
                assert torch.equal(v[c], src[0]), (k, c)
        else:
            assert torch.equal(v, src), k
