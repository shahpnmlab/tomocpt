"""G3d-C4, G3d-C5, G3d-C6: softmax loss through ``BasePickingModel.training_step``, one softmax
step, checkpoints and ``predict_step``.

``train.particles.class_head`` is set inside the tests, so the module imports on code without it.
"""

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as H
from utils.tiny import applied_main_config

CLASSES_1 = ((1, "A", 150.0),)
D_VOX_ABC = [15.0, 30.0, 20.0]


def _cfg(tmp_path, classes, head):
    cfg = H.v2_config(tmp_path, classes)
    if head is not None:
        cfg.train.particles.class_head = head
    return cfg


def _model(cfg, seed=0):
    from tomocpt.networks.pickingModel import BasePickingModel

    torch.manual_seed(seed)
    return BasePickingModel(lr=1e-3, train_continue=None)


def _expected(model, items, renorm_over="all"):
    """Loss of the spec: ``multiclass_picking_loss`` on ``class_probabilities(logits, "softmax",
    class_available)`` with targets ``y / max(1, sum_c y_c)`` (sum over all channels, or over the
    available channels only)."""
    from tomocpt.training.multiclass_loss import class_probabilities, multiclass_picking_loss

    b = H.collate(items)
    av = b["class_available"]
    y = b["target_data"]
    if renorm_over == "all":
        s = y.sum(1, keepdim=True)
    else:
        s = (y * av.view(*av.shape, 1, 1, 1)).sum(1, keepdim=True)
    with torch.no_grad():
        p = class_probabilities(model(b["input_data"]), "softmax", av)
        return float(multiclass_picking_loss(p, y / s.clamp(min=1.0), av, b["class_d_vox"], checkpoint=False))


def _overlap_items():
    """C = 3, all available; classes 0 and 1 overlap strongly (targets sum > 1)."""
    out = []
    for i, shift in enumerate((0.0, 1.0)):
        pts = [[32, 32, 32, 0, 16.0], [32, 32, 32 + shift, 1, 16.0], [14, 14, 14, 2, 12.0],
               [48, 20, 44, 0, 16.0], [48, 20, 44 + shift, 1, 16.0], [16, 46, 20, 2, 12.0],
               [16, 46, 20 + shift, 1, 16.0]]
        y = torch.from_numpy(H.render(pts, 3))
        x = np.clip(np.random.default_rng(80 + i).standard_normal((1, 64, 64, 64)), -3, 3).astype(np.float32)
        out.append({"input_data": torch.from_numpy(x), "target_data": y,
                    "class_available": torch.ones(3, dtype=torch.bool),
                    "class_d_vox": torch.tensor([16.0, 16.0, 12.0]),
                    "apix_work": torch.tensor([10.0]), "dataset_id": torch.tensor(0)})
    return out


# --------------------------------------------------------------------------------------
# G3d-C4
# --------------------------------------------------------------------------------------

def test_G3d_C4_one_class_softmax_loss_equals_sigmoid_loss(tmp_path):
    """C = 1, all available: the first ``training_step`` loss of a softmax model equals that of a
    sigmoid model with the same weights (rtol 1e-6), and the direct loss on
    ``class_probabilities(..., "softmax")`` equals the loss on ``sigmoid`` (rtol 1e-6)."""
    from tomocpt.training.multiclass_loss import class_probabilities, multiclass_picking_loss

    items = [H.v2_item(90, [15.0]), H.v2_item(91, [15.0])]
    losses = {}
    for head in ("sigmoid", "softmax"):
        with applied_main_config(_cfg(tmp_path / head, CLASSES_1, head)):
            model = _model(None)
            b = H.collate(items)
            with torch.no_grad():
                lg = model(b["input_data"])
                args = (b["target_data"], b["class_available"], b["class_d_vox"])
                l_soft = float(multiclass_picking_loss(class_probabilities(lg, "softmax", b["class_available"]),
                                                       *args, checkpoint=False))
                l_sig = float(multiclass_picking_loss(torch.sigmoid(lg), *args, checkpoint=False))
            assert l_soft == pytest.approx(l_sig, rel=1e-6)
            losses[head], _, _ = H.first_step_loss_and_grads(model, items, tmp_path / f"tb_{head}")
    assert losses["softmax"] == pytest.approx(losses["sigmoid"], rel=1e-6)


def test_G3d_C4_targets_above_one_are_renormalised(tmp_path):
    """Overlapping targets (sum > 1): the softmax ``training_step`` loss is finite and equals the
    loss on ``y / max(1, sum_c y_c)`` (rel 1e-5), which differs from the loss on raw targets."""
    from tomocpt.training.multiclass_loss import class_probabilities, multiclass_picking_loss

    items = _overlap_items()
    b = H.collate(items)
    assert float(b["target_data"].sum(1).max()) > 1.3
    with applied_main_config(_cfg(tmp_path, H.CLASSES_ABC, "softmax")):
        model = _model(None)
        want = _expected(model, items)
        with torch.no_grad():
            p = class_probabilities(model(b["input_data"]), "softmax", b["class_available"])
            raw = float(multiclass_picking_loss(p, b["target_data"], b["class_available"], b["class_d_vox"],
                                                checkpoint=False))
        loss, _, _ = H.first_step_loss_and_grads(model, items, tmp_path / "tb")
    assert np.isfinite(loss)
    assert loss == pytest.approx(want, rel=1e-5)
    assert abs(raw - want) > 2e-4 * abs(want)  # about 4.7e-4 here; tolerance on the match is 1e-5


# --------------------------------------------------------------------------------------
# G3d-C5
# --------------------------------------------------------------------------------------

def test_G3d_C5_softmax_training_step_gradients(tmp_path):
    """One softmax step on a v2 batch with class 2 unavailable in every sample and class 1 in one:
    finite loss equal to the spec loss, and every trainable parameter has a gradient."""
    items = [H.v2_item(60, D_VOX_ABC, available=[True, True, False]),
             H.v2_item(61, D_VOX_ABC, available=[True, False, False])]
    with applied_main_config(_cfg(tmp_path, H.CLASSES_ABC, "softmax")):
        model = _model(None)
        wants = [_expected(model, items, "all"), _expected(model, items, "available")]
        loss, grads, _ = H.first_step_loss_and_grads(model, items, tmp_path / "tb")
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert np.isfinite(loss)
    assert any(loss == pytest.approx(w, rel=1e-5) for w in wants), (loss, wants)
    missing = sorted(n for n in trainable if grads.get(n) is None)
    assert not missing, missing[:10]
    assert all(bool(torch.isfinite(grads[n]).all()) for n in trainable)


# --------------------------------------------------------------------------------------
# G3d-C6
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def softmax_ckpt(tmp_path_factory):
    base = H.v2_config

    def v2_config_softmax(*a, **kw):
        cfg = base(*a, **kw)
        cfg.train.particles.class_head = "softmax"
        return cfg

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H, "v2_config", v2_config_softmax)
        with applied_main_config(None):
            return H.build_v2_checkpoint(tmp_path_factory.mktemp("g3d_ckpt"), H.CLASSES_ABC, cond=False)


def _load(path):
    from tomocpt.networks.pickingModel import BasePickingModel

    return BasePickingModel.load_from_checkpoint(path, map_location="cpu").eval()


def _check_subset(out, probs_subset, classes, C):
    """``predict_step(..., classes=...)`` may return all C channels or only ``classes``; the
    channels of ``classes`` must equal ``class_probabilities(logits, "softmax", classes)``."""
    if out.shape[1] == C:
        got = out[:, classes]
    else:
        assert out.shape[1] == len(classes)
        got = out
    np.testing.assert_allclose(got.float().numpy(), probs_subset[:, classes].numpy(), rtol=0, atol=1e-6)


def test_G3d_C6_class_head_stored_in_checkpoint(softmax_ckpt):
    from tomocpt.configManager.configManager import safe_getattr

    ckpt = torch.load(softmax_ckpt["path"], map_location="cpu", weights_only=False)
    assert vars(ckpt["hyper_parameters"]["config"].train.particles)["class_head"] == "softmax"
    model = _load(softmax_ckpt["path"])
    assert safe_getattr(model.hparams.config.train.particles, "class_head", None) == "softmax"


def test_G3d_C6_default_is_sigmoid():
    from tomocpt.defaultConfigs.particles_config import ParticlesConfig
    from tomocpt.mainConfig import MainConfig

    assert ParticlesConfig().class_head == "sigmoid"
    assert MainConfig().train.particles.class_head == "sigmoid"


def test_G3d_C6_softmax_predict_step(softmax_ckpt):
    from tomocpt.training.multiclass_loss import class_probabilities

    model = _load(softmax_ckpt["path"])
    x = softmax_ckpt["x"]
    with torch.no_grad():
        lg = model(x)
        out = model.predict_step(x)
        np.testing.assert_allclose(out.float().numpy(), class_probabilities(lg, "softmax").numpy(),
                                   rtol=0, atol=1e-6)
        assert bool((out.double().sum(1) < 1.0 + 1e-6).all())
        for classes in ([0, 2], [1]):
            sub = model.predict_step(x, 0, 0, apix=None, classes=classes)
            _check_subset(sub, class_probabilities(lg, "softmax", classes), classes, 3)
        # the subset differs from the all-class normalisation (it is not just a channel slice)
        sub = model.predict_step(x, classes=[0, 2])
        got = sub[:, [0, 2]] if sub.shape[1] == 3 else sub
        assert float((got.float() - out[:, [0, 2]].float()).abs().max()) > 1e-4


def test_G3d_C6_sigmoid_predict_step_ignores_classes(tmp_path):
    with applied_main_config(_cfg(tmp_path, H.CLASSES_ABC, None)):
        model = _model(None).eval()
        x = H.fixed_input(1, seed=21)
        with torch.no_grad():
            lg = model(x)
            np.testing.assert_allclose(model.predict_step(x).float().numpy(), torch.sigmoid(lg).numpy(),
                                       rtol=0, atol=1e-6)
            sub = model.predict_step(x, classes=[0, 2])
            _check_subset(sub, torch.sigmoid(lg), [0, 2], 3)


@pytest.mark.parametrize("field", ["missing", "softmax"])
def test_G3d_C6_checkpoint_without_field_predicts_sigmoid(softmax_ckpt, tmp_path, field):
    """The softmax checkpoint with ``class_head`` removed from the pickled ``ParticlesConfig``
    (a checkpoint written before Gate 3d) predicts with sigmoid; control: field kept = softmax."""
    from tomocpt.training.multiclass_loss import class_probabilities

    ckpt = torch.load(softmax_ckpt["path"], map_location="cpu", weights_only=False)
    particles = ckpt["hyper_parameters"]["config"].train.particles
    if field == "missing":
        particles.__dict__.pop("class_head", None)
        assert "class_head" not in vars(particles)
    path = tmp_path / "edited.ckpt"
    torch.save(ckpt, path)
    model = _load(path)
    x = softmax_ckpt["x"]
    with torch.no_grad():
        lg = model(x)
        out = model.predict_step(x).float()
    want = torch.sigmoid(lg) if field == "missing" else class_probabilities(lg, "softmax")
    np.testing.assert_allclose(out.numpy(), want.numpy(), rtol=0, atol=1e-6)


def test_G3d_C6_invalid_class_head_raises(tmp_path):
    with applied_main_config(_cfg(tmp_path, H.CLASSES_ABC, "softmaxx")):
        with pytest.raises(ValueError):
            _model(None)
