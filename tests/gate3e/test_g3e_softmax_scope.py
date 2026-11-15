"""Gate 3e (spec ``03e_AGENT_SOFTMAX_SCOPE.md``): ``train.particles.softmax_scope``.

``all`` (default): the Gate 3d softmax over the background and every available channel.
``particles``: softmax over the background and the particle channels; filament channels are
independent sigmoids; targets are renormalised over the particle channels only.

Public names used: ``class_probabilities(logits, head, available=None, softmax_channels=None)``
and ``renormalise_targets(y, softmax_channels=None)`` (``softmax_channels``: ``[C]`` bool, the
channels inside the softmax; ``None`` = all), ``ParticlesConfig.softmax_scope``.
"""

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as G3
from gate6 import g6_helpers as G6
from utils.tiny import applied_main_config

PARTICLES = ((1, "A", 120.0), (2, "B", 200.0))
FILAMENTS = ((9, "actin", 70.0),)
D_VOX = [12.0, 20.0, 7.0]
IS_PARTICLE = torch.tensor([True, True, False])


def _items(n=2, available=None):
    """Two particles classes and an actin tube that runs through a class-0 particle, so the
    targets of the shared voxels sum above 1."""
    from tomocpt.labels.filaments import render_tubes

    items = []
    for b in range(n):
        it = G3.v2_item(300 + b, D_VOX, apix=10.0, available=available, n_per_class=0)
        poly = G6.line_xyz((4.0, 30.0 + 2 * b, 32.0), (60.0, 34.0, 32.0), n=15)
        tube = render_tubes(G6.segments_from_polyline(poly, 2, 7.0), (64, 64, 64), 3)
        pts = [[32.0, 32.0 + 2 * b, 30.0, 0, 12.0], [16.0, 48.0, 48.0, 1, 20.0], [48.0, 16.0, 14.0, 0, 12.0]]
        it["target_data"] = torch.from_numpy(np.maximum(G3.render(pts, 3), tube))
        items.append(it)
    return items


def _cfg(tmp_path, scope=None, filaments=FILAMENTS):
    cfg = G6.v2_config_with_filaments(tmp_path, PARTICLES, filaments, class_head="softmax")
    if scope is not None:
        cfg.train.particles.softmax_scope = scope
    return cfg


def _model(seed=0):
    from tomocpt.networks.pickingModel import BasePickingModel

    torch.manual_seed(seed)
    return BasePickingModel(lr=1e-3, train_continue=None)


def _ref_all(logits, available=None):
    """Gate 3d softmax, written out."""
    x = logits.float()
    B, C = x.shape[:2]
    m = torch.ones(B, C, dtype=torch.bool) if available is None else available
    m = m.view(B, C, 1, 1, 1)
    x = torch.where(m, x, torch.full_like(x, float("-inf")))
    p = torch.softmax(torch.cat([torch.zeros_like(x[:, :1]), x], 1), 1)[:, 1:]
    return torch.where(m, p, torch.zeros_like(p))


def _ref_particles(logits, is_particle, available=None):
    """Spec 3: softmax over the background and the (available) particle channels, sigmoid on the
    (available) filament channels, 0 on unavailable channels."""
    x = logits.float()
    B, C = x.shape[:2]
    m = torch.ones(B, C, dtype=torch.bool) if available is None else available.clone()
    out = torch.zeros_like(x)
    pm = m & is_particle.view(1, C)
    out += _ref_all(x, pm)
    fm = (m & ~is_particle.view(1, C)).view(B, C, 1, 1, 1)
    return torch.where(fm, torch.sigmoid(x), out)


def _expected_loss(model, items, scope):
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    b = G3.collate(items)
    av, y = b["class_available"], b["target_data"]
    with torch.no_grad():
        lg = model(b["input_data"])
    if scope == "all":
        p, yt = _ref_all(lg, av), y / y.sum(1, keepdim=True).clamp(min=1)
    else:
        p = _ref_particles(lg, IS_PARTICLE, av)
        mk = IS_PARTICLE.view(1, -1, 1, 1, 1)
        s = (y * mk).sum(1, keepdim=True).clamp(min=1)
        yt = torch.where(mk, y / s, y)
    return float(multiclass_picking_loss(p, yt, av, b["class_d_vox"], checkpoint=False))


# --------------------------------------------------------------------------------------
# C1 default `all` is today's behaviour
# --------------------------------------------------------------------------------------

def test_G3e_C1_default_field():
    from tomocpt.defaultConfigs.particles_config import ParticlesConfig
    from tomocpt.mainConfig import MainConfig

    assert ParticlesConfig().softmax_scope == "all"
    assert MainConfig().train.particles.softmax_scope == "all"


def test_G3e_C1_functions_default_unchanged():
    from tomocpt.training.multiclass_loss import class_probabilities, renormalise_targets

    torch.manual_seed(1)
    lg = torch.randn(2, 3, 6, 6, 6) * 3
    av = torch.tensor([[True, True, True], [True, False, True]])
    assert torch.equal(class_probabilities(lg, "softmax", av), class_probabilities(lg, "softmax", av, None))
    np.testing.assert_allclose(class_probabilities(lg, "softmax", av).numpy(), _ref_all(lg, av).numpy(),
                               rtol=0, atol=1e-6)
    y = torch.rand(2, 3, 6, 6, 6)
    assert torch.equal(renormalise_targets(y), renormalise_targets(y, None))
    assert torch.equal(renormalise_targets(y), y / y.sum(1, keepdim=True).clamp(min=1))


@pytest.mark.parametrize("scope", [None, "all"])
def test_G3e_C1_model_default_is_all(tmp_path, scope):
    items = _items()
    assert float(G3.collate(items)["target_data"].sum(1).max()) > 1.3, "particle and tube overlap"
    with applied_main_config(_cfg(tmp_path, scope)):
        model = _model()
        want = _expected_loss(model, items, "all")
        x = G3.collate(items)["input_data"][:1]
        model.eval()
        with torch.no_grad():
            lg = model(x)
            pred = model.predict_step(x)
        np.testing.assert_allclose(pred.float().numpy(), _ref_all(lg).numpy(), rtol=0, atol=1e-6)
        model.train()
        loss, _, _ = G3.first_step_loss_and_grads(model, items, tmp_path / "tb")
    assert loss == pytest.approx(want, rel=1e-5)


# --------------------------------------------------------------------------------------
# C2 probabilities with `particles`
# --------------------------------------------------------------------------------------

def test_G3e_C2_class_probabilities_particles_scope():
    from tomocpt.training.multiclass_loss import class_probabilities

    torch.manual_seed(2)
    lg = torch.randn(2, 3, 6, 6, 6) * 3
    for av in (None, torch.tensor([[True, True, True], [False, True, True]])):
        p = class_probabilities(lg, "softmax", av, IS_PARTICLE)
        np.testing.assert_allclose(p.numpy(), _ref_particles(lg, IS_PARTICLE, av).numpy(), rtol=0, atol=1e-6)
    p = class_probabilities(lg, "softmax", None, IS_PARTICLE)
    np.testing.assert_allclose(p[:, 2].numpy(), torch.sigmoid(lg[:, 2]).numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(p[:, :2].numpy(), class_probabilities(lg[:, :2], "softmax").numpy(),
                               rtol=0, atol=1e-6)
    lg2 = lg.clone()
    lg2[:, 2] += 5.0
    p2 = class_probabilities(lg2, "softmax", None, IS_PARTICLE)
    assert torch.equal(p2[:, :2], p[:, :2]), "filament logits must not enter the particle softmax"
    assert not torch.equal(class_probabilities(lg2, "softmax")[:, :2], class_probabilities(lg, "softmax")[:, :2])
    # a list of bools works too
    assert torch.equal(class_probabilities(lg, "softmax", None, [True, True, False]), p)


# --------------------------------------------------------------------------------------
# C3 targets
# --------------------------------------------------------------------------------------

def test_G3e_C3_renormalise_particles_only():
    from tomocpt.training.multiclass_loss import renormalise_targets

    y = torch.zeros(1, 3, 2, 2, 2)
    y[0, 0] = 0.9
    y[0, 1] = 0.6
    y[0, 2] = 1.0
    y[0, :, 1, 1, 1] = torch.tensor([0.3, 0.2, 1.0])
    out = renormalise_targets(y, IS_PARTICLE)
    assert torch.equal(out[:, 2], y[:, 2]), "filament targets unchanged"
    s = y[:, :2].sum(1, keepdim=True).clamp(min=1)
    assert torch.equal(out[:, :2], y[:, :2] / s)
    assert float(out[0, 0, 1, 1, 1]) == pytest.approx(0.3), "particle sum <= 1 is kept although the filament adds 1"


# --------------------------------------------------------------------------------------
# C4 no filament channels: `particles` == `all`
# --------------------------------------------------------------------------------------

def test_G3e_C4_no_filaments_scope_irrelevant(tmp_path):
    items = [G3.v2_item(310 + i, D_VOX[:2]) for i in range(2)]
    out = {}
    for scope in ("all", "particles"):
        with applied_main_config(_cfg(tmp_path / scope, scope, filaments=())):
            model = _model()
            x = items[0]["input_data"][None]
            model.eval()
            with torch.no_grad():
                pred = model.predict_step(x).clone()
            model.train()
            loss, _, _ = G3.first_step_loss_and_grads(model, items, tmp_path / f"tb_{scope}")
            out[scope] = (loss, pred)
    assert out["all"][0] == out["particles"][0]
    assert torch.equal(out["all"][1], out["particles"][1])


# --------------------------------------------------------------------------------------
# C5 training with `particles`, old checkpoints, no new parameters
# --------------------------------------------------------------------------------------

def test_G3e_C5_particles_training_step(tmp_path):
    items = _items(available=[True, False, True])
    with applied_main_config(_cfg(tmp_path, "particles")):
        model = _model()
        want = _expected_loss(model, items, "particles")
        want_all = _expected_loss(model, items, "all")
        loss, grads, _ = G3.first_step_loss_and_grads(model, items, tmp_path / "tb")
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert np.isfinite(loss)
    assert loss == pytest.approx(want, rel=1e-5)
    assert abs(want - want_all) > 1e-4 * abs(want)
    missing = sorted(n for n in trainable if grads.get(n) is None)
    assert not missing, missing[:10]


def test_G3e_C5_no_new_parameters(tmp_path):
    keys = {}
    for scope in ("all", "particles"):
        with applied_main_config(_cfg(tmp_path / scope, scope)):
            keys[scope] = {k: tuple(v.shape) for k, v in _model().state_dict().items()}
    assert keys["all"] == keys["particles"]


@pytest.fixture(scope="module")
def particles_ckpt(tmp_path_factory):
    from tomocpt.networks.pickingModel import BasePickingModel

    root = tmp_path_factory.mktemp("g3e_ckpt")
    with applied_main_config(None):
        cfg = _cfg(root, "particles")
        with applied_main_config(cfg):
            model = _model()
            trainer = G3.cpu_trainer(root / "tb")
            trainer.fit(model, datamodule=G3.datamodule(_items()))
            path = root / "particles.ckpt"
            trainer.save_checkpoint(path)
    return dict(path=path, x=G3.fixed_input(1, seed=21))


def _load(path):
    from tomocpt.networks.pickingModel import BasePickingModel

    return BasePickingModel.load_from_checkpoint(path, map_location="cpu").eval()


@pytest.mark.parametrize("field", ["missing", "kept"])
def test_G3e_C5_checkpoint_without_field_predicts_all(particles_ckpt, tmp_path, field):
    ckpt = torch.load(particles_ckpt["path"], map_location="cpu", weights_only=False)
    particles = ckpt["hyper_parameters"]["config"].train.particles
    assert vars(particles)["softmax_scope"] == "particles"
    if field == "missing":
        particles.__dict__.pop("softmax_scope")
    path = tmp_path / "edited.ckpt"
    torch.save(ckpt, path)
    model = _load(path)
    x = particles_ckpt["x"]
    with torch.no_grad():
        lg = model(x)
        out = model.predict_step(x).float()
    want = _ref_all(lg) if field == "missing" else _ref_particles(lg, IS_PARTICLE)
    np.testing.assert_allclose(out.numpy(), want.numpy(), rtol=0, atol=1e-6)


# --------------------------------------------------------------------------------------
# C6 predict with a level subset
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("classes", [[0, 2], [1, 2], [0, 1], [2]])
def test_G3e_C6_level_subset(particles_ckpt, classes):
    model = _load(particles_ckpt["path"])
    x = particles_ckpt["x"]
    av = torch.zeros(1, 3, dtype=torch.bool)
    av[0, classes] = True
    with torch.no_grad():
        lg = model(x)
        out = model.predict_step(x, 0, 0, apix=None, classes=classes).float()
    want = _ref_particles(lg, IS_PARTICLE, av)
    got = out[:, classes] if out.shape[1] == 3 else out
    np.testing.assert_allclose(got.numpy(), want[:, classes].numpy(), rtol=0, atol=1e-6)
