"""G3b-C1, G3b-C2, G3b-C3: ``multiclass_picking_loss(..., cross_class_neg_weight=k)``.

The new keyword is passed inside the tests, so the module imports on code without it.
"""

import importlib.util

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as H
from gate3.test_g3_loss import _c4_inputs
from gate3b.g3b_reference import reference_loss

GOLDEN = H.GOLDEN_DIR / "gate3b_c1_loss.npz"
SIZE = 32
N = SIZE ** 3
DELTA = 0.01


def _common():
    spec = importlib.util.spec_from_file_location("_gate3b_common", H.GOLDEN_DIR / "gate3b_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _loss_fn():
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    return multiclass_picking_loss


# --------------------------------------------------------------------------------------
# G3b-C1: k = 0 and the default equal the Gate 3 loss
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("case", ["c1", "c4", "c4w"])
@pytest.mark.parametrize("how", ["default", "k0"])
def test_G3b_C1_k0_equals_gate3_loss_golden(case, checkpoint, how):
    """Loss and gradient w.r.t. ``y_pred`` equal the golden written by the Gate 3 loss
    (``make_goldens_gate3b.py`` at 2b62832 = ``v2-gate-4-passed`` code), rtol 1e-6.
    Cases: C = 1 (B = 3, one sample unavailable), C = 4 with partial availability, and C = 4
    with class weights. ``how="default"`` omits the keyword (only meaningful with Gate 3b,
    so it also requires the keyword to exist)."""
    import inspect

    g = _common()
    fn = _loss_fn()
    assert "cross_class_neg_weight" in inspect.signature(fn).parameters
    assert inspect.signature(fn).parameters["cross_class_neg_weight"].default == 0.0
    z = np.load(GOLDEN)
    y_pred, y_true, *_ = g.case_inputs(case)
    assert float(y_true.double().sum()) == pytest.approx(float(z[f"{case}_ytrue_sum"]), rel=1e-9)
    assert float(y_pred.double().sum()) == pytest.approx(float(z[f"{case}_ypred_sum"]), rel=1e-9)
    kw = {} if how == "default" else {"cross_class_neg_weight": 0.0}
    loss, grad = g.loss_and_grad(fn, case, checkpoint, **kw)
    assert loss == pytest.approx(float(z[f"{case}_ck{int(checkpoint)}_loss"]), rel=1e-6, abs=0)
    want = z[f"{case}_grad"]
    np.testing.assert_allclose(grad.numpy(), want, rtol=1e-6, atol=1e-6 * float(np.abs(want).max()))
    # unavailable pairs: still exactly zero
    _, _, avail, _, _ = g.case_inputs(case)
    for b, c in zip(*np.nonzero(~avail.numpy())):
        assert torch.count_nonzero(grad[b, c]) == 0


# --------------------------------------------------------------------------------------
# G3b-C2: weights
# --------------------------------------------------------------------------------------

def _scene():
    """C = 3 on a 32^3 cube, both samples with the same targets.

    class 0: centre (10, 16, 16), d = 10;  class 1: centre (17, 16, 16), d = 10 (overlaps class 0);
    class 2: centre (17, 16, 25), d = 10 (overlaps class 1, not class 0).
    Sample 0: class 2 unavailable. Sample 1: all available.
    """
    pts = [[10, 16, 16, 0, 10.0], [17, 16, 16, 1, 10.0], [17, 16, 25, 2, 10.0]]
    from tomocpt.labels.helpers import render_targets

    y = render_targets(np.asarray(pts, dtype=np.float64), (SIZE,) * 3, 3, "clip")
    y_true = torch.from_numpy(np.stack([y, y])).double()
    avail = torch.tensor([[True, True, False], [True, True, True]])
    d_vox = torch.full((2, 3), 10.0, dtype=torch.float64)
    return y_true, avail, d_vox


def _voxels(y):
    """Named voxels (z, y, x) of the scene, chosen from the targets ``y`` ``[3, D, H, W]``."""
    y = y.numpy()

    def first(mask, score=None):
        idx = np.argwhere(mask)
        assert len(idx), "scene voxel not found"
        if score is not None:
            idx = idx[np.argsort(score[tuple(idx.T)], kind="stable")]
        return tuple(int(v) for v in idx[0])

    v = {
        "pos_only": (10, 16, 16),                      # y0 = 1, y1 = y2 = 0
        "other_centre": (17, 16, 16),                  # y1 = 1, y0 = y2 = 0
        "unavail_centre": (17, 16, 25),                # y2 = 1, y0 = y1 = 0
        "background": (2, 2, 2),
        # positive of class 0 inside class 1's sphere
        "overlap": first((y[0] > 0) & (y[1] > 0.1)),
        # outside class 0, soft value of class 1 only
        "soft": first((y[0] == 0) & (y[2] == 0) & (y[1] > 0.2) & (y[1] < 0.8), score=-np.abs(y[1] - 0.5)),
        # outside class 0, inside classes 1 and 2 with clearly different values
        "both": first((y[0] == 0) & (y[1] > 0) & (y[2] > 0) & (np.abs(y[1] - y[2]) > 0.02),
                      score=-np.abs(y[1] - y[2])),
    }
    assert y[0][v["pos_only"]] == 1.0 and y[1][v["pos_only"]] == 0 and y[2][v["pos_only"]] == 0
    assert y[1][v["other_centre"]] == 1.0 and y[0][v["other_centre"]] == 0 and y[2][v["other_centre"]] == 0
    assert y[2][v["unavail_centre"]] == 1.0 and y[0][v["unavail_centre"]] == 0 and y[1][v["unavail_centre"]] == 0
    assert y[:, 2, 2, 2].sum() == 0
    return v


def _grad(y_true, avail, d_vox, k, checkpoint, y_pred=None):
    fn = _loss_fn()
    yp = (y_true + DELTA if y_pred is None else y_pred).clone().requires_grad_(True)
    loss = fn(yp, y_true, avail, d_vox, checkpoint=checkpoint, cross_class_neg_weight=k)
    loss.backward()
    return loss.detach(), yp.grad


@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("k", [0.0, 0.5, 2.0])
def test_G3b_C2_voxel_weights_follow_the_formula(k, checkpoint):
    """``y_pred = y_true + delta`` (float64): the Hessian term has no gradient (a constant shift)
    and the Huber term is quadratic, so the gradient at a voxel of an available pair is exactly
    ``w * delta / (N * n_pairs)``. Each named voxel is compared with the spec formula
    ``w = where(y_c > 0, w_pos, 2 w_neg + k w_pos o)``, ``o`` = max over available d != c, rel 1e-6."""
    y_true, avail, d_vox = _scene()
    _, g = _grad(y_true, avail, d_vox, k, checkpoint)
    v = _voxels(y_true[0])
    n_pairs = int(avail.sum())
    S = y_true.flatten(2).sum(2)
    w_pos, w_neg = N / (S + 1), N / (N - S + 1)

    def expect(b, c, vox):
        yc = float(y_true[(b, c) + vox])
        if yc > 0:
            w = float(w_pos[b, c])
        else:
            o = max([float(y_true[(b, d) + vox]) for d in range(3) if d != c and bool(avail[b, d])] + [0.0])
            w = float(2 * w_neg[b, c] + k * w_pos[b, c] * o)
        return w * DELTA / (N * n_pairs)

    checks = [
        (0, 0, "pos_only"), (0, 0, "overlap"), (0, 0, "other_centre"), (0, 0, "soft"),
        (0, 0, "unavail_centre"), (0, 0, "background"),
        (0, 1, "pos_only"), (0, 1, "other_centre"), (0, 1, "unavail_centre"),
        (1, 0, "unavail_centre"), (1, 0, "both"), (1, 0, "other_centre"), (1, 0, "background"),
        (1, 2, "other_centre"), (1, 2, "pos_only"), (1, 2, "unavail_centre"),
    ]
    for b, c, name in checks:
        got = float(g[(b, c) + v[name]])
        assert got == pytest.approx(expect(b, c, v[name]), rel=1e-6, abs=0), (b, c, name, k)
    # unavailable pair: exactly zero gradient
    assert torch.count_nonzero(g[0, 2]) == 0


def test_G3b_C2_extra_weight_only_on_negatives_inside_other_available_classes():
    """With k > 0: voxels with ``y_true[c] > 0`` and voxels outside every other available class
    have the same gradient as with k = 0; voxels inside another available class get a larger one;
    an unavailable class d (sample 0, class 2) does not raise channel c's weight."""
    y_true, avail, d_vox = _scene()
    _, g0 = _grad(y_true, avail, d_vox, 0.0, False)
    _, gk = _grad(y_true, avail, d_vox, 1.0, False)
    for b in range(2):
        for c in range(3):
            if not bool(avail[b, c]):
                assert torch.count_nonzero(gk[b, c]) == 0
                continue
            o = torch.zeros_like(y_true[b, c])
            for d in range(3):
                if d != c and bool(avail[b, d]):
                    o = torch.maximum(o, y_true[b, d])
            same = (y_true[b, c] > 0) | (o == 0)
            raised = (y_true[b, c] == 0) & (o > 0)
            assert int(raised.sum()) > 100 and int((same & (y_true[b, c] > 0)).sum()) > 100
            np.testing.assert_allclose(gk[b, c][same].numpy(), g0[b, c][same].numpy(), rtol=1e-9, atol=0)
            assert bool((gk[b, c][raised] > g0[b, c][raised] * (1 + 1e-6)).all())
    # the unavailable class 2 of sample 0 leaves channels 0 and 1 of sample 0 at the k = 0 gradient
    in2 = y_true[0, 2] > 0
    outside01 = (y_true[0, 0] == 0) & (y_true[0, 1] == 0)
    for c in (0, 1):
        np.testing.assert_allclose(gk[0, c][in2 & outside01].numpy(), g0[0, c][in2 & outside01].numpy(),
                                   rtol=1e-9, atol=0)


@pytest.mark.parametrize("checkpoint", [True, False])
def test_G3b_C2_unavailable_class_target_is_ignored(checkpoint):
    """Replacing the target of an unavailable pair by noise changes neither the loss nor any
    gradient (it is excluded from ``o`` of the other classes and masked itself)."""
    y_true, avail, d_vox = _scene()
    g = torch.Generator().manual_seed(3)
    y_pred = torch.rand(y_true.shape, generator=g, dtype=torch.float64)
    loss, grad = _grad(y_true, avail, d_vox, 2.0, checkpoint, y_pred=y_pred)
    other = y_true.clone()
    other[0, 2] = torch.rand(other[0, 2].shape, generator=g, dtype=torch.float64)
    loss2, grad2 = _grad(other, avail, d_vox, 2.0, checkpoint, y_pred=y_pred)
    assert torch.equal(loss, loss2)
    assert torch.equal(grad, grad2)
    assert torch.count_nonzero(grad[0, 2]) == 0


@pytest.mark.parametrize("k", [0.5, 2.0])
def test_G3b_C2_loss_and_gradient_equal_reference(k):
    """Random C = 4 batch (64^3, mixed availability and diameters, from Gate 3 C4) and the C = 4
    weighted golden case: loss and full gradient equal the independent reference, rtol 1e-5."""
    fn = _loss_fn()
    y_pred, b = _c4_inputs(seed=2)
    cases = [(y_pred, b["target_data"], b["class_available"], b["class_d_vox"], None)]
    cases.append(_common().case_inputs("c4w"))
    for yp0, yt, av, dv, cw in cases:
        yp = yp0.clone().requires_grad_(True)
        loss = fn(yp, yt, av, dv, class_weights=cw, checkpoint=False, cross_class_neg_weight=k)
        loss.backward()
        yr = yp0.clone().requires_grad_(True)
        ref = reference_loss(yr, yt, av, dv, k=k, class_weights=cw)
        ref.backward()
        assert float(loss) == pytest.approx(float(ref), rel=1e-5)
        np.testing.assert_allclose(yp.grad.numpy(), yr.grad.numpy(), rtol=1e-5,
                                   atol=1e-6 * float(yr.grad.abs().max()))
        with torch.no_grad():
            l0 = float(fn(yp0, yt, av, dv, class_weights=cw, checkpoint=False))
        assert float(loss) > l0 * 1.01  # the extra term matters on these batches


# --------------------------------------------------------------------------------------
# G3b-C3: checkpointed per-class form equals the vectorised form
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0.25, 3.0])
def test_G3b_C3_checkpointed_equals_vectorised(k):
    fn = _loss_fn()
    y_pred, b = _c4_inputs(seed=3)
    args = (b["target_data"], b["class_available"], b["class_d_vox"])
    out = {}
    for ck in (True, False):
        yp = y_pred.clone().requires_grad_(True)
        loss = fn(yp, *args, checkpoint=ck, cross_class_neg_weight=k)
        loss.backward()
        out[ck] = (float(loss), yp.grad.clone())
    assert out[True][0] == pytest.approx(out[False][0], rel=1e-5)
    np.testing.assert_allclose(out[True][1].numpy(), out[False][1].numpy(), rtol=1e-5,
                               atol=1e-6 * float(out[False][1].abs().max()))
    with torch.no_grad():
        ref = float(reference_loss(y_pred, *args, k=k))
    assert out[True][0] == pytest.approx(ref, rel=1e-5)
