"""G3d-C2, G3d-C3: ``tomocpt.training.multiclass_loss.class_probabilities``."""

import numpy as np
import pytest
import torch


def _cp():
    from tomocpt.training.multiclass_loss import class_probabilities

    return class_probabilities


def _logits(B=2, C=4, S=8, scale=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((B, C, S, S, S), generator=g) * scale


def _reference(logits, avail):
    """``p_c = exp(l_c) / (1 + sum_{d available} exp(l_d))`` in float64; 0 for unavailable c."""
    l = logits.double()
    m = avail.view(*avail.shape, 1, 1, 1).to(torch.bool)
    e = torch.where(m, torch.exp(l), torch.zeros_like(l))
    return e / (1.0 + e.sum(1, keepdim=True))


# --------------------------------------------------------------------------------------
# G3d-C2
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("how", ["none", "mask", "indices"])
def test_G3d_C2_one_class_equals_sigmoid(how):
    cp = _cp()
    lg = _logits(C=1, scale=5.0)
    avail = {"none": None, "mask": torch.ones(2, 1, dtype=torch.bool), "indices": [0]}[how]
    p = cp(lg, "softmax", avail)
    assert p.dtype == torch.float32 and p.shape == lg.shape
    np.testing.assert_allclose(p.numpy(), torch.sigmoid(lg).numpy(), rtol=0, atol=1e-6)


def test_G3d_C2_sigmoid_head_is_sigmoid():
    """``class_head="sigmoid"`` with all channels: plain sigmoid (unchanged behaviour)."""
    lg = _logits()
    np.testing.assert_allclose(_cp()(lg, "sigmoid").numpy(), torch.sigmoid(lg).numpy(), rtol=0, atol=1e-6)


def test_G3d_C2_multi_class_ranges_and_formula():
    cp = _cp()
    lg = _logits(C=4, scale=3.0)
    avail = torch.tensor([[True, False, True, True], [True, True, True, False]])
    p = cp(lg, "softmax", avail)
    assert p.dtype == torch.float32 and p.shape == lg.shape
    s = p.double().sum(1)
    assert bool((s < 1.0).all())
    m = avail.view(2, 4, 1, 1, 1).expand_as(p)
    assert bool((p[m] > 0).all()) and bool((p[m] < 1).all())
    assert torch.count_nonzero(p[~m]) == 0
    np.testing.assert_allclose(p.double().numpy(), _reference(lg, avail).numpy(), rtol=1e-5, atol=1e-7)
    # all available (None) = the formula with every channel
    p_all = cp(lg, "softmax", None)
    np.testing.assert_allclose(p_all.double().numpy(), _reference(lg, torch.ones(2, 4, dtype=torch.bool)).numpy(),
                               rtol=1e-5, atol=1e-7)
    assert bool((p_all.double().sum(1) < 1.0).all())


def test_G3d_C2_index_sequence_equals_mask():
    cp = _cp()
    lg = _logits(C=4)
    mask = torch.tensor([[True, False, True, False]] * 2)
    np.testing.assert_allclose(cp(lg, "softmax", [0, 2]).numpy(), cp(lg, "softmax", mask).numpy(),
                               rtol=0, atol=1e-7)


def test_G3d_C2_finite_for_extreme_logits():
    """Logits of +-80: finite, in [0, 1], sum <= 1. (Strict (0, 1) cannot hold in float32 at
    +80, where exp(80) / (1 + exp(80)) rounds to 1; see README.)"""
    cp = _cp()
    lg = _logits(C=3, scale=1.0)
    lg[:, 0] = 80.0
    lg[0, 1] = -80.0
    lg[1, 1] = 80.0
    lg[:, 2, :4] = -80.0
    for avail in (None, torch.tensor([[True, True, False], [True, True, True]])):
        p = cp(lg, "softmax", avail)
        assert bool(torch.isfinite(p).all())
        assert bool((p >= 0).all()) and bool((p <= 1).all())
        assert bool((p.double().sum(1) <= 1.0 + 1e-6).all())
        ref = _reference(lg, torch.ones(2, 3, dtype=torch.bool) if avail is None else avail)
        np.testing.assert_allclose(p.double().numpy(), ref.numpy(), rtol=1e-5, atol=1e-6)
    # one class at +-80 is still the sigmoid
    one = torch.tensor([80.0, -80.0, 0.0]).view(1, 1, 3, 1, 1)
    np.testing.assert_allclose(cp(one, "softmax").numpy(), torch.sigmoid(one).numpy(), rtol=0, atol=1e-6)


def test_G3d_C2_bf16_input():
    """bf16 logits: float32 output, finite, equal to the float32 computation on the same values."""
    cp = _cp()
    lg = _logits(C=4, scale=3.0)
    lg[0, 0, 0] = 80.0
    lg[1, 2, 1] = -80.0
    lb = lg.to(torch.bfloat16)
    avail = torch.tensor([[True, True, False, True], [True, True, True, True]])
    p = cp(lb, "softmax", avail)
    assert p.dtype == torch.float32
    assert bool(torch.isfinite(p).all())
    np.testing.assert_allclose(p.numpy(), cp(lb.float(), "softmax", avail).numpy(), rtol=0, atol=1e-6)
    assert bool((p.double().sum(1) <= 1.0 + 1e-6).all())


# --------------------------------------------------------------------------------------
# G3d-C3
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("new_value", [80.0, -80.0, "noise"])
def test_G3d_C3_unavailable_logit_has_no_effect(new_value):
    cp = _cp()
    lg = _logits(C=4)
    avail = torch.tensor([[True, False, True, True], [False, True, True, False]])
    p = cp(lg, "softmax", avail)
    other = lg.clone()
    g = torch.Generator().manual_seed(5)
    for b, c in zip(*np.nonzero(~avail.numpy())):
        if new_value == "noise":
            other[b, c] = torch.randn(other[b, c].shape, generator=g) * 20
        else:
            other[b, c] = new_value
    q = cp(other, "softmax", avail)
    m = avail.view(2, 4, 1, 1, 1).expand_as(p)
    np.testing.assert_allclose(q[m].numpy(), p[m].numpy(), rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(q[~m]) == 0
