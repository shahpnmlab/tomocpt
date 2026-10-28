"""G5-C4 (masked L1), G5-C7 (dense InfoNCE), G5-C8 (antisymmetric relative-scale head).

Contracts assumed here (README.md):
* ``masked_l1(pred, target, mask, visible_weight)``: tensors of one shape; ``mask`` is bool
  (True = masked, i.e. hidden from the network); returns a scalar.
* ``dense_info_nce(f_a, f_b, pts_a, pts_b, temperature)``: ``f`` is ``[B, D, d, h, w]``; ``pts``
  is ``[B, n, 3]`` float, ``(z, y, x)`` in view voxel coordinates (the 64^3 view and the feature
  map cover the same field of view). The test keeps points in [8, 28] so that they are inside a
  32^3 map under any reasonable mapping.
* ``RelativeScaleHead(in_dim)``; ``head(p_a, p_b)`` with ``p`` of shape ``[B, in_dim]`` returns
  ``[B]`` or ``[B, 1]``.
"""

import numpy as np
import pytest
import torch

#: tiny model, feature_size 12: hidden[3] has 8 fs = 96 channels, hidden[4] 16 fs = 192
P_DIM = 96 + 192


# --------------------------------------------------------------------------------------
# G5-C4
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("visible_weight", [0.1, 0.37, 1.0])
def test_G5_C4_masked_l1_gradient_ratio(visible_weight):
    from tomocpt.networks.ssl.losses import masked_l1

    shape = (2, 1, 16, 16, 16)
    base = torch.randn(shape)
    pred = base.clone().requires_grad_(True)
    target = base - 0.5  # equal residuals everywhere
    idx = torch.arange(int(np.prod(shape))).reshape(shape)
    mask = (idx % 2) == 0  # exactly half of the voxels masked
    loss = masked_l1(pred, target, mask, visible_weight)
    assert loss.ndim == 0
    loss.backward()
    g_m, g_v = pred.grad[mask], pred.grad[~mask]
    assert torch.allclose(g_m, g_m[0]) and torch.allclose(g_v, g_v[0])
    assert float(g_m[0]) > 0
    np.testing.assert_allclose(float(g_v[0]) / float(g_m[0]), visible_weight, rtol=1e-5)


# --------------------------------------------------------------------------------------
# G5-C7
# --------------------------------------------------------------------------------------

def _dense_inputs(B=2, D=16, n=64, size=32):
    g = torch.Generator().manual_seed(0)
    f = torch.randn(B, D, size, size, size, generator=g)
    pts = 8.0 + 20.0 * torch.rand(B, n, 3, generator=g)
    perm = torch.stack([torch.randperm(n, generator=g) for _ in range(B)])
    shuffled = torch.stack([pts[b, perm[b]] for b in range(B)])
    return f, pts, shuffled


def test_G5_C7_dense_info_nce_prefers_true_correspondences():
    from tomocpt.networks.ssl.losses import dense_info_nce

    f, pts, shuffled = _dense_inputs()
    good = dense_info_nce(f, f.clone(), pts, pts.clone(), 0.1)
    bad = dense_info_nce(f, f.clone(), pts, shuffled, 0.1)
    assert good.ndim == 0 and torch.isfinite(good) and torch.isfinite(bad)
    assert float(good) < float(bad)


def test_G5_C7_dense_info_nce_finite_under_bf16_autocast():
    from tomocpt.networks.ssl.losses import dense_info_nce

    f, pts, shuffled = _dense_inputs()
    fa = f.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        good = dense_info_nce(fa, f.clone(), pts, pts.clone(), 0.1)
        bad = dense_info_nce(f, f.clone(), pts, shuffled, 0.1)
    assert torch.isfinite(good) and torch.isfinite(bad)
    good.float().backward()
    assert torch.isfinite(fa.grad).all()
    if torch.cuda.is_available():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = dense_info_nce(f.cuda(), f.cuda(), pts.cuda(), shuffled.cuda(), 0.1)
        assert torch.isfinite(out)


# --------------------------------------------------------------------------------------
# G5-C8
# --------------------------------------------------------------------------------------

def test_G5_C8_relative_scale_head_is_antisymmetric():
    from tomocpt.networks.ssl.heads import RelativeScaleHead

    torch.manual_seed(0)
    head = RelativeScaleHead(P_DIM)
    with torch.no_grad():  # also with non-trivial weights in every layer
        for p in head.parameters():
            p.add_(0.1 * torch.randn_like(p))
    a, b = torch.randn(4, P_DIM), torch.randn(4, P_DIM)
    head.eval()
    with torch.no_grad():
        r_ab, r_ba = head(a, b), head(b, a)
        assert r_ab.shape[0] == 4 and r_ab.numel() == 4
        torch.testing.assert_close(r_ab, -r_ba, atol=1e-6, rtol=0)
        assert float(r_ab.abs().max()) > 0
        torch.testing.assert_close(head(a, a), torch.zeros_like(r_ab), atol=1e-6, rtol=0)
