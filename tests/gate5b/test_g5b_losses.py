"""G5b-C3 (``vicreg_terms``) and G5b-C4 (``nt_xent_with_queue``, function level).

Contracts assumed here (README.md):
* ``vicreg_terms(z) -> (variance_term, covariance_term)``: ``z`` is ``[N, D]``; two scalar tensors,
  **unweighted** (the model multiplies them by ``variance_weight`` / ``covariance_weight``).
  Variance term: hinge on the per-dimension std, ``mean_d relu(1 - sqrt(var_d + eps))`` (VICReg;
  a sum over dimensions instead of the mean also passes). Covariance term: squared off-diagonal
  entries of the covariance matrix (``sum / D`` in VICReg; any positive normalisation passes).
  A dict with one key containing ``var`` and one containing ``cov`` is also accepted.
* ``nt_xent_with_queue(z_a, z_b, queue, temperature, l1_weight)``: the Gate 5
  ``nt_xent(z_a, z_b, temperature, l1_weight)`` with the queue rows (``[Q, D]``, ``Q`` may be 0) as
  extra negatives for every one of the ``2B`` anchors (cosine similarity, same temperature); the
  ``l1_weight`` term stays on ``z`` only. Arguments are passed positionally.
"""

import math

import pytest
import torch
import torch.nn.functional as F


def _terms(out):
    if isinstance(out, dict):
        var = [v for k, v in out.items() if "var" in k.lower()]
        cov = [v for k, v in out.items() if "cov" in k.lower()]
        assert len(var) == 1 and len(cov) == 1, sorted(out)
        out = (var[0], cov[0])
    assert len(out) == 2
    for t in out:
        assert isinstance(t, torch.Tensor) and t.numel() == 1
    return out[0].reshape(()), out[1].reshape(())


def _vicreg(z):
    from tomocpt.networks.ssl.losses import vicreg_terms

    return _terms(vicreg_terms(z))


def _decorrelated(n=256, d=16, seed=0):
    """``[n, d]`` with centred, exactly orthogonal columns of std 1 ... 4 (diagonal covariance)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    x = x - x.mean(0)
    q, _ = torch.linalg.qr(x)                       # orthonormal, centred columns
    scale = torch.linspace(1.0, 4.0, d, dtype=torch.float64)
    return (q * math.sqrt(n - 1) * scale).float()   # column std = scale (unbiased)


# --------------------------------------------------------------------------------------
# G5b-C3
# --------------------------------------------------------------------------------------

def test_G5b_C3_variance_term_zero_when_every_std_at_least_one():
    z = _decorrelated()
    var, _ = _vicreg(z)
    assert float(var) == pytest.approx(0.0, abs=1e-6)
    var2, _ = _vicreg(3.0 * torch.randn(512, 32))  # std ~3 in every dimension
    assert float(var2) == pytest.approx(0.0, abs=1e-6)


def test_G5b_C3_variance_term_positive_when_collapsed():
    z = torch.full((64, 32), 0.3) + torch.linspace(-1, 1, 32)  # every row equal
    var, _ = _vicreg(z)
    assert float(var) > 0.5  # mean form ~1 (1 - sqrt(eps)); sum form ~32


def test_G5b_C3_covariance_term_zero_for_decorrelated_dims():
    _, cov = _vicreg(_decorrelated())
    assert float(cov) == pytest.approx(0.0, abs=1e-6)
    z = _decorrelated()
    z[:, 1] = z[:, 0]  # two identical dimensions
    _, cov_corr = _vicreg(z)
    assert float(cov_corr) > 1e-2


def test_G5b_C3_gradients_finite_under_bf16():
    """CPU bf16 autocast, normal and fully collapsed embeddings from a linear projection."""
    lin = torch.nn.Linear(24, 32)
    for x in (torch.randn(16, 24), torch.ones(16, 24)):
        lin.zero_grad(set_to_none=True)
        xin = x.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            var, cov = _vicreg(lin(xin))
            loss = 25.0 * var + 1.0 * cov
        assert torch.isfinite(loss)
        loss.backward()
        for g in (lin.weight.grad, lin.bias.grad, xin.grad):
            assert g is not None and torch.isfinite(g).all()


def test_G5b_C3_projection_head_uses_batchnorm():
    """Spec: ``ProjectionHead`` gets BatchNorm between its layers (SyncBatchNorm under DDP)."""
    from torch.nn.modules.batchnorm import _BatchNorm

    from tomocpt.networks.ssl.heads import ProjectionHead

    head = ProjectionHead(192, 256)
    assert any(isinstance(m, _BatchNorm) for m in head.modules())
    head.train()
    z = head(torch.randn(8, 192))
    assert z.shape == (8, 256)


# --------------------------------------------------------------------------------------
# G5b-C4 (function)
# --------------------------------------------------------------------------------------

def _pairs(b=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, d, generator=g), torch.randn(b, d, generator=g)


def _reference(z_a, z_b, queue, t, l1):
    """NT-Xent with the queue rows as extra negatives of every anchor (float64)."""
    z = torch.cat([z_a, z_b]).double()
    zn = F.normalize(z, dim=1)
    qn = F.normalize(queue.double(), dim=1)
    n = z_a.shape[0]
    sim = zn @ zn.t() / t
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool), float("-inf"))
    logits = torch.cat([sim, zn @ qn.t() / t], dim=1)
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(logits, labels) + l1 * z.abs().mean()


@pytest.mark.parametrize("l1", [0.0, 1e-4])
def test_G5b_C4_empty_queue_equals_nt_xent(l1):
    from tomocpt.networks.ssl.losses import nt_xent, nt_xent_with_queue

    z_a, z_b = _pairs()
    empty = torch.empty(0, z_a.shape[1])
    got = nt_xent_with_queue(z_a, z_b, empty, 0.1, l1)
    want = nt_xent(z_a, z_b, 0.1, l1)
    assert got.ndim == 0
    torch.testing.assert_close(got.float(), want.float(), rtol=1e-6, atol=1e-6)


def test_G5b_C4_queue_rows_are_extra_negatives():
    from tomocpt.networks.ssl.losses import nt_xent_with_queue

    z_a, z_b = _pairs()
    g = torch.Generator().manual_seed(1)
    queue = torch.randn(32, z_a.shape[1], generator=g)
    got = nt_xent_with_queue(z_a, z_b, queue, 0.1, 1e-4)
    torch.testing.assert_close(got.double(), _reference(z_a, z_b, queue, 0.1, 1e-4), rtol=1e-5, atol=1e-5)
    # Hard negatives (copies of the positives) raise the loss.
    empty = nt_xent_with_queue(z_a, z_b, torch.empty(0, z_a.shape[1]), 0.1, 0.0)
    hard = nt_xent_with_queue(z_a, z_b, torch.cat([z_a, z_b]), 0.1, 0.0)
    assert float(hard) > float(empty) + 0.1


def test_G5b_C4_gradients_finite_under_bf16():
    from tomocpt.networks.ssl.losses import nt_xent_with_queue

    lin = torch.nn.Linear(16, 16)
    z_a, z_b = _pairs()
    queue = torch.randn(64, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = nt_xent_with_queue(lin(z_a), lin(z_b), queue, 0.1, 1e-4)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(lin.weight.grad).all() and lin.weight.grad.abs().sum() > 0
