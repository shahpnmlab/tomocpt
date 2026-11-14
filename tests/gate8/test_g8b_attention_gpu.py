"""Gate 8b.4 addition (9 Nov 2026): the fused path runs on CUDA only (CPU keeps MONAI's eager
attention, bit-identical to the legacy goldens), so the fused numerics are checked on the GPU in
float32 as well as in bf16 (``test_g8b_attention.py``)."""
import pytest
import torch

from gate8.test_g8b_attention import _eager_and_sdpa, _shift_mask

CUDA = torch.cuda.is_available()


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("shifted", [False, True])
def test_G8b_C4_gpu_fp32_output_and_grads_equal_eager(shifted):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    eager, fused = (m.cuda() for m in _eager_and_sdpa())
    n, nw, b = 343, 4, 8
    x = torch.randn(b, n, 36, device="cuda")
    mask = _shift_mask(nw, n).cuda() if shifted else None
    outs, grads = [], []
    for m in (eager, fused):
        xi = x.clone().requires_grad_(True)
        y = m(xi, mask)
        (y * torch.linspace(-1, 1, y.numel(), device="cuda").view_as(y)).sum().backward()
        outs.append(y.detach())
        grads.append({"x": xi.grad, **{k: p.grad.clone() for k, p in m.named_parameters()}})
    torch.testing.assert_close(outs[1], outs[0], atol=1e-5, rtol=0)
    for k in grads[0]:
        a, f = grads[0][k], grads[1][k]
        assert float((f - a).abs().max()) <= 1e-5 * float(a.abs().max()) + 1e-7, k


def test_G8b_C4_cpu_runs_the_eager_attention():
    eager, fused = _eager_and_sdpa()
    x = torch.randn(4, 343, 36)
    with torch.no_grad():
        assert torch.equal(fused(x, None), eager(x, None))
        m = _shift_mask(2, 343)
        assert torch.equal(fused(x, m), eager(x, m))
