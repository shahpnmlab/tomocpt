"""Gate 8b.4: fused window attention (spec G8b-C4, C6), on by default (human decision 9 Nov 2026:
speed-ups are not opt-in; no config field).

``tomocpt.networks.attention.SDPAWindowAttention`` (a ``monai`` ``WindowAttention`` subclass) and
``swap_window_attention(module) -> int`` (number of modules swapped, in place). Every
``MySwinUNETR`` / ``ScaleConditionedSwinUNETR`` built by tomocpt uses it.
"""
import pytest
import torch
import torch.nn.functional as F

CUDA = torch.cuda.is_available()


def _eager_and_sdpa(dim=36, heads=3, window=(7, 7, 7), attn_drop=0.0, seed=0):
    from monai.networks.nets.swin_unetr import WindowAttention

    from tomocpt.networks.attention import SDPAWindowAttention, swap_window_attention

    torch.manual_seed(seed)
    eager = WindowAttention(dim, heads, window, qkv_bias=True, attn_drop=attn_drop)
    with torch.no_grad():
        eager.relative_position_bias_table.normal_(0, 0.5)
    holder = torch.nn.Sequential(type(eager)(dim, heads, window, qkv_bias=True, attn_drop=attn_drop))
    holder[0].load_state_dict(eager.state_dict())
    assert swap_window_attention(holder) == 1
    assert isinstance(holder[0], SDPAWindowAttention)
    return eager, holder[0]


def _shift_mask(nw, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(nw, n, n)
    m[torch.rand(nw, n, n, generator=g) < 0.3] = -100.0
    return m


def test_G8b_C4_state_dict_keys_and_shapes_unchanged():
    eager, fused = _eager_and_sdpa()
    a, b = eager.state_dict(), fused.state_dict()
    assert list(a) == list(b)
    assert all(a[k].shape == b[k].shape for k in a)


@pytest.mark.parametrize("shifted", [False, True])
def test_G8b_C4_cpu_fp32_output_and_grads_equal_eager(shifted):
    eager, fused = _eager_and_sdpa()
    n, nw, b = 343, 4, 8
    x = torch.randn(b, n, 36)
    mask = _shift_mask(nw, n) if shifted else None
    outs, grads = [], []
    for m in (eager, fused):
        xi = x.clone().requires_grad_(True)
        y = m(xi, mask)
        (y * torch.linspace(-1, 1, y.numel()).view_as(y)).sum().backward()
        outs.append(y.detach())
        grads.append({"x": xi.grad, **{k: p.grad.clone() for k, p in m.named_parameters()}})
        m.zero_grad()
    torch.testing.assert_close(outs[1], outs[0], atol=1e-5, rtol=0)
    for k in grads[0]:
        torch.testing.assert_close(grads[1][k], grads[0][k], atol=1e-5, rtol=1e-4, msg=k)


def test_G8b_C4_tiny_model_forward_equals_eager_and_checkpoints_interchange():
    from utils.tiny import tiny_swin_kwargs

    from monai.networks.nets.swin_unetr import WindowAttention

    from tomocpt.networks.attention import SDPAWindowAttention
    from tomocpt.networks.swinunetr import MySwinUNETR

    torch.manual_seed(0)
    model = MySwinUNETR(**tiny_swin_kwargs()).eval()
    attn = [m for m in model.modules() if isinstance(m, WindowAttention)]
    assert attn and all(isinstance(m, SDPAWindowAttention) for m in attn), "tomocpt models use fused attention"
    eager = MySwinUNETR(**tiny_swin_kwargs()).eval()
    for m in eager.modules():
        if isinstance(m, SDPAWindowAttention):
            m.__class__ = WindowAttention  # back to MONAI's forward
    eager.load_state_dict(model.state_dict(), strict=True)
    model.load_state_dict(eager.state_dict(), strict=True)
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], eager(x)[0], atol=1e-5, rtol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("shifted", [False, True])
def test_G8b_C4_gpu_bf16_close_to_eager(shifted):
    eager, fused = _eager_and_sdpa()
    eager, fused = eager.cuda(), fused.cuda()
    n, nw, b = 343, 8, 64
    x = torch.randn(b, n, 36, device="cuda")
    mask = _shift_mask(nw, n).cuda() if shifted else None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = eager(x, mask).float()
        f = fused(x, mask).float()
    assert float((a - f).abs().max()) <= 2e-2
    assert float(F.cosine_similarity(a.flatten(), f.flatten(), dim=0)) >= 0.9999


def test_G8b_C4_dropout_is_applied_in_training_only():
    _, fused = _eager_and_sdpa(attn_drop=0.5)
    x = torch.randn(4, 343, 36)
    fused.eval()
    with torch.no_grad():
        e1, e2 = fused(x, None), fused(x, None)
    assert torch.equal(e1, e2)
    fused.train()
    with torch.no_grad():
        t1, t2 = fused(x, None), fused(x, None)
    assert not torch.equal(t1, t2)


@pytest.mark.parametrize("case", ["sigmoid_uncond", "softmax_cond"])
def test_G8b_C6_fused_attention_every_parameter_gets_a_gradient(tmp_path, case):
    from gate3 import g3_helpers as G3
    from gate7 import g7_helpers as H7
    from gate7.test_g7_model import _common, _items_with_mask
    from utils.tiny import applied_main_config

    from tomocpt.networks.attention import SDPAWindowAttention
    from tomocpt.networks.pickingModel import BasePickingModel

    g7 = _common()
    head, cond = g7.CASES[case]
    cfg = H7.disc_config(tmp_path, g7.PARTICLES, g7.FILAMENTS, class_head=head, cond=cond, contrastive_weight=0.1)
    cfg.train.network.SwinUNETR.attn_drop_rate = 0.1
    cfg.train.particles.offset_weight = 1.0
    with applied_main_config(cfg):
        torch.manual_seed(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        assert any(isinstance(m, SDPAWindowAttention) for m in model.modules())
        loss, grads, _ = G3.first_step_loss_and_grads(model, _items_with_mask(g7), tmp_path / "tb")
    assert torch.isfinite(torch.tensor(loss))
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, missing[:10]
    tables = [g for n, g in grads.items() if n.endswith("relative_position_bias_table")]
    assert tables and all(torch.count_nonzero(g) > 0 for g in tables)
