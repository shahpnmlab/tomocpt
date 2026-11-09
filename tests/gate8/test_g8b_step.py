"""Gate 8b.1-8b.3: training step without behaviour change (spec ``08_AGENT_PERFORMANCE.md``).

* G8b-C1 new fields exist with defaults that reproduce today's behaviour.
* G8b-C2 ``SwinUNETR.use_checkpoint`` on vs off: logits, loss, gradients bit-identical (CPU fp32,
  dropout on) over 3 optimiser steps.
* G8b-C3 ``multiclass_picking_loss(..., hessian_block=k)`` vs ``k=1`` (CPU E1 tolerance); an
  unavailable class keeps a (zero) gradient path; the model passes ``train.particles.hessian_block``;
  (slow, GPU) no host sync in the loss.
* 8b.3 ``Data(pin_memory=, prefetch_factor=)`` reach the ``DataLoader``; the EMA update is
  bit-identical to the per-tensor loop.

Orchestrator rulings on names (the spec names the fields, not the ``Data`` arguments):
``Data(..., pin_memory: bool = False, prefetch_factor: Optional[int] = None)``; ``prefetch_factor``
is passed to the ``DataLoader`` only when it is not ``None`` and ``workers_for_data > 0``.
"""
import copy

import pytest
import torch

CUDA = torch.cuda.is_available()

# CPU fp32 E1 tolerances (spec Section 3.1)
LOSS_RTOL = 1e-6
GRAD_ATOL = 1e-6
GRAD_SUM_RTOL = 1e-4


# --------------------------------------------------------------------------------------
# G8b-C1
# --------------------------------------------------------------------------------------

def test_G8b_C1_new_fields_and_defaults():
    from tomocpt.mainConfig import MainConfig

    cfg = MainConfig()
    assert cfg.train.particles.hessian_block == 1
    assert cfg.train.pin_memory is False
    assert cfg.train.prefetch_factor is None


def test_G8b_C1_loss_default_is_the_class_loop():
    import inspect

    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    assert inspect.signature(multiclass_picking_loss).parameters["hessian_block"].default == 1


# --------------------------------------------------------------------------------------
# G8b-C2
# --------------------------------------------------------------------------------------

def _swin(use_checkpoint: bool):
    from utils.tiny import tiny_swin_kwargs

    from tomocpt.networks.swinunetr import MySwinUNETR

    kw = tiny_swin_kwargs()
    kw.update(use_checkpoint=use_checkpoint, drop_rate=0.3, attn_drop_rate=0.3, dropout_path_rate=0.2)
    torch.manual_seed(0)
    return MySwinUNETR(**kw)


@pytest.fixture
def one_thread():
    """The CPU backward of Swin's ``relative_position_bias_table`` (indexed accumulation) is not
    bit-reproducible run to run with several threads, with or without checkpointing (measured
    9 Nov 2026); with one thread it is."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_G8b_C2_swin_checkpointing_is_bit_identical(one_thread):
    runs = {}
    for flag in (True, False):
        model = _swin(flag).train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        g = torch.Generator().manual_seed(5)
        rec = []
        torch.manual_seed(123)  # dropout stream
        for _ in range(3):
            x = torch.randn(1, 1, 64, 64, 64, generator=g)
            y = torch.rand(1, 1, 64, 64, 64, generator=g)
            logits = model(x)[0]
            loss = torch.nn.functional.huber_loss(torch.sigmoid(logits), y)
            opt.zero_grad()
            loss.backward()
            rec.append((logits.detach().clone(), loss.detach().clone(),
                        {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}))
            opt.step()
        runs[flag] = rec
    for (l1, s1, g1), (l0, s0, g0) in zip(runs[True], runs[False]):
        assert torch.equal(l1, l0)
        assert torch.equal(s1, s0)
        assert g1.keys() == g0.keys()
        for n in g1:
            assert torch.equal(g1[n], g0[n]), n


# --------------------------------------------------------------------------------------
# G8b-C3
# --------------------------------------------------------------------------------------

def _loss_inputs(C: int, head: str, seed: int = 0, B: int = 3, size: int = 24):
    from tomocpt.training.multiclass_loss import class_probabilities

    g = torch.Generator().manual_seed(seed)
    logits = (2 * torch.randn(B, C, size, size, size, generator=g)).requires_grad_(True)
    y = torch.zeros(B, C, size, size, size)
    ax = torch.arange(size, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
    for b in range(B):
        for c in range(C):
            for _ in range(2):
                ctr = torch.randint(4, size - 4, (3,), generator=g).float()
                r2 = (zz - ctr[0]) ** 2 + (yy - ctr[1]) ** 2 + (xx - ctr[2]) ** 2
                y[b, c] = torch.maximum(y[b, c], torch.exp(-r2 / (2 * (2.0 + c) ** 2)))
    available = torch.rand(B, C, generator=g) > 0.3
    available[0, 0] = True
    if C >= 3:
        available[:, C - 1] = False  # a class unavailable in the whole batch
    d_vox = 10 + 20 * torch.rand(B, C, generator=g)
    probs = class_probabilities(logits, head, available if head == "softmax" else None)
    return logits, probs, y, available, d_vox


def _loss_and_grad(C, head, block, checkpoint, extra):
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    logits, probs, y, available, d_vox = _loss_inputs(C, head)
    loss = multiclass_picking_loss(probs, y, available, d_vox, checkpoint=checkpoint, hessian_block=block, **extra)
    loss.backward()
    return loss.detach(), logits.grad.detach()


EXTRAS = {
    "plain": {},
    "weights_cross": {"class_weights": "auto", "cross_class_neg_weight": 0.5},
    "focal_core": {"value_loss": "focal", "focal_channels": "auto", "focal_gamma": 2.0,
                   "core_weight": 2.0, "core_power": 4.0},
}


def _extra(name, C):
    e = dict(EXTRAS[name])
    if e.get("class_weights") == "auto":
        e["class_weights"] = [1.0 + 0.5 * c for c in range(C)]
    if e.get("focal_channels") == "auto":
        e["focal_channels"] = [c % 2 == 0 for c in range(C)]
    return e


@pytest.mark.parametrize("C", [1, 3, 7])
@pytest.mark.parametrize("head", ["sigmoid", "softmax"])
@pytest.mark.parametrize("block", [0, 2, 3])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("extra", list(EXTRAS))
def test_G8b_C3_hessian_block_matches_class_loop(C, head, block, checkpoint, extra):
    kw = _extra(extra, C)
    ref_loss, ref_grad = _loss_and_grad(C, head, 1, checkpoint, kw)
    loss, grad = _loss_and_grad(C, head, block, checkpoint, kw)
    torch.testing.assert_close(loss, ref_loss, rtol=LOSS_RTOL, atol=0)
    torch.testing.assert_close(grad, ref_grad, atol=GRAD_ATOL, rtol=0)
    s_ref, s = ref_grad.abs().sum(), grad.abs().sum()
    assert abs(float(s - s_ref)) <= GRAD_SUM_RTOL * float(s_ref) + 1e-12


@pytest.mark.parametrize("block", [1, 0, 2])
def test_G8b_C3_unavailable_class_keeps_a_zero_gradient_path(block):
    """One parameter per channel: the channel unavailable in the whole batch gets a zero
    gradient, not ``None`` (invariant 5)."""
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    C, B, s = 3, 2, 16
    params = [torch.nn.Parameter(torch.tensor(0.5 + 0.1 * c)) for c in range(C)]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, 1, s, s, s, generator=g)
    probs = torch.sigmoid(torch.cat([p * x for p in params], dim=1))
    y = torch.rand(B, C, s, s, s, generator=g)
    available = torch.tensor([[True, True, False], [True, False, False]])
    loss = multiclass_picking_loss(probs, y, available, torch.full((B, C), 20.0), hessian_block=block)
    loss.backward()
    assert params[2].grad is not None and float(params[2].grad) == 0.0
    assert float(params[0].grad) != 0.0


def test_G8b_C3_no_class_available_is_zero_with_gradient_path():
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    logits = torch.randn(2, 2, 16, 16, 16, requires_grad=True)
    for block in (1, 0):
        loss = multiclass_picking_loss(torch.sigmoid(logits), torch.rand(2, 2, 16, 16, 16),
                                       torch.zeros(2, 2, dtype=torch.bool), torch.full((2, 2), 20.0),
                                       hessian_block=block)
        assert float(loss) == 0.0
        loss.backward()
        assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_G8b_C3_model_passes_hessian_block(tmp_path, monkeypatch):
    from gate6 import g6_helpers as H6
    from utils.tiny import applied_main_config

    import tomocpt.networks.pickingModel as pm

    seen = {}
    real = pm.multiclass_picking_loss

    def spy(*a, **k):
        seen.setdefault("hessian_block", k.get("hessian_block", "missing"))
        return real(*a, **k)

    monkeypatch.setattr(pm, "multiclass_picking_loss", spy)
    cfg = H6.v2_config_with_filaments(tmp_path, ((1, "a", 120.0), (2, "b", 200.0)), ())
    cfg.train.particles.hessian_block = 0
    with applied_main_config(cfg):
        model = pm.BasePickingModel(lr=1e-3, train_continue=None)
        B = 2
        batch = {"class_available": torch.ones(B, 2, dtype=torch.bool), "class_d_vox": torch.full((B, 2), 20.0)}
        model.multiclass_loss(torch.rand(B, 2, 16, 16, 16), torch.rand(B, 2, 16, 16, 16), batch)
    assert seen["hessian_block"] == 0


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("block", [1, 0])
def test_G8b_C3_loss_has_no_host_sync(block):
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(2, 3, 32, 32, 32, device=dev, generator=g, requires_grad=True)
    y = torch.rand(2, 3, 32, 32, 32, device=dev, generator=g)
    available = torch.tensor([[True, True, False], [True, False, False]], device=dev)
    d_vox = torch.full((2, 3), 20.0, device=dev)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        loss = multiclass_picking_loss(torch.sigmoid(logits), y, available, d_vox, hessian_block=block)
        loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(loss.detach().cpu())


# --------------------------------------------------------------------------------------
# 8b.3 loader flags, EMA
# --------------------------------------------------------------------------------------

def test_G8b_C1_loader_flags_reach_the_dataloader():
    from torch.utils.data import TensorDataset

    from tomocpt.dataManager.dataloading import Data

    ds = TensorDataset(torch.zeros(4, 1))
    default = Data("unused", True, 2, 2)._get_dataloader(ds)
    assert default.pin_memory is False
    assert default.prefetch_factor == 2  # torch default with workers
    dl = Data("unused", True, 2, 2, pin_memory=True, prefetch_factor=6)._get_dataloader(ds)
    assert dl.pin_memory is True and dl.prefetch_factor == 6
    dl0 = Data("unused", True, 2, 0, pin_memory=True, prefetch_factor=6)._get_dataloader(ds)
    assert dl0.pin_memory is True and dl0.prefetch_factor is None  # no workers: torch rejects a factor


def test_G8b_C1_run_passes_loader_flags(tmp_path, monkeypatch):
    """``training/run.py`` passes ``train.pin_memory`` / ``train.prefetch_factor`` to ``Data``."""
    import inspect

    import tomocpt.training.run as run

    src = inspect.getsource(run)
    assert "pin_memory=config.pin_memory" in src
    assert "prefetch_factor=config.prefetch_factor" in src


def test_G8b_ema_update_is_bit_identical_to_the_tensor_loop():
    from tomocpt.training.averaging import WeightAveraging, ema_decay_at

    torch.manual_seed(0)
    module = torch.nn.Sequential(torch.nn.Conv3d(1, 4, 3), torch.nn.InstanceNorm3d(4, affine=True),
                                 torch.nn.Conv3d(4, 2, 1))
    cb = WeightAveraging(mode="ema", ema_decay=0.999)
    cb.on_fit_start(None, module)
    ref = {k: v.detach().clone() for k, v in module.state_dict().items() if v.is_floating_point()}
    ref_n = 0
    for step in range(5):
        with torch.no_grad():
            for p in module.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        cb.on_train_batch_end(None, module, None, None, step)
        d = ema_decay_at(ref_n, 0.999)
        for k, v in module.state_dict().items():
            if k in ref:
                ref[k].mul_(d).add_(v.detach().to(ref[k].dtype), alpha=1.0 - d)
        ref_n += 1
    assert cb.n_updates == ref_n
    for k in ref:
        assert torch.equal(cb.averaged[k], ref[k]), k


# --------------------------------------------------------------------------------------
# G8b-C6 (8b.1-8b.3 options; 8b.4-8b.6 options get their own test when they exist)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["sigmoid_uncond", "softmax_cond"])
def test_G8b_C6_options_on_every_parameter_gets_a_gradient(tmp_path, case):
    from gate3 import g3_helpers as G3
    from gate7 import g7_helpers as H7
    from gate7.test_g7_model import _common, _items_with_mask
    from utils.tiny import applied_main_config

    from tomocpt.networks.pickingModel import BasePickingModel

    g7 = _common()
    head, cond = g7.CASES[case]
    cfg = H7.disc_config(tmp_path, g7.PARTICLES, g7.FILAMENTS, class_head=head, cond=cond, contrastive_weight=0.1)
    cfg.train.network.SwinUNETR.use_checkpoint = False
    cfg.train.particles.loss_checkpoint = False
    cfg.train.particles.hessian_block = 0
    cfg.train.particles.offset_weight = 1.0
    with applied_main_config(cfg):
        torch.manual_seed(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        assert model.offset_head is not None and model.contrastive_head is not None
        loss, grads, trainer = G3.first_step_loss_and_grads(model, _items_with_mask(g7), tmp_path / "tb")
    assert torch.isfinite(torch.tensor(loss))
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, f"parameters without a gradient: {missing[:10]} ({len(missing)} in all)"
    assert not H7.trainable_not_in_optimizer(model, trainer)
