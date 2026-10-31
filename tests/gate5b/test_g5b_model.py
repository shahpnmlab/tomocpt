"""G5b-C4 (the queue on the model) and G5b-C5 (one v2 step: gradients, dense gradient into
``decoder2``). Tiny SSL v2 model, CPU fp32, ``default_collate`` of ``make_training_pair`` items
(``g5b_helpers.fit_ssl_v2``; fits built once per session by the ``g5b_fit`` fixture).

Queue contract (README.md): the model holds the queue of past global projections as a registered
**buffer** (not a parameter) whose name contains ``queue`` (any module of the model; floating,
2-D, ``[Q, D]`` or ``[D, Q]`` with ``D`` the projection size), with at most ``queue_size`` entries,
filled in training steps with detached copies of the ``ProjectionHead`` outputs (any subset of the
step's rows, raw or L2-normalised), oldest entries dropped first. Other bookkeeping buffers (a
pointer or a count) are allowed. Whether the buffer is persistent is not tested (note: the frozen
Gate 5 C14 transfer test requires that the checkpoint keys the picker leaves behind are exactly the
SSL head modules' state; a persistent queue must therefore live inside a head module).
"""

import pytest
import torch
import torch.nn.functional as F

from gate5 import g5_helpers as H
from gate5b import g5b_helpers as HB
from utils.tiny import applied_main_config


def _the_queue(model):
    qs = HB.queue_buffers(model)
    assert len(qs) == 1, f"expected one floating 2-D buffer named *queue*, found {[n for n, _ in qs]}"
    return qs[0]


def _proj_dim(model):
    return int(model.hparams.config.train.network.SELF_SUPERVISED_EMBEDING_SIZE)


# --------------------------------------------------------------------------------------
# G5b-C4 (model)
# --------------------------------------------------------------------------------------

def test_G5b_C4_queue_is_a_detached_buffer_capped_at_queue_size(tmp_path):
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    with applied_main_config(H.ssl_v2_main_config(tmp_path)):  # every SSL v2 default
        model = SelfSupervisedModel(lr=1e-3, train_continue=None)
    assert H.resolved(model.hparams.config.train.ssl).queue_size == 1024
    name, buf = _the_queue(model)
    params = dict(model.named_parameters())
    assert name not in params
    assert not any("queue" in n.lower() for n in params), "the queue must not be a parameter"
    assert not any(p is buf or p.data_ptr() == buf.data_ptr() for p in model.parameters())
    assert not buf.requires_grad and buf.grad_fn is None
    rows = HB.queue_rows(buf, _proj_dim(model))
    assert rows.shape[0] <= 1024


def test_G5b_C4_queue_is_fifo_detached_and_capped(g5b_fit):
    """``queue_size = 4``, 3 training steps of 2 pairs (4 projections per step, >= 2 per step
    enqueued under any reasonable policy): after the last step the queue holds exactly 4 entries,
    every one a projection of step 2 or 3, none of step 1, at least one of step 3."""
    fit = g5b_fit("default_q4")
    model = fit["model"]
    steps = [torch.cat(p) for p in fit["projections"]]
    assert len(steps) == 3 and all(s.shape[0] >= 4 for s in steps), [tuple(s.shape) for s in steps]
    _, buf = _the_queue(model)
    assert not buf.requires_grad and buf.grad_fn is None
    rows = HB.queue_rows(buf.detach().float().cpu(), steps[0].shape[1])
    assert rows.shape[0] == 4, f"queue holds {rows.shape[0]} entries, queue_size = 4"
    qn = F.normalize(rows, dim=1)

    def matches(step):
        return (qn @ F.normalize(step, dim=1).t()).max(dim=1).values > 0.9999

    old, mid, new = (matches(s) for s in steps)
    assert not old.any(), "entries of the oldest step survived (not FIFO, or not capped)"
    assert (mid | new).all(), "queue entries that are not recent projections"
    assert new.any(), "the newest projections were not enqueued"


# --------------------------------------------------------------------------------------
# G5b-C5
# --------------------------------------------------------------------------------------

def test_G5b_C5_every_parameter_gets_a_gradient_each_step(g5b_fit):
    """DDP rule with every SSL v2 default (VICReg, queue, dense_grad_scale = 1): every trainable
    parameter has a finite gradient in each of the 3 steps (the queue is empty in step 1 and full
    in step 3)."""
    fit = g5b_fit("default_q4")
    assert len(fit["grads"]) == 3
    for i, grads in enumerate(fit["grads"]):
        missing = sorted(n for n, g in grads.items() if g is None)
        assert not missing, f"step {i}: parameters without a gradient: {missing[:10]} ({len(missing)})"
        bad = sorted(n for n, g in grads.items() if not torch.isfinite(g).all())
        assert not bad, f"step {i}: non-finite gradients: {bad[:10]}"
    for cls in H.head_classes():
        assert any(isinstance(m, cls) for m in fit["model"].modules()), cls.__name__


def _decoder2_grad_norm(fit):
    grads = fit["grads"][0]
    names = [n for n in grads if ".decoder2." in f".{n}"]
    assert names, "no decoder2 parameters"
    return sum(0.0 if grads[n] is None else float(grads[n].double().norm()) for n in names)


def test_G5b_C5_dense_loss_reaches_decoder2(g5b_fit):
    """Only the dense InfoNCE task reaches the network (recon, global and scale weights 0, no
    VICReg). With ``dense_grad_scale = 1`` ``decoder2`` gets a non-zero gradient; with 0 (head
    only, control) it gets none."""
    full = g5b_fit("dense_only_full")
    assert H.resolved(full["model"].hparams.config.train.ssl).dense_grad_scale == 1.0
    assert _decoder2_grad_norm(full) > 1e-8
    head = g5b_fit("dense_only_head")
    assert _decoder2_grad_norm(head) == 0.0
