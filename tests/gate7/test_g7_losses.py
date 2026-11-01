"""G7-C4 ``ContrastiveMemoryBank``, G7-C5 ``supcon_loss``, G7-C6 ``focal_heatmap_loss`` and the
Huber value term (signatures: ``g7_helpers``)."""


import numpy as np
import pytest
import torch

from gate3 import g3_helpers as G3
from gate7 import g7_helpers as H


# --------------------------------------------------------------------------------------
# G7-C4
# --------------------------------------------------------------------------------------

def _rows(emb, lab, label):
    e = emb[lab == label]
    return sorted(e[:, 0].tolist())


def test_G7_C4_bank_fifo_detached_capped_buffer():
    from tomocpt.training.discriminative import ContrastiveMemoryBank

    bank = ContrastiveMemoryBank(n_classes=2, size=8, dim=4)
    assert isinstance(bank, torch.nn.Module)
    assert len(list(bank.parameters())) == 0, "the bank must not hold parameters"

    def vecs(ks):
        v = torch.zeros(len(ks), 4)
        v[:, 0] = torch.tensor(ks, dtype=torch.float32)
        return v

    src = vecs(list(range(5))).requires_grad_(True)
    bank.push(src * 1.0, torch.zeros(5, dtype=torch.long))
    bank.push(vecs([5, 6, 7, 8]), torch.zeros(4, dtype=torch.long))
    bank.push(vecs([9, 10, 11]), torch.zeros(3, dtype=torch.long))
    bank.push(vecs([100, 101, 102]), torch.full((3,), -1, dtype=torch.long))
    bank.push(vecs(list(range(200, 220))), torch.ones(20, dtype=torch.long))
    emb, lab = bank.get()
    assert emb.shape[1] == 4 and lab.shape == (emb.shape[0],)
    assert not emb.requires_grad and emb.grad_fn is None, "stored embeddings must be detached"
    assert _rows(emb, lab, 0) == [float(k) for k in range(4, 12)], "class queue is not FIFO / not capped"
    assert _rows(emb, lab, 1) == [float(k) for k in range(212, 220)], "a push larger than the queue keeps the newest"
    assert _rows(emb, lab, -1) == [100.0, 101.0, 102.0]
    assert emb.shape[0] == 8 + 8 + 3

    sd = bank.state_dict()
    assert len(sd) > 0, "the queues must be registered buffers (stored in state_dict)"
    other = ContrastiveMemoryBank(n_classes=2, size=8, dim=4)
    other.load_state_dict(sd)
    e2, l2 = other.get()
    for label in (0, 1, -1):
        assert _rows(e2, l2, label) == _rows(emb, lab, label)


def test_G7_C4_negative_queue_is_capped():
    from tomocpt.training.discriminative import ContrastiveMemoryBank

    bank = ContrastiveMemoryBank(n_classes=1, size=5, dim=3)
    for k in range(4):
        bank.push(torch.full((3, 3), float(k)), torch.full((3,), -1, dtype=torch.long))
    emb, lab = bank.get()
    assert int((lab == -1).sum()) == 5
    assert int((lab == 0).sum()) == 0


# --------------------------------------------------------------------------------------
# G7-C5
# --------------------------------------------------------------------------------------

def supcon_reference(z, y, bz, by, tau):
    """Khosla et al. 2020, L_out, float64. Anchors: batch entries with label >= 0 that have at
    least one positive (same label) among the other batch entries and the bank. The denominator
    runs over every other batch entry and every bank entry. Mean over anchors; 0 if none."""
    z = np.asarray(z, np.float64)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    allz, ally = z, np.asarray(y)
    if bz is not None and len(bz):
        bz = np.asarray(bz, np.float64)
        bz = bz / np.linalg.norm(bz, axis=1, keepdims=True)
        allz, ally = np.concatenate([z, bz]), np.concatenate([ally, np.asarray(by)])
    losses = []
    for i in range(len(z)):
        if y[i] < 0:
            continue
        others = [j for j in range(len(allz)) if j != i]
        pos = [j for j in others if ally[j] == y[i]]
        if not pos:
            continue
        s = allz[others] @ z[i] / tau
        logden = np.log(np.exp(s - s.max()).sum()) + s.max()
        lp = [(allz[j] @ z[i]) / tau - logden for j in pos]
        losses.append(-np.mean(lp))
    return float(np.mean(losses)) if losses else 0.0


def test_G7_C5_reference_toy_case():
    from tomocpt.training.discriminative import supcon_loss

    rng = np.random.default_rng(0)
    z = rng.normal(size=(11, 8))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = np.array([0, 0, 1, 1, 1, 2, -1, -1, 0, 2, 3])  # class 3 has no positive -> not an anchor
    bz = rng.normal(size=(6, 8))
    bz /= np.linalg.norm(bz, axis=1, keepdims=True)
    by = np.array([0, 1, -1, 2, 2, -1])
    zt, bzt = torch.tensor(z, dtype=torch.float32), torch.tensor(bz, dtype=torch.float32)
    for tau in (0.1, 0.5):
        ref = supcon_reference(zt.double().numpy(), y, bzt.double().numpy(), by, tau)
        got = supcon_loss(zt, torch.tensor(y), bzt, torch.tensor(by), temperature=tau)
        assert float(got) == pytest.approx(ref, rel=1e-5), (tau, float(got), ref)
    ref = supcon_reference(zt.double().numpy(), y, None, None, 0.1)
    got = supcon_loss(zt, torch.tensor(y), temperature=0.1)
    assert float(got) == pytest.approx(ref, rel=1e-5)


def test_G7_C5_separated_classes_give_lower_loss_than_shuffled_labels():
    from tomocpt.training.discriminative import supcon_loss

    g = torch.Generator().manual_seed(0)
    y = torch.arange(4).repeat_interleave(8)
    z = torch.nn.functional.one_hot(y, 16).float() + 0.1 * torch.randn(32, 16, generator=g)
    sep = float(supcon_loss(z, y, temperature=0.1))
    shuffled = [float(supcon_loss(z, y[torch.randperm(32, generator=g)], temperature=0.1)) for _ in range(5)]
    assert sep < min(shuffled), (sep, shuffled)


def test_G7_C5_finite_under_bf16():
    from tomocpt.training.discriminative import supcon_loss

    g = torch.Generator().manual_seed(1)
    base = torch.randn(1, 64, generator=g)
    z = (base + 1e-3 * torch.randn(48, 64, generator=g)).to(torch.bfloat16).requires_grad_(True)
    y = torch.arange(3).repeat(16)
    bz = (base + 1e-3 * torch.randn(32, 64, generator=g)).to(torch.bfloat16)
    by = torch.cat([torch.arange(3).repeat(10), torch.tensor([-1, -1])])
    loss = supcon_loss(z, y, bz, by, temperature=0.1)
    assert torch.isfinite(loss).all()
    loss.float().backward()
    assert z.grad is not None and torch.isfinite(z.grad.float()).all()


def test_G7_C5_no_anchor_gives_zero_with_gradient_path():
    from tomocpt.training.discriminative import supcon_loss

    z = torch.randn(6, 8, requires_grad=True)
    for y in (torch.full((6,), -1, dtype=torch.long), torch.arange(6)):  # only negatives / no positives
        z.grad = None
        loss = supcon_loss(z, y, temperature=0.1)
        assert float(loss) == 0.0
        assert loss.requires_grad
        loss.backward()
        assert z.grad is not None and torch.count_nonzero(z.grad) == 0


# --------------------------------------------------------------------------------------
# G7-C6
# --------------------------------------------------------------------------------------

def focal_reference(p, y, avail, pos, n, a, b):
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    B, C = avail.shape
    terms = []
    for i in range(B):
        for c in range(C):
            if not avail[i, c]:
                continue
            pp, yy, mm = p[i, c], y[i, c], pos[i, c]
            s = np.sum(((1 - pp) ** a * np.log(pp))[mm])
            s += np.sum(((1 - yy) ** b * pp ** a * np.log(1 - pp))[~mm])
            terms.append(-s / max(float(n[i, c]), 1.0))
    return float(np.mean(terms)) if terms else 0.0


@pytest.mark.parametrize("alpha,beta", [(2.0, 4.0), (1.5, 3.0)])
def test_G7_C6_focal_equals_centernet_formula(alpha, beta):
    from tomocpt.training.discriminative import focal_heatmap_loss

    rng = np.random.default_rng(0)
    shape = (2, 2, 6, 6, 6)
    p = rng.uniform(0.02, 0.98, shape).astype(np.float32)
    y = rng.uniform(0.0, 0.9, shape).astype(np.float32)
    pos = np.zeros(shape, bool)
    pos[0, 0, 1, 2, 3] = pos[0, 0, 4, 4, 1] = True
    pos[1, 0, 2, 2, 2] = True
    pos[1, 1, 0, 5, 5] = pos[1, 1, 3, 1, 4] = pos[1, 1, 5, 5, 0] = True
    y[pos] = 0.8  # an off-grid peak voxel: the positive set comes from ``positive``, not y == 1
    avail = np.array([[True, False], [True, True]])
    n = np.array([[2.0, 0.0], [1.0, 3.0]], np.float32)
    got = focal_heatmap_loss(torch.from_numpy(p), torch.from_numpy(y), torch.from_numpy(avail),
                             torch.from_numpy(pos), torch.from_numpy(n), alpha=alpha, beta=beta)
    ref = focal_reference(p, y, avail, pos, n, alpha, beta)
    assert float(got) == pytest.approx(ref, rel=1e-6)
    # an unavailable pair contributes nothing, whatever its values
    p2 = p.copy()
    p2[0, 1] = 0.5
    got2 = focal_heatmap_loss(torch.from_numpy(p2), torch.from_numpy(y), torch.from_numpy(avail),
                              torch.from_numpy(pos), torch.from_numpy(n), alpha=alpha, beta=beta)
    assert float(got2) == float(got)


def test_G7_C6_huber_value_term_equals_gate6(tmp_path):
    """With ``value_loss = "huber"`` (default and explicit) the model's v2 loss is
    ``multiclass_picking_loss`` exactly (Gate 6)."""
    from utils.tiny import applied_main_config

    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.training.multiclass_loss import class_probabilities, multiclass_picking_loss, renormalise_targets

    particles = ((1, "A", 150.0), (2, "B", 300.0))
    batch = G3.v2_batch([15.0, 30.0], B=2, seed=3)
    for head in ("sigmoid", "softmax"):
        for explicit in (False, True):
            cfg = H.disc_config(tmp_path / f"{head}{explicit}", particles, class_head=head)
            if explicit:
                cfg.train.discriminative.value_loss = "huber"
            with applied_main_config(cfg):
                torch.manual_seed(0)
                model = BasePickingModel(lr=1e-3, train_continue=None)
            logits = torch.randn(2, 2, 64, 64, 64, generator=torch.Generator().manual_seed(4))
            y = batch["target_data"]
            if head == "softmax":
                yp, yt = class_probabilities(logits, "softmax", batch["class_available"]), renormalise_targets(y)
            else:
                yp, yt = torch.sigmoid(logits), y
            ref = multiclass_picking_loss(yp, yt, batch["class_available"], batch["class_d_vox"])
            got = model.multiclass_loss(yp, yt, batch)
            assert torch.equal(got, ref), (head, explicit, float(got), float(ref))


# --------------------------------------------------------------------------------------
# G7-C6b: reconciled focal value term (human decision, 1 Nov 2026; README "Reconciled focal")
# --------------------------------------------------------------------------------------

def test_G7_C6b_soft_focal_minimiser_is_the_target():
    """``|y - p|^g * BCE(p, y)`` is >= 0, 0 at ``p = y`` and minimised there (Huber's minimiser)."""
    from tomocpt.training.discriminative import soft_focal_penalty

    p = torch.linspace(1e-4, 1 - 1e-4, 20001, dtype=torch.float64)
    for y in (0.0, 0.1, 0.37, 0.5, 0.8, 1.0):
        q = soft_focal_penalty(p, torch.full_like(p, y)).double()
        assert float(q.min()) >= 0.0
        assert abs(float(p[q.argmin()]) - y) <= 1e-3, y
    y = torch.tensor([0.0, 0.25, 0.6, 1.0])
    assert float(soft_focal_penalty(y.clamp(1e-6, 1 - 1e-6), y).abs().max()) < 1e-9


def _value_reference(p, y, avail, sel, gamma, k=0.0):
    """float64 numpy: per-pair voxel means of the Section 1.5 weights times the soft focal penalty
    (``sel``) or Huber (delta 1); returns ``focal_mean - huber_mean`` over available pairs."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    B, C = avail.shape
    out = []
    for b in range(B):
        for c in range(C):
            if not avail[b, c]:
                continue
            yy, pp = y[b, c], p[b, c]
            n, s = yy.size, yy.sum()
            w = np.where(yy > 0, n / (s + 1), 2 * n / (n - s + 1))
            d = np.abs(pp - yy)
            hub = np.where(d <= 1, 0.5 * d ** 2, d - 0.5)
            q = np.abs(yy - pc[b, c]) ** gamma * -(yy * np.log(pc[b, c]) + (1 - yy) * np.log1p(-pc[b, c]))
            pen = q if sel[c] else hub
            out.append(np.mean(w * pen) - np.mean(w * hub))
    return float(np.mean(out))


@pytest.mark.parametrize("gamma", [2.0, 1.5])
def test_G7_C6b_focal_value_term_is_huber_weights_times_soft_focal(gamma):
    """``multiclass_picking_loss(value_loss="focal")`` minus the Huber loss equals, per available pair,
    ``mean(w * q) - mean(w * huber)`` with the Section 1.5 weights (Hessian term unchanged);
    channels outside ``focal_channels`` keep the Huber penalty."""
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    batch = G3.v2_batch([15.0, 30.0, 7.0], B=2, seed=5, available=[[True, True, True], [True, False, True]])
    y = batch["target_data"]
    p = torch.rand(y.shape, generator=torch.Generator().manual_seed(6)) * 0.98 + 0.01
    av, dv = batch["class_available"], batch["class_d_vox"]
    sel = [True, True, False]
    hub = multiclass_picking_loss(p, y, av, dv, checkpoint=False)
    foc = multiclass_picking_loss(p, y, av, dv, checkpoint=False, value_loss="focal",
                                  focal_channels=sel, focal_gamma=gamma)
    ref = _value_reference(p.numpy(), y.numpy(), av.numpy(), sel, gamma)
    assert float(foc - hub) == pytest.approx(ref, rel=1e-5, abs=1e-6)
    none_sel = multiclass_picking_loss(p, y, av, dv, checkpoint=False, value_loss="focal",
                                       focal_channels=[False, False, False])
    assert torch.equal(none_sel, hub)


def test_G7_C6b_model_uses_soft_focal_on_particles_and_huber_on_filaments(tmp_path):
    from utils.tiny import applied_main_config

    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    particles, filaments = ((1, "A", 150.0), (2, "B", 300.0)), ((3, "mt", 70.0),)
    batch = G3.v2_batch([15.0, 30.0, 7.0], B=2, seed=3)
    cfg = H.disc_config(tmp_path, particles, filaments, class_head="sigmoid", value_loss="focal")
    with applied_main_config(cfg):
        torch.manual_seed(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
    logits = torch.randn(2, 3, 64, 64, 64, generator=torch.Generator().manual_seed(4))
    yp, yt = torch.sigmoid(logits), batch["target_data"]
    ref = multiclass_picking_loss(yp, yt, batch["class_available"], batch["class_d_vox"],
                                  value_loss="focal", focal_channels=[True, True, False])
    assert torch.equal(model.multiclass_loss(yp, yt, batch), ref)
    huber = multiclass_picking_loss(yp, yt, batch["class_available"], batch["class_d_vox"])
    assert not torch.equal(ref, huber)
