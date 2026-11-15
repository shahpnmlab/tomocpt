"""Large class counts (TODO 5.1 smoke run, C = 134, 15 Nov 2026).

* G8-L1 contrastive anchors and focal positives come from per-channel peak masks with the same
  result as the all-channel mask (which needed several GB at C = 134).
* G8-L2 the loss is checkpointed automatically only when the Hessian term's saved activations
  would not fit (CZII B = 16 and C = 134 B <= 8 stay off on a 48 GB GPU, C = 134 B = 16 is on,
  never on the CPU); checkpointing does not change the loss or the gradients.
"""

import types

import pytest
import torch


def _target(B=3, C=5, n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.zeros(B, C, n, n, n)
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float32),) * 3, indexing="ij")
    for b in range(B):
        for c in range(C - 1):  # last channel stays empty: skipped by the per-channel path
            for _ in range(3):
                z, y, x = (torch.rand(3, generator=g) * (n - 1)).tolist()
                r2 = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
                t[b, c] = torch.maximum(t[b, c], torch.exp(-r2 / 4.0))
    t[1, 2] *= 0.3  # below min_target in one pair
    return t


def _reference_anchors(target, available, d_vox, is_filament, min_target, max_per_class, generator):
    """The pre-fix sample_anchors: one max-pool over all channels."""
    from tomocpt.training import discriminative as D

    B, C = target.shape[:2]
    peaks = D._peak_mask(target, min_target)
    idx, lab = [], []
    for c in range(C):
        rows = []
        for b in range(B):
            if not bool(available[b, c]):
                continue
            inst = D._instances(target, peaks[:, c], b, c, float(d_vox[b, c]), bool(is_filament[c]))
            rows.extend([b, *map(int, v)] for v in inst)
        if len(rows) > int(max_per_class):
            perm = torch.randperm(len(rows), generator=generator)[: int(max_per_class)].tolist()
            rows = [rows[i] for i in sorted(perm)]
        idx.extend(rows)
        lab.extend([c] * len(rows))
    return torch.tensor(idx, dtype=torch.long).reshape(-1, 4), torch.tensor(lab, dtype=torch.long)


@pytest.mark.parametrize("max_per_class", [32, 2])
def test_G8_L1_anchors_per_channel(max_per_class):
    from tomocpt.training.discriminative import sample_anchors

    t = _target()
    B, C = t.shape[:2]
    avail = torch.ones(B, C, dtype=torch.bool)
    avail[0, 1] = False
    d = torch.full((B, C), 6.0)
    fil = [False, False, True, False, False]
    got = sample_anchors(t, avail, d, fil, 0.5, max_per_class, torch.Generator().manual_seed(1))
    ref = _reference_anchors(t, avail, d, fil, 0.5, max_per_class, torch.Generator().manual_seed(1))
    assert len(got[0]) > 0
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_G8_L1_focal_positives_per_channel():
    from tomocpt.training import discriminative as D

    t = _target(seed=3)
    B, C = t.shape[:2]
    avail = torch.ones(B, C, dtype=torch.bool)
    avail[2, 0] = False
    d = torch.full((B, C), 6.0)
    fil = [False, True, False, False, False]
    pos, n = D.focal_positives(t, avail, d, fil, 0.5)

    peaks = D._peak_mask(t, 0.5)
    ref_pos = torch.zeros_like(peaks)
    ref_n = torch.zeros(B, C)
    for b in range(B):
        for c in range(C):
            if not avail[b, c]:
                continue
            inst = D._instances(t, peaks[:, c], b, c, 6.0, fil[c])
            ref_n[b, c] = len(inst)
            if fil[c]:
                ref_pos[b, c] = peaks[b, c]
            elif len(inst):
                ii = torch.as_tensor(inst)
                ref_pos[b, c, ii[:, 0], ii[:, 1], ii[:, 2]] = True
    assert torch.equal(pos, ref_pos) and torch.equal(n, ref_n)
    assert pos.any()


def test_G8_L2_auto_rule(monkeypatch):
    from tomocpt.training import multiclass_loss as M

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev: types.SimpleNamespace(total_memory=48 * 2 ** 30))
    v = 64 ** 3
    assert not M.auto_loss_checkpoint(16, 6, v, "cuda")      # CZII: unchanged fast path
    assert not M.auto_loss_checkpoint(8, 134, v, "cuda")     # measured 25 GB peak without it
    assert M.auto_loss_checkpoint(16, 134, v, "cuda")        # out of memory without it
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda dev: types.SimpleNamespace(total_memory=24 * 2 ** 30))
    assert M.auto_loss_checkpoint(8, 134, v, "cuda")         # 24 GB GPU
    assert not M.auto_loss_checkpoint(16, 134, v, "cpu")
    assert not M.auto_loss_checkpoint(16, 134, v, None)


def test_G8_L2_checkpoint_is_exact():
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    torch.manual_seed(0)
    B, C, n = 2, 4, 16
    logits = torch.randn(B, C, n, n, n)
    y = torch.rand(B, C, n, n, n) * (torch.rand(B, C, n, n, n) > 0.8)
    avail = torch.ones(B, C, dtype=torch.bool)
    dv = torch.full((B, C), 12.0)
    out = []
    for ck in (False, True):
        x = logits.clone().requires_grad_(True)
        loss = multiclass_picking_loss(x, y, avail, dv, checkpoint=ck, hessian_block=2)
        loss.backward()
        out.append((loss.detach(), x.grad.clone()))
    assert torch.equal(out[0][0], out[1][0])
    assert torch.allclose(out[0][1], out[1][1], rtol=0, atol=1e-7)
