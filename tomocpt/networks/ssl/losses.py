"""SSL v2 losses (Gate 5)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["masked_l1", "nt_xent", "dense_info_nce"]


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, visible_weight: float) -> torch.Tensor:
    """``L1(masked) + visible_weight * L1(visible)``: the mean absolute error over the masked
    voxels plus ``visible_weight`` times the mean over the visible voxels. ``mask`` is bool,
    True = masked (hidden from the network). A part without voxels contributes 0."""
    mask = mask.to(torch.bool).expand_as(pred)
    err = (pred.float() - target.float()).abs()
    m = mask.float()
    n_masked = m.sum()
    n_visible = m.numel() - n_masked
    masked = (err * m).sum() / n_masked.clamp_min(1.0)
    visible = (err * (1.0 - m)).sum() / n_visible.clamp_min(1.0)
    return masked + float(visible_weight) * visible


def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float, l1_weight: float) -> torch.Tensor:
    """NT-Xent between paired embeddings ``z_a[i] <-> z_b[i]`` (``[B, D]``; the other ``2B - 2``
    embeddings are negatives), plus ``l1_weight * mean(|z|)`` (a mean, so the weight does not
    grow with batch size or width)."""
    with torch.autocast(device_type=z_a.device.type, enabled=False):
        z = torch.cat([z_a, z_b], dim=0).float()
        zn = F.normalize(z, dim=1)
        sim = zn @ zn.t() / float(temperature)
        n = z_a.shape[0]
        eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float("-inf"))
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, labels) + float(l1_weight) * z.abs().mean()


def _sample(f: torch.Tensor, pts: torch.Tensor, view_size: int) -> torch.Tensor:
    """Features of ``f`` (``[B, D, d, h, w]``, covering the view) at view voxel coordinates
    ``pts`` (``[B, n, 3]``, ``(z, y, x)``, voxel ``i`` at ``i``). Returns ``[B, n, D]``."""
    grid = (pts.float() + 0.5) / float(view_size) * 2.0 - 1.0
    grid = grid.flip(-1).reshape(pts.shape[0], 1, 1, pts.shape[1], 3)  # (x, y, z) for grid_sample
    out = F.grid_sample(f.float(), grid.to(f.device), mode="bilinear", padding_mode="border", align_corners=False)
    return out.reshape(f.shape[0], f.shape[1], pts.shape[1]).transpose(1, 2)


def dense_info_nce(f_a: torch.Tensor, f_b: torch.Tensor, pts_a: torch.Tensor, pts_b: torch.Tensor,
                   temperature: float, view_size: int = 64) -> torch.Tensor:
    """Dense InfoNCE: the embedding of anchor ``k`` in view A has its positive at the same
    anchor in view B; every other anchor of the batch is a negative. Symmetric (A->B and B->A).

    ``f_*``: ``[B, D, d, h, w]`` feature maps covering the view's field of view;
    ``pts_*``: ``[B, n, 3]`` view voxel coordinates of a ``view_size``^3 view. Computed in
    float32 (safe under bf16 autocast)."""
    with torch.autocast(device_type=f_a.device.type, enabled=False):
        ea = F.normalize(_sample(f_a, pts_a, view_size).reshape(-1, f_a.shape[1]), dim=1)
        eb = F.normalize(_sample(f_b, pts_b, view_size).reshape(-1, f_b.shape[1]), dim=1)
        logits = ea @ eb.t() / float(temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
