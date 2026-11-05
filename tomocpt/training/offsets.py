"""Offset-head targets (Gate 4f, TODO 3b.3b c).

The target of the offset head at voxel ``v`` is the vector from ``v`` to the centre of the particle
blob around it. It is computed from the (augmented) dense target ``y``: the ``y``-weighted
centroid in a box around ``v`` that covers the whole blob (the target is 0 beyond the class radius
R), so flips and rotations of the augmentation need no extra handling. Only core voxels
(``y > 0.5``) get a target.
"""
import math
from typing import Optional, Sequence, Tuple

import torch

#: Core voxels: target above this value.
CORE_THRESHOLD = 0.5
#: Box half-width per class: ceil(BOX_FACTOR * R) + 1 (covers the blob from any core voxel).
BOX_FACTOR = 1.3


def _box_sum(v: torch.Tensor, h: int) -> torch.Tensor:
    """Sum of ``v`` ``[N, D, H, W]`` over the box ``|u| <= h`` per axis (zero outside)."""
    for dim in (1, 2, 3):
        v = torch.movedim(v, dim, -1)
        cs = torch.nn.functional.pad(v, (h + 1, h)).cumsum(-1)
        n = v.shape[-1]
        v = cs[..., 2 * h + 1:2 * h + 1 + n] - cs[..., :n]
        v = torch.movedim(v, -1, dim)
    return v


def offset_targets(y: torch.Tensor, class_d_vox: torch.Tensor, available: torch.Tensor,
                   particle_channels: Optional[Sequence[bool]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(target [B, 3, D, H, W], mask [B, 1, D, H, W] bool)``: offsets ``(dz, dy, dx)`` in voxels
    from each core voxel to the centroid of its blob; where cores of two classes overlap, the
    larger ``y`` wins. ``class_d_vox`` ``[B, C]`` (diameter in voxels), ``available`` ``[B, C]``."""
    with torch.autocast(device_type=y.device.type, enabled=False):
        y = y.detach().float()
        B, C, D, H, W = y.shape
        dev = y.device
        sel = [True] * C if particle_channels is None else [bool(v) for v in particle_channels]
        avail = available.to(device=dev, dtype=torch.bool).view(B, C)
        d_vox = class_d_vox.to(device=dev, dtype=torch.float32).view(B, C)
        grids = torch.meshgrid(*[torch.arange(s, device=dev, dtype=torch.float32) for s in (D, H, W)],
                               indexing="ij")
        target = torch.zeros(B, 3, D, H, W, device=dev)
        best = torch.zeros(B, D, H, W, device=dev)
        pairs = [(b, c) for b in range(B) for c in range(C) if sel[c] and bool(avail[b, c])]
        by_h = {}
        for b, c in pairs:
            h = int(math.ceil(BOX_FACTOR * float(d_vox[b, c]) / 2.0)) + 1
            by_h.setdefault(h, []).append((b, c))
        for h, group in by_h.items():
            bs = [b for b, _ in group]
            Y = torch.stack([y[b, c] for b, c in group])  # [n, D, H, W]
            core = Y > CORE_THRESHOLD
            if not bool(core.any()):
                continue
            s0 = _box_sum(Y, h)
            for i, g in enumerate(grids):
                cen = _box_sum(Y * g, h) / s0.clamp(min=1e-12)
                off = cen - g
                for j, b in enumerate(bs):
                    win = core[j] & (Y[j] > best[b])
                    target[b, i] = torch.where(win, off[j], target[b, i])
            for j, b in enumerate(bs):
                best[b] = torch.where(core[j], torch.maximum(best[b], Y[j]), best[b])
        mask = (best > CORE_THRESHOLD).unsqueeze(1)
    return target, mask


def offset_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean ``|pred - target|`` over masked voxels and the 3 components; 0 (with ``pred`` in the
    graph, for DDP) when the mask is empty."""
    m = mask.to(pred.dtype)
    n = m.sum() * 3
    l1 = ((pred.float() - target) .abs() * m.float()).sum()
    return torch.where(n > 0, l1 / n.clamp(min=1), 0.0 * pred.float().sum())
