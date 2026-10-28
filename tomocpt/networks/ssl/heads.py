"""SSL v2 heads (Gate 5). They live on the ``SelfSupervisedModel``, not on the network, so a
transfer to the picker drops exactly these modules."""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["ProjectionHead", "DenseProjectionHead", "RelativeScaleHead"]


class ProjectionHead(nn.Module):
    """Global contrastive projection: 2-layer MLP on a pooled feature ``[B, in_dim]``."""

    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int | None = None):
        super().__init__()
        hidden = hidden or in_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


class DenseProjectionHead(nn.Module):
    """Dense contrastive projection: one 1x1x1 convolution on a feature map
    ``[B, in_channels, d, h, w]`` -> ``[B, out_dim, d, h, w]`` (a per-voxel linear map)."""

    def __init__(self, in_channels: int, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_dim, kernel_size=1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.proj(f)


class RelativeScaleHead(nn.Module):
    """Predicts ``log2(s_a / s_b)`` from pooled encoder features of the two views:
    ``r = (g([p_a, p_b]) - g([p_b, p_a])) / 2``, antisymmetric by construction
    (``r(a, b) = -r(b, a)``, ``r(a, a) = 0``); ``g`` is a 2-layer MLP. Returns ``[B]``."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(2 * in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        n = p_a.shape[0]
        both = self.g(torch.cat([torch.cat([p_a, p_b], dim=1), torch.cat([p_b, p_a], dim=1)], dim=0))
        return ((both[:n] - both[n:]) / 2).reshape(n)
