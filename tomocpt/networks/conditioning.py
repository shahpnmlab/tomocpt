"""Voxel-size conditioning (Gate 3).

``ScaleEmbedding`` turns the working voxel size (Å per voxel) into an embedding vector and
``FiLM`` modulates a feature map with it: ``f <- f * (1 + gamma) + beta``. FiLM layers start
at zero, so a conditioned network with freshly built FiLM layers computes exactly what the
unconditioned network computes (CLAUDE.md invariant 6).
"""
import torch
from torch import nn

APIX_REFERENCE = 10.0  # t = log(apix / APIX_REFERENCE); t = 0 at 10 Å/px


class ScaleEmbedding(nn.Module):
    """Embed the voxel size: ``t = log(apix / 10)``, Fourier features
    ``[t, sin(2^k t), cos(2^k t)]`` for ``k = 0 .. n_freq - 1``, then
    ``Linear(1 + 2 n_freq -> 64) -> SiLU -> Linear(64 -> dim)``."""

    def __init__(self, dim: int = 64, n_freq: int = 8, hidden: int = 64):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * n_freq, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, apix) -> torch.Tensor:
        """``apix``: float, or tensor of shape ``[B]`` or ``[B, 1]`` (Å per voxel).
        Returns ``[B, dim]`` (``[1, dim]`` for a scalar)."""
        weight = self.mlp[0].weight
        apix = torch.as_tensor(apix, device=weight.device).to(weight.dtype).reshape(-1, 1)
        t = torch.log(apix / APIX_REFERENCE)
        freqs = 2.0 ** torch.arange(self.n_freq, device=t.device, dtype=t.dtype)
        features = torch.cat([t, torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
        return self.mlp(features)


class FiLM(nn.Module):
    """Feature-wise affine modulation, ``f * (1 + gamma) + beta`` with
    ``gamma, beta = Linear(cond_dim -> 2 C)(e)``. Weights and bias start at zero."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.channels = channels
        self.lin = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, f: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """``f``: ``[B, C, *spatial]``; ``e``: ``[B, cond_dim]`` or ``[1, cond_dim]``."""
        gamma, beta = self.lin(e).to(f.dtype).chunk(2, dim=1)
        shape = (gamma.shape[0], self.channels) + (1,) * (f.ndim - 2)
        return f * (1 + gamma.reshape(shape)) + beta.reshape(shape)
