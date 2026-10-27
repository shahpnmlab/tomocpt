"""Multi-class picking loss for pipeline v2 (CLAUDE.md Section 5.4).

Section 1.5 applied per class, with weights computed per sample and per class, the Hessian
term normalised by ``(class_d_vox / 20)^2`` and pairs ``(b, c)`` that are not available
masked out. With one class, every pair available and ``class_d_vox = 20`` it equals
``BasePickingModel.loss``.
"""
from typing import Optional, Sequence, Union

import kornia
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

HESSIAN_REFERENCE_D_VOX = 20.0  # diameter (voxels) at which the Hessian scale is 1
HESSIAN_WEIGHT = 10.0


def _hessian_term(y_pred_c: torch.Tensor, y_true_c: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mean over voxels of ``sum_6 huber(s H(y_pred), s H(y_true))`` for one class.

    ``y_pred_c``, ``y_true_c``: ``[B, 1, D, H, W]``; ``scale``: ``[B]``. Returns ``[B]``."""
    h_pred = kornia.filters.spatial_gradient3d(y_pred_c, mode="diff", order=2)  # [B,1,6,D,H,W]
    h_true = kornia.filters.spatial_gradient3d(y_true_c, mode="diff", order=2)
    s = scale.view(-1, 1, 1, 1, 1, 1)
    h = nn.functional.huber_loss(s * h_pred, s * h_true, reduction="none").sum(2)
    return h.flatten(1).mean(1)


def multiclass_picking_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    class_available: torch.Tensor,
    class_d_vox: torch.Tensor,
    class_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    checkpoint: bool = True,
) -> torch.Tensor:
    """Pipeline-v2 picking loss.

    Args:
        y_pred: sigmoid output ``[B, C, D, H, W]``.
        y_true: soft targets ``[B, C, D, H, W]`` in [0, 1].
        class_available: ``[B, C]`` bool; unavailable pairs contribute exactly 0.
        class_d_vox: ``[B, C]`` class diameter in working voxels (Hessian scale).
        class_weights: optional, one weight per channel; multiplies each class's term.
        checkpoint: run the Hessian term one class at a time under gradient checkpointing.

    Returns the mean over available ``(b, c)`` pairs (0 if none is available). Every output
    channel is part of the graph, so DDP sees a gradient (possibly zero) for every parameter.
    """
    B, C = y_pred.shape[:2]
    y_true = y_true.to(y_pred.dtype)
    available = class_available.to(device=y_pred.device, dtype=torch.bool).view(B, C)

    # Section 1.5 weights, per sample and per class
    n_vox = y_true[0, 0].numel()
    s = y_true.flatten(2).sum(2)  # [B, C], sum of soft labels
    w_pos = (n_vox / (s + 1)).view(B, C, 1, 1, 1)
    w_neg = (n_vox / (n_vox - s + 1)).view(B, C, 1, 1, 1)
    w = torch.where(y_true > 0, w_pos, 2 * w_neg)
    value_term = (w * nn.functional.huber_loss(y_pred, y_true, reduction="none")).flatten(2).mean(2)

    scale = (class_d_vox.to(device=y_pred.device, dtype=y_pred.dtype).view(B, C) / HESSIAN_REFERENCE_D_VOX) ** 2
    use_checkpoint = checkpoint and torch.is_grad_enabled()
    hessian_terms = []
    for c in range(C):
        args = (y_pred[:, c:c + 1], y_true[:, c:c + 1], scale[:, c])
        if use_checkpoint:
            hessian_terms.append(torch_checkpoint(_hessian_term, *args, use_reentrant=False))
        else:
            hessian_terms.append(_hessian_term(*args))
    per_pair = value_term + HESSIAN_WEIGHT * torch.stack(hessian_terms, dim=1)  # [B, C]

    if class_weights is not None:
        cw = torch.as_tensor(class_weights, device=per_pair.device, dtype=per_pair.dtype).view(1, C)
        per_pair = per_pair * cw

    # torch.where (not multiplication) so that a NaN/inf in a masked pair cannot leak
    per_pair = torch.where(available, per_pair, torch.zeros_like(per_pair))
    loss = per_pair.sum() / available.sum().clamp(min=1)

    # DDP: a class unavailable in the whole batch still gets a (zero) gradient path
    unavailable = ~available.any(dim=0)
    if bool(unavailable.any()):
        loss = loss + 0 * y_pred[:, unavailable].sum()
    return loss
