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
CLASS_HEADS = ("sigmoid", "softmax")


def _channel_mask(available, B: int, C: int, device) -> torch.Tensor:
    """``[B, C]`` bool from a ``[B, C]`` bool tensor, a sequence of channel indices, or ``None``."""
    if available is None:
        return torch.ones((B, C), dtype=torch.bool, device=device)
    if torch.is_tensor(available) and available.dtype == torch.bool:
        return available.to(device).view(B, C)
    idx = torch.as_tensor(list(available) if not torch.is_tensor(available) else available,
                          dtype=torch.long, device=device).reshape(-1)
    mask = torch.zeros(C, dtype=torch.bool, device=device)
    mask[idx] = True
    return mask.view(1, C).expand(B, C)


def class_probabilities(logits: torch.Tensor, class_head: str, available=None) -> torch.Tensor:
    """Per-class probabilities ``[B, C, D, H, W]`` (float32) from logits (Gate 3d).

    ``sigmoid``: independent sigmoids. ``softmax``: softmax over an implicit background logit
    fixed at 0 and the available channels, ``p_c = exp(l_c) / (1 + sum_d exp(l_d))``;
    unavailable channels are 0. With one class both heads are equal."""
    if class_head == "sigmoid":
        return torch.sigmoid(logits.float())
    if class_head != "softmax":
        raise ValueError(f"class_head must be one of {CLASS_HEADS}, got {class_head!r}")
    x = logits.float()
    B, C = x.shape[:2]
    mask = _channel_mask(available, B, C, x.device).view(B, C, *([1] * (x.ndim - 2)))
    x = torch.where(mask, x, torch.full_like(x, float("-inf")))
    p = torch.softmax(torch.cat([torch.zeros_like(x[:, :1]), x], dim=1), dim=1)[:, 1:]
    return torch.where(mask, p, torch.zeros_like(p))


def renormalise_targets(y_true: torch.Tensor) -> torch.Tensor:
    """``y / max(1, sum_c y_c)`` per voxel: softmax probabilities cannot sum above 1."""
    return y_true / y_true.sum(dim=1, keepdim=True).clamp(min=1)


def _hessian_term(y_pred_c: torch.Tensor, y_true_c: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mean over voxels of ``sum_6 huber(s H(y_pred), s H(y_true))`` for one class.

    ``y_pred_c``, ``y_true_c``: ``[B, 1, D, H, W]``; ``scale``: ``[B]``. Returns ``[B]``."""
    h_pred = kornia.filters.spatial_gradient3d(y_pred_c, mode="diff", order=2)  # [B,1,6,D,H,W]
    h_true = kornia.filters.spatial_gradient3d(y_true_c, mode="diff", order=2)
    s = scale.view(-1, 1, 1, 1, 1, 1)
    h = nn.functional.huber_loss(s * h_pred, s * h_true, reduction="none").sum(2)
    return h.flatten(1).mean(1)


def _other_class_target(y_true: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """``o[b, c] = max over available classes d != c of y_true[b, d]`` (0 if there is none)."""
    B, C = available.shape
    masked = torch.where(available.view(B, C, 1, 1, 1), y_true, torch.zeros_like(y_true))
    top2, idx = masked.topk(2, dim=1)  # [B, 2, D, H, W]
    channel = torch.arange(C, device=y_true.device).view(1, C, 1, 1, 1)
    # the largest target of the other classes: the top value unless channel c holds it
    return torch.where(idx[:, :1] == channel, top2[:, 1:2], top2[:, :1])


def multiclass_picking_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    class_available: torch.Tensor,
    class_d_vox: torch.Tensor,
    class_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    checkpoint: bool = True,
    cross_class_neg_weight: float = 0.0,
) -> torch.Tensor:
    """Pipeline-v2 picking loss.

    Args:
        y_pred: sigmoid output ``[B, C, D, H, W]``.
        y_true: soft targets ``[B, C, D, H, W]`` in [0, 1].
        class_available: ``[B, C]`` bool; unavailable pairs contribute exactly 0.
        class_d_vox: ``[B, C]`` class diameter in working voxels (Hessian scale).
        class_weights: optional, one weight per channel; multiplies each class's term.
        checkpoint: run the Hessian term one class at a time under gradient checkpointing.
        cross_class_neg_weight: ``k`` of Gate 3b. A voxel outside class c but inside another
            available class d gets the negative weight ``2·w_neg + k·w_pos·o`` in channel c,
            where ``o`` is the largest soft target of the other available classes. 0 = off.

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
    neg = 2 * w_neg
    if cross_class_neg_weight and C > 1:
        neg = neg + cross_class_neg_weight * w_pos * _other_class_target(y_true, available)
    w = torch.where(y_true > 0, w_pos, neg)
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
