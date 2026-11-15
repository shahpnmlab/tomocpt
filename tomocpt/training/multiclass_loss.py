"""Multi-class picking loss for pipeline v2 (CLAUDE.md Section 5.4).

Section 1.5 applied per class, with weights computed per sample and per class, the Hessian
term normalised by ``(class_d_vox / 20)^2`` and pairs ``(b, c)`` that are not available
masked out. With one class, every pair available and ``class_d_vox = 20`` it equals
``BasePickingModel.loss``.
"""
from typing import Optional, Sequence, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

HESSIAN_REFERENCE_D_VOX = 20.0  # diameter (voxels) at which the Hessian scale is 1
HESSIAN_WEIGHT = 10.0
#: Gate 8b: memory the Hessian term may use per call, and its measured cost per voxel of one
#: (sample, class) pair (fp32, value term included: ~34 MB per 64^3 pair).
HESSIAN_BLOCK_BYTES = 4 * 2 ** 30
HESSIAN_BYTES_PER_VOXEL = 136


def auto_hessian_block(n_samples: int, n_voxels: int, budget_bytes: int = HESSIAN_BLOCK_BYTES) -> int:
    """Classes per Hessian call so that one call stays within ``budget_bytes`` (at least 1).
    B = 16 at 64^3: up to 7 classes in one call."""
    per_class = max(1, int(n_samples) * int(n_voxels) * HESSIAN_BYTES_PER_VOXEL)
    return max(1, int(budget_bytes) // per_class)


#: Activations the Hessian term keeps for backward per voxel of one (sample, class) pair without
#: checkpointing (8b.2, C = 134, B = 4: 12.3 - 6.3 GB over 536 pairs of 64^3).
HESSIAN_SAVED_BYTES_PER_VOXEL = 48
#: Fraction of the GPU memory those activations may take before the loss is checkpointed
#: automatically (TODO 5.1: CZII B = 16 1.2 GB -> off; C = 134: B = 8 13.5 GB -> off on 48 GB
#: (measured peak 25 GB), B = 16 27 GB -> on (out of memory without it)).
HESSIAN_SAVED_FRACTION = 0.4


def auto_loss_checkpoint(n_samples: int, n_classes: int, n_voxels: int, device=None,
                         fraction: float = HESSIAN_SAVED_FRACTION) -> bool:
    """True when the Hessian term's saved activations of all classes would exceed ``fraction`` of
    the GPU memory (checkpointing gives the same loss and gradients, at a recompute cost).
    Off on the CPU."""
    dev = torch.device(device) if device is not None else None
    if dev is None or dev.type != "cuda" or not torch.cuda.is_available():
        return False
    total = torch.cuda.get_device_properties(dev).total_memory
    saved = int(n_samples) * int(n_classes) * int(n_voxels) * HESSIAN_SAVED_BYTES_PER_VOXEL
    return saved > float(fraction) * total
CLASS_HEADS = ("sigmoid", "softmax")
SOFTMAX_SCOPES = ("all", "particles")


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


def class_probabilities(logits: torch.Tensor, class_head: str, available=None,
                        softmax_channels=None) -> torch.Tensor:
    """Per-class probabilities ``[B, C, D, H, W]`` (float32) from logits (Gate 3d).

    ``sigmoid``: independent sigmoids. ``softmax``: softmax over an implicit background logit
    fixed at 0 and the available channels, ``p_c = exp(l_c) / (1 + sum_d exp(l_d))``;
    unavailable channels are 0. With one class both heads are equal.
    ``softmax_channels`` (Gate 3e): ``[C]`` bool, the channels inside the softmax (``None`` = all);
    the other available channels are independent sigmoids."""
    if class_head == "sigmoid":
        return torch.sigmoid(logits.float())
    if class_head != "softmax":
        raise ValueError(f"class_head must be one of {CLASS_HEADS}, got {class_head!r}")
    x = logits.float()
    B, C = x.shape[:2]
    shape = (B, C, *([1] * (x.ndim - 2)))
    avail = _channel_mask(available, B, C, x.device)
    mask = avail
    if softmax_channels is not None:
        in_softmax = torch.as_tensor(softmax_channels, dtype=torch.bool, device=x.device).view(1, C)
        mask = avail & in_softmax
    mask = mask.view(shape)
    xs = torch.where(mask, x, torch.full_like(x, float("-inf")))
    p = torch.softmax(torch.cat([torch.zeros_like(xs[:, :1]), xs], dim=1), dim=1)[:, 1:]
    p = torch.where(mask, p, torch.zeros_like(p))
    if softmax_channels is None:
        return p
    independent = (avail & ~in_softmax).view(shape)
    return torch.where(independent, torch.sigmoid(x), p)


def renormalise_targets(y_true: torch.Tensor, softmax_channels=None) -> torch.Tensor:
    """``y / max(1, sum_c y_c)`` per voxel: softmax probabilities cannot sum above 1.
    ``softmax_channels`` (Gate 3e): ``[C]`` bool; the sum and the division cover these channels
    only, the other targets are unchanged (``None`` = all)."""
    if softmax_channels is None:
        return y_true / y_true.sum(dim=1, keepdim=True).clamp(min=1)
    C = y_true.shape[1]
    m = torch.as_tensor(softmax_channels, dtype=torch.bool, device=y_true.device).view(1, C, *([1] * (y_true.ndim - 2)))
    s = (y_true * m).sum(dim=1, keepdim=True).clamp(min=1)
    return torch.where(m, y_true / s, y_true)


#: Non-zero entries (channel, z, y, x, value) of Kornia's ``get_diff_kernel3d_2nd_order`` (xx, yy, zz,
#: xy, xz, yz), Kornia 0.7.4.
_DIFF2_TAPS = (
    (0, 1, 1, 0, 1.0), (0, 1, 1, 1, -2.0), (0, 1, 1, 2, 1.0),
    (1, 1, 0, 1, 1.0), (1, 1, 1, 1, -2.0), (1, 1, 2, 1, 1.0),
    (2, 0, 1, 1, 1.0), (2, 1, 1, 1, -2.0), (2, 2, 1, 1, 1.0),
    (3, 1, 0, 0, 1.0), (3, 1, 0, 2, -1.0), (3, 1, 2, 0, -1.0), (3, 1, 2, 2, 1.0),
    (4, 0, 0, 1, 1.0), (4, 0, 2, 1, -1.0), (4, 2, 0, 1, -1.0), (4, 2, 2, 1, 1.0),
    (5, 0, 1, 0, 1.0), (5, 0, 1, 2, -1.0), (5, 2, 1, 0, -1.0), (5, 2, 1, 2, 1.0),
)
_KERNEL_CACHE = {}


def _diff2_kernel(device, dtype) -> torch.Tensor:
    """Kornia's 2nd-order diff kernel ``[6, 1, 3, 3, 3]``, built on ``device`` by scalar fills (no
    host-to-device copy, so no host sync; Gate 8b) and cached per device and dtype."""
    key = (str(device), dtype)
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = torch.zeros((6, 1, 3, 3, 3), device=device, dtype=dtype)
        for c, z, y, x, v in _DIFF2_TAPS:
            k[c, 0, z, y, x].fill_(v)
        _KERNEL_CACHE[key] = k
    return k


def _hessian3d(v: torch.Tensor) -> torch.Tensor:
    """``kornia.filters.spatial_gradient3d(v, mode="diff", order=2)``, same operations (replicate pad,
    flipped grouped conv3d) with a cached kernel. ``[B, C, D, H, W] -> [B, C, 6, D, H, W]``."""
    b, c, d, h, w = v.shape
    kernel = _diff2_kernel(v.device, v.dtype)
    kernel_flip = kernel.repeat(c, 1, 1, 1, 1).flip(-3)
    x = nn.functional.pad(v, [1] * 6, "replicate")
    return nn.functional.conv3d(x, kernel_flip, padding=0, groups=c).view(b, c, 6, d, h, w)


def _hessian_term(y_pred_c: torch.Tensor, y_true_c: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mean over voxels of ``sum_6 huber(s H(y_pred), s H(y_true))`` per class.

    ``y_pred_c``, ``y_true_c``: ``[B, k, D, H, W]`` (a block of ``k`` classes); ``scale``: ``[B, k]``
    (or ``[B]`` for ``k = 1``). Returns ``[B, k]``. The target Hessian needs no graph."""
    B, k = y_pred_c.shape[:2]
    h_pred = _hessian3d(y_pred_c)  # [B,k,6,D,H,W]
    with torch.no_grad():
        h_true = _hessian3d(y_true_c)
    s = scale.reshape(B, k, 1, 1, 1, 1)
    h = nn.functional.huber_loss(s * h_pred, s * h_true, reduction="none").sum(2)
    return h.flatten(2).mean(2)


def _other_class_target(y_true: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """``o[b, c] = max over available classes d != c of y_true[b, d]`` (0 if there is none)."""
    B, C = available.shape
    masked = torch.where(available.view(B, C, 1, 1, 1), y_true, torch.zeros_like(y_true))
    top2, idx = masked.topk(2, dim=1)  # [B, 2, D, H, W]
    channel = torch.arange(C, device=y_true.device).view(1, C, 1, 1, 1)
    # the largest target of the other classes: the top value unless channel c holds it
    return torch.where(idx[:, :1] == channel, top2[:, 1:2], top2[:, :1])


def _value_weights(y_true: torch.Tensor, available: torch.Tensor, cross_class_neg_weight: float) -> torch.Tensor:
    """Section 1.5 weights per sample and class (plus the Gate 3b cross-class term):
    ``w_pos = N/(S+1)`` where ``y > 0``, else ``2 N/(N-S+1)`` (+ ``k w_pos o``)."""
    B, C = y_true.shape[:2]
    n_vox = y_true[0, 0].numel()
    s = y_true.flatten(2).sum(2)  # [B, C], sum of soft labels
    w_pos = (n_vox / (s + 1)).view(B, C, 1, 1, 1)
    w_neg = (n_vox / (n_vox - s + 1)).view(B, C, 1, 1, 1)
    neg = 2 * w_neg
    if cross_class_neg_weight and C > 1:
        neg = neg + cross_class_neg_weight * w_pos * _other_class_target(y_true, available)
    return torch.where(y_true > 0, w_pos, neg)


def _core_weighted(w: torch.Tensor, y_true: torch.Tensor, core_weight: float, core_power: float,
                   core_channels: Optional[Sequence[bool]]) -> torch.Tensor:
    """``w * (1 + core_weight * y^core_power)`` on ``core_channels`` (Gate 4f); ``w`` itself when off."""
    if not core_weight:
        return w
    C = w.shape[1]
    factor = 1.0 + float(core_weight) * y_true.clamp(min=0) ** float(core_power)
    if core_channels is not None:
        sel = [bool(v) for v in core_channels]
        if len(sel) != C:
            raise ValueError(f"core_channels has {len(sel)} entries for {C} channels")
        factor = torch.where(torch.tensor(sel, device=w.device).view(1, C, 1, 1, 1), factor,
                             torch.ones_like(factor))
    return w * factor


def multiclass_picking_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    class_available: torch.Tensor,
    class_d_vox: torch.Tensor,
    class_weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    checkpoint: bool = True,
    cross_class_neg_weight: float = 0.0,
    value_loss: str = "huber",
    focal_channels: Optional[Sequence[bool]] = None,
    focal_gamma: float = 2.0,
    core_weight: float = 0.0,
    core_power: float = 4.0,
    core_channels: Optional[Sequence[bool]] = None,
    hessian_block: int = 1,
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
        value_loss: ``"huber"`` (the weighted Huber term above) or ``"focal"`` (Gate 7, human
            decision 1 Nov 2026): the same weights ``w`` and voxel mean, with the per-voxel
            penalty ``discriminative.soft_focal_penalty`` (``|y - p|^focal_gamma * BCE(p, y)``,
            minimiser ``p = y``) on the channels where ``focal_channels`` is True (default all)
            and the Huber penalty on the others (filament channels). The Hessian term is the same.

        core_weight, core_power, core_channels: centroid weight (Gate 4f): on the channels where
            ``core_channels`` is True (default all; the model passes its particle channels) the value
            weight ``w`` becomes ``w * (1 + core_weight * y^core_power)``. 0 = off (bit-identical).
        hessian_block: classes per Hessian call (Gate 8b): 1 = one class at a time (default), ``k``
            = blocks of ``k`` channels, 0 = all channels in one call. Same maths (the Hessian is
            per channel); peak memory grows with the block. Checkpointing, if on, wraps each block.

    Returns the mean over available ``(b, c)`` pairs (0 if none is available). Every output
    channel is part of the graph, so DDP sees a gradient (possibly zero) for every parameter.
    """
    B, C = y_pred.shape[:2]
    y_true = y_true.to(y_pred.dtype)
    available = class_available.to(device=y_pred.device, dtype=torch.bool).view(B, C)

    if value_loss == "huber":
        w = _core_weighted(_value_weights(y_true, available, cross_class_neg_weight), y_true, core_weight,
                           core_power, core_channels)
        value_term = (w * nn.functional.huber_loss(y_pred, y_true, reduction="none")).flatten(2).mean(2)
    elif value_loss == "focal":
        from tomocpt.training.discriminative import soft_focal_penalty

        w = _core_weighted(_value_weights(y_true, available, cross_class_neg_weight), y_true, core_weight,
                           core_power, core_channels)
        sel = [True] * C if focal_channels is None else [bool(v) for v in focal_channels]
        if len(sel) != C:
            raise ValueError(f"focal_channels has {len(sel)} entries for {C} channels")
        sel_t = torch.tensor(sel, device=y_pred.device).view(1, C, 1, 1, 1)
        penalty = torch.where(sel_t, soft_focal_penalty(y_pred, y_true, focal_gamma).to(y_pred.dtype),
                              nn.functional.huber_loss(y_pred, y_true, reduction="none"))
        value_term = (w * penalty).flatten(2).mean(2)
    else:
        raise ValueError(f"value_loss must be 'huber' or 'focal', got {value_loss!r}")

    scale = (class_d_vox.to(device=y_pred.device, dtype=y_pred.dtype).view(B, C) / HESSIAN_REFERENCE_D_VOX) ** 2
    use_checkpoint = checkpoint and torch.is_grad_enabled()
    block = C if int(hessian_block) <= 0 else min(int(hessian_block), C)
    hessian_terms = []
    for c0 in range(0, C, block):
        c1 = min(c0 + block, C)
        args = (y_pred[:, c0:c1], y_true[:, c0:c1], scale[:, c0:c1])
        if use_checkpoint:
            hessian_terms.append(torch_checkpoint(_hessian_term, *args, use_reentrant=False))
        else:
            hessian_terms.append(_hessian_term(*args))
    per_pair = value_term + HESSIAN_WEIGHT * torch.cat(hessian_terms, dim=1)  # [B, C]

    if class_weights is not None:
        cw = torch.as_tensor(class_weights, device=per_pair.device, dtype=per_pair.dtype).view(1, C)
        per_pair = per_pair * cw

    # torch.where (not multiplication) so that a NaN/inf in a masked pair cannot leak
    per_pair = torch.where(available, per_pair, torch.zeros_like(per_pair))
    loss = per_pair.sum() / available.sum().clamp(min=1)

    # DDP: a class unavailable in the whole batch still gets a (zero) gradient path. Always added
    # (an exact zero), so there is no host sync on ``unavailable`` (Gate 8b).
    unavailable = ~available.any(dim=0)
    per_channel = y_pred.sum(dim=(0, *range(2, y_pred.ndim)))  # [C]
    return loss + 0 * torch.where(unavailable, per_channel, torch.zeros_like(per_channel)).sum()
