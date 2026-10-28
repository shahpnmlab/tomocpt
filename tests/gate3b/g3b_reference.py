"""Independent reference of the Gate 3b loss, written from ``03b_AGENT_CROSS_CLASS_NEG.md``
(Section 5.4 per available pair, with the cross-class negative weight). No tests here."""

import kornia
import torch
from torch import nn


def other_class_target(y_true, available, b, c):
    """``o[b, c]``: voxel-wise max of ``y_true[b, d]`` over available ``d != c`` (0 if none)."""
    o = torch.zeros_like(y_true[b, c])
    for d in range(y_true.shape[1]):
        if d != c and bool(available[b, d]):
            o = torch.maximum(o, y_true[b, d])
    return o


def reference_loss(y_pred, y_true, available, d_vox, k=0.0, class_weights=None):
    B, C = available.shape
    N = y_true[0, 0].numel()
    total, n = 0.0, 0
    for b in range(B):
        for c in range(C):
            if not bool(available[b, c]):
                continue
            yp, yt = y_pred[b:b + 1, c:c + 1], y_true[b:b + 1, c:c + 1]
            S = yt.sum()
            w_pos, w_neg = N / (S + 1), N / (N - S + 1)
            o = other_class_target(y_true, available, b, c)[None, None]
            w = torch.where(yt > 0, w_pos, 2 * w_neg + k * w_pos * o)
            s = (d_vox[b, c] / 20.0) ** 2
            hp = s * kornia.filters.spatial_gradient3d(yp, mode="diff", order=2)
            ht = s * kornia.filters.spatial_gradient3d(yt, mode="diff", order=2)
            term = (w * nn.functional.huber_loss(yp, yt, reduction="none")).mean() \
                + 10 * nn.functional.huber_loss(hp, ht, reduction="none").sum(2).mean()
            wc = 1.0 if class_weights is None else float(class_weights[c])
            total = total + wc * term
            n += 1
    return total / n
