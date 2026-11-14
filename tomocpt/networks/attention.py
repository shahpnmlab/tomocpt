"""Fused window attention for MONAI's Swin blocks (Gate 8b.4).

``SDPAWindowAttention`` is MONAI 1.4.0's ``WindowAttention`` with the same parameters and buffers
(state dicts and checkpoints are interchangeable) and a ``forward`` that calls
``torch.nn.functional.scaled_dot_product_attention`` with the relative position bias (+ shift mask)
as an additive mask, on CUDA. On the CPU it runs MONAI's eager attention (bit-identical to the
legacy goldens; training and predict speed come from the GPU).

With ``feature_size=36`` the head dimension is 12 at every stage; torch 2.5.1's flash and
memory-efficient kernels refuse it, so on CUDA q, k, v are zero-padded to a multiple of 8 (zero
columns add nothing to q·k; the padded output columns are dropped). Measured on an A6000 (bf16, B=16
Swin stages 1-2, forward + backward): ~2.1x faster and 2.7x less memory than the eager attention.
"""
import torch
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import WindowAttention
from torch import nn

#: Head dimension multiple required by the fused CUDA kernels.
_HEAD_DIM_MULTIPLE = 8


class SDPAWindowAttention(WindowAttention):
    def forward(self, x, mask):
        if not x.is_cuda:
            # CPU: MONAI's eager attention, bit-identical to the legacy goldens (no speed to gain)
            return super().forward(x, mask)
        b, n, c = x.shape
        h = self.num_heads
        d = c // h
        q, k, v = self.qkv(x).reshape(b, n, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_position_bias_table[self.relative_position_index[:n, :n].reshape(-1)]
        # contiguous: the fused kernels need a unit-stride last dimension in the mask
        bias = bias.reshape(n, n, h).permute(2, 0, 1).contiguous()  # [h, n, n]
        # The small bias is cast and then broadcast over the batch (stride 0), never copied to
        # [b, h, n, n]. Shifted windows: the window index is folded into the heads
        # ([b/nw, nw*h, n, d]) so that the [nw, h, n, n] bias + mask broadcasts the same way.
        if mask is not None:
            nw = mask.shape[0]
            bias = (bias.unsqueeze(0) + mask.unsqueeze(1)).to(q.dtype).reshape(1, nw * h, n, n)
            q, k, v = (t.reshape(b // nw, nw * h, n, d) for t in (q, k, v))
        else:
            bias = bias.to(q.dtype).unsqueeze(0)  # [1, h, n, n]
        bias = bias.expand(q.shape[0], -1, -1, -1)
        pad = (-d) % _HEAD_DIM_MULTIPLE
        if pad:
            q, k, v = (F.pad(t, (0, pad)) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale)
        if pad:
            out = out[..., :d]
        x = out.reshape(b, h, n, d).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


def swap_window_attention(module: nn.Module) -> int:
    """Turn every MONAI ``WindowAttention`` in ``module`` into :class:`SDPAWindowAttention` in place
    (same parameters, buffers and names). Returns the number of modules swapped."""
    n = 0
    for m in module.modules():
        if type(m) is WindowAttention:
            m.__class__ = SDPAWindowAttention
            n += 1
    return n
