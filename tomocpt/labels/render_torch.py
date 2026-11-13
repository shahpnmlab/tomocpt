"""Torch ports of ``labels.helpers.render_targets`` and ``labels.filaments.render_tubes`` (Gate 8c.2).

Same Section 1.3 profile and ``clip`` policy as the CPU renderers (equal within 1e-5), on the
device of the inputs. A ``batch_index`` renders several samples at once into ``[B, C, *shape]``.
Offsets are formed in float64 on the small per-point arrays; the dense work is float32.
"""
from typing import Optional, Sequence

import torch

_SS = 0.33 / 2.0  # small / large Gaussian widths as fractions of R (Section 1.3)
_SL = 0.66 / 2.0
_POINT_BUDGET = 1 << 24   # window voxels per scatter (particles)
_TUBE_BUDGET = 1 << 23    # window voxels per pass (tubes)


def _profile_unit(u: torch.Tensor) -> torch.Tensor:
    u2 = u * u
    v = 0.5 * torch.exp(-u2 / (2.0 * _SS * _SS)) + 0.5 * torch.exp(-u2 / (2.0 * _SL * _SL))
    return torch.where(u <= 1.0, v, torch.zeros_like(v))


def _prepare(rows: torch.Tensor, width: int, cls_col: int, d_col: int, n_classes: int,
             batch_index: Optional[torch.Tensor]):
    rows = rows.reshape(-1, width).to(torch.float64)
    device = rows.device
    if batch_index is None:
        bi = torch.zeros(len(rows), dtype=torch.int64, device=device)
    else:
        bi = batch_index.reshape(-1).to(device=device, dtype=torch.int64)
    R = rows[:, d_col] / 2.0
    keep = R > 0
    rows, R, bi = rows[keep], R[keep], bi[keep]
    ci = torch.round(rows[:, cls_col]).to(torch.int64)
    if len(ci) and bool(((ci < 0) | (ci >= int(n_classes))).any()):
        raise ValueError(f"class index outside [0, {n_classes})")
    return rows, R, bi, ci


def render_targets_torch(points: torch.Tensor, out_shape_zyx: Sequence[int], n_classes: int,
                         batch_index: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor:
    """Soft particle targets ``[C, *shape]`` (``[B, C, *shape]`` with ``batch_index``) from
    ``points [N, 5]`` = ``z, y, x, class_index, d_vox`` (float positions), ``clip`` policy,
    max-combined per channel."""
    shape = tuple(int(s) for s in out_shape_zyx)
    C = int(n_classes)
    B = int(batch_size) if batch_index is not None else 1
    pts, R, bi, ci = _prepare(points, 5, 3, 4, C, batch_index)
    out = torch.zeros((B, C) + shape, dtype=torch.float32, device=points.device)
    if len(pts):
        order = torch.argsort(R)
        pts, R, bi, ci = pts[order], R[order], bi[order], ci[order]
        widths = (torch.floor(2.0 * R).long() + 2).tolist()
        flat = out.view(-1)
        start = 0
        while start < len(pts):
            end = start + 1
            while end < len(pts) and (end + 1 - start) * widths[end] ** 3 <= _POINT_BUDGET:
                end += 1
            _splat(flat, pts[start:end], R[start:end], bi[start:end], ci[start:end], widths[end - 1], shape, C)
            start = end
    return out if batch_index is not None else out[0]


def _splat(flat, p, R, bi, ci, w, shape, C):
    device = p.device
    lo = torch.floor(p[:, :3] - R[:, None])                                   # [n, 3]
    g = lo[:, :, None] + torch.arange(w, device=device, dtype=torch.float64)  # [n, 3, w] voxel indices
    d = (g - p[:, :3, None]).to(torch.float32)
    r = torch.sqrt(d[:, 0, :, None, None] ** 2 + d[:, 1, None, :, None] ** 2 + d[:, 2, None, None, :] ** 2)
    v = _profile_unit(r / R.to(torch.float32)[:, None, None, None])
    gi = g.long()
    ok = [(gi[:, a] >= 0) & (gi[:, a] < shape[a]) for a in range(3)]
    inb = ok[0][:, :, None, None] & ok[1][:, None, :, None] & ok[2][:, None, None, :]
    base = (bi * C + ci)[:, None, None, None]
    idx = ((base * shape[0] + gi[:, 0, :, None, None]) * shape[1] + gi[:, 1, None, :, None]) * shape[2] \
        + gi[:, 2, None, None, :]
    sel = inb & (v > 0)
    flat.scatter_reduce_(0, idx[sel], v[sel], reduce="amax", include_self=True)


def render_tubes_torch(segments: torch.Tensor, out_shape_zyx: Sequence[int], n_classes: int,
                       batch_index: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor:
    """Soft tube targets ``[C, *shape]`` (``[B, C, *shape]``) from ``segments [M, 8]`` =
    ``z0, y0, x0, z1, y1, x1, class_index, d_vox``: the profile of the distance to the nearest
    segment of the channel divided by its ``R`` (round caps), in bounds only.

    Each segment is evaluated only in its box (segment bounds +/- ``R``, clipped to the output), the
    only voxels where the profile is non-zero, and the values are max-combined: the profile does not
    increase with the distance, so the max over segments is the profile of the nearest one."""
    shape = tuple(int(s) for s in out_shape_zyx)
    C = int(n_classes)
    B = int(batch_size) if batch_index is not None else 1
    segs, R, bi, ci = _prepare(segments, 8, 6, 7, C, batch_index)
    device = segments.device
    out = torch.zeros((B, C) + shape, dtype=torch.float32, device=device)
    if len(segs):
        p0, p1 = segs[:, 0:3], segs[:, 3:6]
        top = torch.tensor(shape, dtype=torch.float64, device=device) - 1
        lo = torch.ceil(torch.minimum(p0, p1) - R[:, None]).clamp_min(0)
        hi = torch.minimum(torch.floor(torch.maximum(p0, p1) + R[:, None]), top)
        ext = hi - lo + 1
        keep = (ext > 0).all(1)
        segs, R, bi, ci, lo = segs[keep], R[keep], bi[keep], ci[keep], lo[keep]
        if len(segs):
            w_all = ext[keep].amax(1).long()
            order = torch.argsort(w_all)
            segs, R, bi, ci, lo = segs[order], R[order], bi[order], ci[order], lo[order]
            widths = w_all[order].tolist()
            flat = out.view(-1)
            start = 0
            while start < len(segs):
                end = start + 1
                while end < len(segs) and (end + 1 - start) * widths[end] ** 3 <= _TUBE_BUDGET:
                    end += 1
                _tube_window(flat, segs[start:end], R[start:end], bi[start:end], ci[start:end], lo[start:end],
                             widths[end - 1], shape, C)
                start = end
    return out if batch_index is not None else out[0]


def _tube_window(flat, s, R, bi, ci, lo, w, shape, C):
    device = s.device
    g = lo[:, :, None] + torch.arange(w, device=device, dtype=torch.float64)    # [n, 3, w] voxel indices
    rel = (g - s[:, 0:3, None]).to(torch.float32)                                # voxel - p0
    ab = (s[:, 3:6] - s[:, 0:3]).to(torch.float32)
    L2 = (ab * ab).sum(1)
    wz, wy, wx = rel[:, 0, :, None, None], rel[:, 1, None, :, None], rel[:, 2, None, None, :]
    az, ay, ax = (ab[:, a, None, None, None] for a in range(3))
    t = (wz * az + wy * ay + wx * ax) / torch.where(L2 > 0, L2, torch.ones_like(L2))[:, None, None, None]
    t = torch.where(L2[:, None, None, None] > 0, t.clamp(0.0, 1.0), torch.zeros_like(t))
    d2 = (wz - t * az) ** 2 + (wy - t * ay) ** 2 + (wx - t * ax) ** 2
    v = _profile_unit(torch.sqrt(d2) / R.to(torch.float32)[:, None, None, None])
    gi = g.long()
    ok = [gi[:, a] < shape[a] for a in range(3)]
    inb = ok[0][:, :, None, None] & ok[1][:, None, :, None] & ok[2][:, None, None, :]
    base = (bi * C + ci)[:, None, None, None]
    idx = ((base * shape[0] + gi[:, 0, :, None, None]) * shape[1] + gi[:, 1, None, :, None]) * shape[2] \
        + gi[:, 2, None, None, :]
    sel = inb & (v > 0)
    flat.scatter_reduce_(0, idx[sel], v[sel], reduce="amax", include_self=True)
