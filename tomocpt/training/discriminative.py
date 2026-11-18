"""Discriminative training of pipeline-v2 pickers (Gate 7, ``07_AGENT_DISCRIMINATIVE.md``).

* (a) Supervised dense contrastive loss: anchors are the peak voxels of the (augmented) dense
  target, negatives are hard negatives ranked by the model's own score (with a cap that skips
  near-certain picks) and voxels of explicit negative masks; SupCon over the batch and a memory bank.
* (b) Focal value term: the CenterNet penalty-reduced focal loss on per-class probabilities.

Coordinates are target voxels ``(b, z, y, x)``; the model maps them onto its feature grid.
Every function takes its settings as arguments (invariant 3).

Human decision (1 Nov 2026): anchors and focal positives are the target's peak voxels (local
maxima) with value ``>= min_target`` (default 0.5); an off-grid centre's peak voxel is below 1.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["sample_anchors", "sample_hard_negatives", "ContrastiveMemoryBank", "supcon_loss",
           "focal_heatmap_loss", "soft_focal_penalty", "focal_positives"]

_BIG = 1e12


# --------------------------------------------------------------------------------------
# peaks
# --------------------------------------------------------------------------------------

def _peak_mask(target: torch.Tensor, min_target: float) -> torch.Tensor:
    """Non-strict local maxima (26-neighbourhood) with value ``>= min_target``. ``[B, C, ...]`` bool."""
    t = target.float()
    return (t >= F.max_pool3d(t, 3, 1, 1)) & (t >= float(min_target))


def _channel_peaks(target: torch.Tensor, min_target: float, wanted: torch.Tensor):
    """``(c, peaks [B, D, H, W] bool)`` per channel with a wanted ``(b, c)`` pair whose target
    reaches ``min_target``; one channel at a time, so memory does not grow with the class count
    (the full ``[B, C, ...]`` max-pool needs several GB at C ~ 134). Same peaks as ``_peak_mask``."""
    reach = (target.detach().flatten(2).amax(-1) >= float(min_target)).cpu() & wanted
    for c in range(target.shape[1]):
        if bool(reach[:, c].any()):
            yield c, _peak_mask(target[:, c:c + 1], min_target)[:, 0]


def _greedy_spacing(coords: np.ndarray, values: np.ndarray, min_dist: float) -> np.ndarray:
    """Indices kept by greedy non-maximum suppression (highest value first): every kept pair is
    at least ``min_dist`` apart."""
    order = np.argsort(-values, kind="stable")
    kept: list = []
    md2 = float(min_dist) ** 2
    for i in order:
        if kept:
            d2 = ((coords[kept] - coords[i]) ** 2).sum(1)
            if np.any(d2 < md2):
                continue
        kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def _instances(target, peaks, b: int, c: int, d: float, filament: bool) -> np.ndarray:
    """Instance voxels ``[K, 3]`` (z, y, x) of one ``(b, c)``: particle centres (peaks at least
    d/2 apart) or ridge voxels subsampled at one diameter. ``peaks``: ``[B, D, H, W]`` of channel c."""
    pz = torch.nonzero(peaks[b]).cpu().numpy()
    if not len(pz):
        return np.zeros((0, 3), np.int64)
    vals = target[b, c][tuple(torch.as_tensor(pz.T, device=target.device))].float().cpu().numpy()
    keep = _greedy_spacing(pz.astype(np.float64), vals, d if filament else d / 2.0)
    return pz[keep]


def sample_anchors(target: torch.Tensor, available: torch.Tensor, d_vox: torch.Tensor,
                   is_filament: Sequence[bool], min_target: float = 0.5, max_per_class: int = 32,
                   generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contrastive anchors from a dense target.

    Args:
        target: ``[B, C, D, H, W]`` soft targets (after augmentation).
        available: ``[B, C]`` bool; unavailable pairs give no anchor.
        d_vox: ``[B, C]`` class diameters in target voxels.
        is_filament: one bool per channel.
        min_target: lowest target value of an anchor voxel.
        max_per_class: at most this many anchors per class in the batch (random subset).
        generator: CPU generator for the subset.

    Particle anchors are the peak voxels of the instances (local maxima at least d/2 apart);
    filament anchors are ridge voxels (local maxima across the tube) at least one diameter apart.

    Returns ``(index [N, 4] int64 = (b, z, y, x), label [N] int64 channel)`` on the CPU.
    """
    B, C = target.shape[:2]
    avail = torch.as_tensor(available).bool().view(B, C).cpu()
    d = torch.as_tensor(d_vox).float().view(B, C).cpu()
    idx, lab = [], []
    for c, peaks in _channel_peaks(target, min_target, avail):
        rows = []
        for b in range(B):
            if not bool(avail[b, c]):
                continue
            inst = _instances(target, peaks, b, c, float(d[b, c]), bool(is_filament[c]))
            rows.extend([b, *map(int, v)] for v in inst)
        if len(rows) > int(max_per_class):
            perm = torch.randperm(len(rows), generator=generator)[: int(max_per_class)].tolist()
            rows = [rows[i] for i in sorted(perm)]
        idx.extend(rows)
        lab.extend([c] * len(rows))
    return (torch.tensor(idx, dtype=torch.long).reshape(-1, 4), torch.tensor(lab, dtype=torch.long))


def focal_positives(target: torch.Tensor, available: torch.Tensor, d_vox: torch.Tensor,
                    is_filament: Sequence[bool], min_target: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Positives and instance counts of the focal value term.

    Positives: particle peak voxels (local maxima at least d/2 apart), filament ridge voxels (all
    local maxima across the tube). Instances: particle centres; filament ridge length in units of
    one diameter (ridge voxels subsampled at one diameter). Returns ``(positive [B, C, D, H, W]
    bool, num_instances [B, C] float32)`` on the device of ``target``."""
    B, C = target.shape[:2]
    avail = torch.as_tensor(available).bool().view(B, C).cpu()
    d = torch.as_tensor(d_vox).float().view(B, C).cpu()
    positive = torch.zeros(target.shape, dtype=torch.bool, device=target.device)
    n = torch.zeros(B, C, dtype=torch.float32)
    for c, peaks in _channel_peaks(target, min_target, avail):
        for b in range(B):
            if not bool(avail[b, c]):
                continue
            fil = bool(is_filament[c])
            inst = _instances(target, peaks, b, c, float(d[b, c]), fil)
            n[b, c] = float(len(inst))
            if fil:
                positive[b, c] = peaks[b]
            elif len(inst):
                ii = torch.as_tensor(inst, device=target.device)
                positive[b, c, ii[:, 0], ii[:, 1], ii[:, 2]] = True
    return positive, n.to(target.device)


# --------------------------------------------------------------------------------------
# hard negatives
# --------------------------------------------------------------------------------------

def _squared_distance_to(mask: torch.Tensor) -> torch.Tensor:
    """Exact squared Euclidean distance of every voxel to the nearest True voxel of a 3-D
    ``mask [D, H, W]`` (separable min-plus transform; ``_BIG`` when the mask is empty)."""
    f = torch.where(mask, torch.zeros((), device=mask.device), torch.full((), _BIG, device=mask.device)).float()
    for axis in range(3):
        n = f.shape[axis]
        x = torch.arange(n, device=f.device, dtype=torch.float32)
        sq = (x[:, None] - x[None, :]) ** 2  # [out, in]
        g = f.movedim(axis, -1)  # [..., in]
        g = (g.unsqueeze(-2) + sq).amin(-1)  # [..., out]
        f = g.movedim(-1, axis)
    return f


def _exclusion_zone(target: torch.Tensor, d_vox: torch.Tensor, exclusion_radius: float) -> torch.Tensor:
    """``[B, D, H, W]`` bool: voxels inside a labelled object (target > 0) of any channel or within
    ``exclusion_radius`` x the class radius of it."""
    B, C = target.shape[:2]
    out = torch.zeros((B,) + tuple(target.shape[2:]), dtype=torch.bool, device=target.device)
    d = torch.as_tensor(d_vox).float().view(B, C)
    for b in range(B):
        for c in range(C):
            sup = target[b, c] > 0
            if not bool(sup.any()):
                continue
            r = float(exclusion_radius) * float(d[b, c]) / 2.0
            if r <= 0:
                out[b] |= sup
            else:
                out[b] |= _squared_distance_to(sup) <= r * r
    return out


def sample_hard_negatives(prob: torch.Tensor, target: torch.Tensor, d_vox: torch.Tensor, n: int,
                          min_score: float = 0.3, max_score: float = 0.9, exclusion_radius: float = 1.0,
                          negative_mask: Optional[torch.Tensor] = None, n_mask: int = 0,
                          generator: Optional[torch.Generator] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negatives for the contrastive loss.

    Args:
        prob: detached class probabilities ``[B, C, D, H, W]``; the score of a voxel is the
            highest class probability.
        target: ``[B, C, D, H, W]`` soft targets; no negative lies inside a labelled object or
            within ``exclusion_radius`` x its class radius (``d_vox / 2``).
        d_vox: ``[B, C]`` class diameters in target voxels.
        n: hard negatives per batch: the highest-scoring voxels with score in
            ``[min_score, max_score]`` (the cap skips near-certain picks, often unlabelled particles).
        negative_mask: optional ``[B, 1, D, H, W]`` bool of structures that are never a class.
        n_mask: negatives per batch drawn at random from ``negative_mask`` (outside the exclusion zone).

    Returns ``(index [M, 4] int64 (b, z, y, x), score [M], from_mask [M] bool)`` on the CPU: the
    hard negatives first (score descending), then the mask negatives.
    """
    with torch.no_grad():
        score = prob.detach().float().amax(1)  # [B, D, H, W]
        excl = _exclusion_zone(target.detach(), d_vox, exclusion_radius)
        shape = tuple(score.shape)
        eligible = (~excl) & (score >= float(min_score)) & (score <= float(max_score))
        k = min(int(n), int(eligible.sum()))
        if k > 0:
            flat = torch.where(eligible, score, torch.full_like(score, -1.0)).flatten()
            top = torch.topk(flat, k).indices  # sorted descending
            idx = torch.stack(torch.unravel_index(top, shape), 1).cpu()
            sc = flat[top].cpu()
        else:
            idx, sc = torch.zeros(0, 4, dtype=torch.long), torch.zeros(0)
        from_mask = torch.zeros(len(idx), dtype=torch.bool)
        if negative_mask is not None and int(n_mask) > 0:
            m = (negative_mask.bool()[:, 0] & ~excl).flatten()
            if k > 0:
                m[top] = False
            cand = torch.nonzero(m).flatten().cpu()
            if len(cand):
                pick = cand[torch.randperm(len(cand), generator=generator)[: int(n_mask)]]
                midx = torch.stack(torch.unravel_index(pick, shape), 1)
                idx = torch.cat([idx, midx])
                sc = torch.cat([sc, score.flatten()[pick.to(score.device)].cpu()])
                from_mask = torch.cat([from_mask, torch.ones(len(pick), dtype=torch.bool)])
    return idx.long(), sc, from_mask


# --------------------------------------------------------------------------------------
# memory bank
# --------------------------------------------------------------------------------------

class ContrastiveMemoryBank(nn.Module):
    """Per-class FIFO queues of detached embeddings plus one queue for negatives (label -1).

    The queues are registered buffers (saved in ``state_dict``, never parameters). Under DDP each
    rank keeps its own queues.

    Gate 9: ``max_age > 0`` (``train.discriminative.bank_max_age``) makes ``get(step)`` skip entries
    pushed more than ``max_age`` steps before ``step``. The step of each entry is kept in the
    non-persistent buffer ``stamp`` (-1 = unknown), so the ``state_dict`` keys stay ``queue``,
    ``ptr``, ``count`` and old and new checkpoints load strictly either way. Entries restored from
    a checkpoint have unknown age; the first ``get`` / ``push`` with a step stamps them with that
    step, so they are used for at most ``max_age`` more steps."""

    def __init__(self, n_classes: int, size: int, dim: int, max_age: int = 0):
        super().__init__()
        self.n_classes, self.size, self.dim = int(n_classes), int(size), int(dim)
        self.max_age = max(0, int(max_age or 0))
        self.register_buffer("queue", torch.zeros(self.n_classes + 1, self.size, self.dim))
        self.register_buffer("ptr", torch.zeros(self.n_classes + 1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(self.n_classes + 1, dtype=torch.long))
        self.register_buffer("stamp", torch.full((self.n_classes + 1, self.size), -1, dtype=torch.long),
                             persistent=False)

    @torch.no_grad()
    def _stamp_unknown(self, step: Optional[int]) -> None:
        """Give the stored entries of unknown age (restored from a checkpoint) the step ``step``."""
        if step is None:
            return
        filled = torch.arange(self.size, device=self.stamp.device)[None, :] < self.count[:, None]
        self.stamp[filled & (self.stamp < 0)] = int(step)

    @torch.no_grad()
    def push(self, embeddings: torch.Tensor, labels: torch.Tensor, step: Optional[int] = None) -> None:
        """Append ``embeddings [N, dim]`` to the queues of their ``labels [N]`` (-1 = negatives),
        stamped with the global step ``step`` (None = unknown)."""
        self._stamp_unknown(step)
        e = embeddings.detach().to(self.queue.device, self.queue.dtype)
        labels = labels.to(self.queue.device).long()
        for q in labels.unique().tolist():
            row = self.n_classes if q < 0 else int(q)
            if row > self.n_classes:
                continue
            v = e[labels == q][-self.size:]
            k = len(v)
            pos = (int(self.ptr[row]) + torch.arange(k, device=v.device)) % self.size
            self.queue[row, pos] = v
            self.stamp[row, pos] = -1 if step is None else int(step)
            self.ptr[row] = (int(self.ptr[row]) + k) % self.size
            self.count[row] = min(self.size, int(self.count[row]) + k)

    def get(self, step: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every stored entry: ``(embeddings [K, dim], labels [K])`` (detached). With ``max_age > 0``
        and a ``step``, only the entries at most ``max_age`` steps old."""
        aged = self.max_age > 0 and step is not None
        if aged:
            self._stamp_unknown(step)
        es, ls = [], []
        for row in range(self.n_classes + 1):
            k = int(self.count[row])
            if k:
                e = self.queue[row, :k]
                if aged:
                    e = e[(int(step) - self.stamp[row, :k]) <= self.max_age]
                    k = len(e)
                    if not k:
                        continue
                es.append(e)
                ls.append(torch.full((k,), -1 if row == self.n_classes else row, dtype=torch.long,
                                     device=self.queue.device))
        if not es:
            return self.queue.new_zeros(0, self.dim), torch.zeros(0, dtype=torch.long, device=self.queue.device)
        return torch.cat(es).detach(), torch.cat(ls)


# --------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------

def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, bank_embeddings: Optional[torch.Tensor] = None,
                bank_labels: Optional[torch.Tensor] = None, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020, L_out), computed in float32.

    ``embeddings [N, d]`` with ``labels [N]`` (``-1`` = background / negative: never an anchor,
    always in the denominator); the bank entries are candidates (positives and denominator), never
    anchors. Embeddings are L2-normalised here. An anchor's loss is the negative mean, over its
    positives (same label, other entries), of ``log softmax`` of its similarities to every other
    entry. Anchors without a positive are skipped; with none the loss is ``0`` with a gradient path.
    """
    labels = labels.to(embeddings.device).long()
    z = F.normalize(embeddings.float(), dim=1)
    allz, ally = z, labels
    if bank_embeddings is not None and len(bank_embeddings):
        allz = torch.cat([z, F.normalize(bank_embeddings.detach().float().to(z.device), dim=1)])
        ally = torch.cat([labels, bank_labels.to(z.device).long()])
    N = z.shape[0]
    zero = 0.0 * embeddings.float().sum()
    if N == 0 or not bool((labels >= 0).any()):
        return zero
    s = (z @ allz.T) / float(temperature)  # [N, N + K]
    self_mask = torch.zeros_like(s, dtype=torch.bool)
    self_mask[torch.arange(N), torch.arange(N)] = True
    s = s.masked_fill(self_mask, float("-inf"))
    logp = s - torch.logsumexp(s, dim=1, keepdim=True)
    pos = (labels[:, None] == ally[None, :]) & ~self_mask & (labels[:, None] >= 0)
    npos = pos.sum(1)
    ok = npos > 0
    if not bool(ok.any()):
        return zero
    per = -torch.where(pos, logp, torch.zeros_like(logp)).sum(1)[ok] / npos[ok].float()
    return per.mean() + zero


def soft_focal_penalty(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Per-voxel soft-target ("quality") focal penalty (Li et al. 2020, Generalized Focal Loss):
    ``|y - p|^gamma * BCE(p, y)``, in float32. It is >= 0 and 0 only at ``p = y``, so its minimiser
    is the Huber term's: it keeps soft targets (human decision, 1 Nov 2026). The picking loss
    weights it with the Huber weights (``multiclass_picking_loss(value_loss="focal")``)."""
    p = pred.float().clamp(1e-6, 1.0 - 1e-6)
    y = target.float()
    bce = -(y * torch.log(p) + (1.0 - y) * torch.log1p(-p))
    return (y - p).abs() ** gamma * bce


def focal_heatmap_loss(pred: torch.Tensor, target: torch.Tensor, available: torch.Tensor,
                       positive: torch.Tensor, num_instances: torch.Tensor, alpha: float = 2.0,
                       beta: float = 4.0, reduction: str = "mean") -> torch.Tensor:
    """CenterNet penalty-reduced focal loss (Zhou et al. 2019) per ``(b, c)``.

    Not used by training (human decision, 1 Nov 2026): its minimiser is 0 at every non-peak voxel
    and, normalised by the instance count, it collapsed to a flat output in the Gate 7 slow tests.
    ``value_loss = "focal"`` uses :func:`soft_focal_penalty` with the Huber weights instead.

    ``pred``: probabilities ``[B, C, D, H, W]``; ``target``: soft targets; ``positive``: bool, the
    voxels treated as ``y = 1`` (peak voxels); ``num_instances [B, C]``. Per pair
    ``-(sum_pos (1-p)^alpha log p + sum_other (1-y)^beta p^alpha log(1-p)) / max(n, 1)``.
    ``reduction="mean"``: mean over available pairs (0 if none); ``"none"``: ``[B, C]`` (unavailable
    pairs 0)."""
    B, C = pred.shape[:2]
    p = pred.float().clamp(1e-6, 1.0 - 1e-6)
    y = target.float()
    pos = positive.bool()
    term = torch.where(pos, (1.0 - p) ** alpha * torch.log(p),
                       (1.0 - y).clamp(min=0.0) ** beta * p ** alpha * torch.log1p(-p))
    per = -term.flatten(2).sum(2) / num_instances.to(device=p.device, dtype=torch.float32).view(B, C).clamp(min=1.0)
    avail = torch.as_tensor(available).to(p.device).bool().view(B, C)
    per = torch.where(avail, per, torch.zeros_like(per))
    if reduction == "none":
        return per
    return per.sum() / avail.sum().clamp(min=1)
