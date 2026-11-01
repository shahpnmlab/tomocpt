"""Helpers for the Gate 7 tests (no tests here).

New public names (spec ``07_AGENT_DISCRIMINATIVE.md``, "Public names") are imported inside the
functions, so the modules collect before the implementation exists.

Orchestrator rulings on the call signatures (the spec names the functions but not their
arguments; recorded in ``tests/gate7/README.md``):

``sample_anchors(target, available, d_vox, is_filament, min_target=0.5, max_per_class=32, generator=None)``
    ``target [B, C, D, H, W]``, ``available [B, C]`` bool, ``d_vox [B, C]``, ``is_filament``: ``C`` bools.
    Returns ``(index [N, 4] int64 = (b, z, y, x) in target voxels, label [N] int64 channel)``.
``sample_hard_negatives(prob, target, d_vox, n, min_score=0.3, max_score=0.9, exclusion_radius=1.0,
negative_mask=None, n_mask=0, generator=None)``
    ``prob``: detached class probabilities ``[B, C, D, H, W]``; score of a voxel = max over channels.
    ``negative_mask [B, 1, D, H, W]`` bool or ``None``. Returns ``(index [M, 4] int64, score [M],
    from_mask [M] bool)``: the score-ranked hard negatives first (descending), then mask negatives.
``ContrastiveMemoryBank(n_classes, size, dim)`` (``nn.Module``)
    ``push(embeddings [N, dim], labels [N])`` (label ``-1`` = negative queue);
    ``get() -> (embeddings [K, dim], labels [K])``, every stored entry.
``supcon_loss(embeddings, labels, bank_embeddings=None, bank_labels=None, temperature=0.1)``
    Label ``-1`` = background / negative (never an anchor). Embeddings are L2-normalised inside.
``focal_heatmap_loss(pred, target, available, positive, num_instances, alpha=2.0, beta=4.0)``
    ``positive [B, C, D, H, W]`` bool (the peak voxels, CenterNet's ``y = 1``), ``num_instances
    [B, C]``. Per pair ``-(sum_pos (1-p)^a log p + sum_other (1-y)^b p^a log(1-p)) / max(n, 1)``;
    mean over available pairs.

Human decision (1 Nov 2026): the anchors and focal positives are the peak voxels (local maxima) of
the target with value ``>= anchor_min_target``, default **0.5** (not 0.99: an off-grid centre's
peak voxel is 0.54-0.92 for 7-20 voxel particles). The picking model keeps the head as
``BasePickingModel.contrastive_head`` (a ``DenseProjectionHead``; ``None`` when off).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

CHUNK = 64


def render_particles(points, n_classes, shape=(CHUNK,) * 3) -> np.ndarray:
    """``render_targets`` of ``[N, 5]`` rows ``z, y, x, class_index, d_vox``."""
    from tomocpt.labels.helpers import render_targets

    return render_targets(np.asarray(points, np.float64).reshape(-1, 5), shape, n_classes, "clip")


def render_line_tube(p0_zyx, p1_zyx, class_index, d_vox, n_classes, shape=(CHUNK,) * 3) -> np.ndarray:
    from tomocpt.labels.filaments import render_tubes

    seg = np.array([[*p0_zyx, *p1_zyx, class_index, d_vox]], np.float32)
    return render_tubes(seg, shape, n_classes)


def point_segment_distance(p, a, b) -> np.ndarray:
    """Distance of points ``p [N, 3]`` to the segment ``a``-``b``."""
    p, a, b = (np.asarray(v, np.float64) for v in (p, a, b))
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(1e-12, ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def well_separated_centres(rng, n, d_vox, margin, min_sep, shape=(CHUNK,) * 3):
    """``n`` off-grid centres (z, y, x), ``margin`` from the border, pairwise ``>= min_sep``."""
    out = []
    for _ in range(100000):
        c = np.array([rng.uniform(margin, s - 1 - margin) for s in shape])
        if all(np.linalg.norm(c - o) >= min_sep for o in out):
            out.append(c)
            if len(out) == n:
                return np.array(out)
    raise RuntimeError("could not place centres")


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


# --------------------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------------------

def disc_config(tmp_path, particles, filaments=(), class_head="softmax", cond=False, **disc):
    """Tiny CPU v2 config (``g6_helpers.v2_config_with_filaments``) with
    ``train.discriminative`` fields set from ``disc``."""
    from gate6 import g6_helpers as H6

    cfg = H6.v2_config_with_filaments(Path(tmp_path), particles, filaments, cond=cond, class_head=class_head)
    for k, v in disc.items():
        setattr(cfg.train.discriminative, k, v)
    return cfg


def trainable_not_in_optimizer(model, trainer) -> list:
    ids = {id(p) for opt in trainer.optimizers for g in opt.param_groups for p in g["params"]}
    return sorted(n for n, p in model.named_parameters() if p.requires_grad and id(p) not in ids)


# --------------------------------------------------------------------------------------
# C9 look-alike phantom
# --------------------------------------------------------------------------------------

def add_sheets(clean_volume: np.ndarray, n_sheets: int, thickness_vox: float, rng, exclude: np.ndarray):
    """Add ``n_sheets`` flat dark slabs (membrane-like planes through the volume, random
    orientation, ``thickness_vox`` thick) to a noise-free density ``clean_volume`` (dark = negative).
    ``exclude`` (bool) marks voxels kept free (labelled objects plus a margin). Returns
    ``(volume, sheet_mask)``; the mask is the slab dilated by one voxel, outside ``exclude``."""
    shape = clean_volume.shape
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
    vol = clean_volume.copy()
    mask = np.zeros(shape, bool)
    for _ in range(n_sheets):
        nrm = rng.normal(size=3)
        nrm /= np.linalg.norm(nrm)
        c = np.array([rng.uniform(0.3 * s, 0.7 * s) for s in shape])
        dist = np.abs((zz - c[0]) * nrm[0] + (yy - c[1]) * nrm[1] + (xx - c[2]) * nrm[2])
        slab = (dist <= thickness_vox / 2) & ~exclude
        vol = np.where(slab, np.minimum(vol, -1.0), vol)
        mask |= (dist <= thickness_vox / 2 + 1.0) & ~exclude
    return vol.astype(np.float32), mask
