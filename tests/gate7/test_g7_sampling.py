"""G7-C2 ``sample_anchors`` and G7-C3 ``sample_hard_negatives`` (signatures: ``g7_helpers``).

Targets are drawn with the production renderers (``render_targets``, ``render_tubes``). Particle
centres are off-grid, so the peak voxel is below 1 (human decision 1 Nov 2026: anchors are peak
voxels ``>= anchor_min_target``, default 0.5).

Tolerances: a particle anchor is the voxel nearest to the centre, so every coordinate is within
0.5 voxel (spec). A filament anchor lies within 0.5 voxel of the centreline when the centreline
runs through voxel centres (spec); for an oblique off-grid centreline the nearest voxel centre can
be up to sqrt(2)/2 = 0.71 voxel away, so that case uses 0.75 (orchestrator ruling).
"""

import numpy as np
import pytest
import torch

from gate7 import g7_helpers as H

C = 3
D_VOX = (12.0, 8.0, 7.0)  # particle, particle, filament
IS_FIL = (False, False, True)


def _particle_batch(seed=0, n_per_class=(4, 5), B=2):
    rng = np.random.default_rng(seed)
    tgt, centres = [], []
    for b in range(B):
        pts, cb = [], {}
        all_c = H.well_separated_centres(rng, sum(n_per_class), max(D_VOX), margin=8, min_sep=16)
        k = 0
        for c, n in enumerate(n_per_class):
            cb[c] = all_c[k:k + n]
            for z, y, x in cb[c]:
                pts.append([z, y, x, c, D_VOX[c]])
            k += n
        tgt.append(H.render_particles(pts, C))
        centres.append(cb)
    return torch.from_numpy(np.stack(tgt)), centres


def _anchors(target, available=None, **kw):
    from tomocpt.training.discriminative import sample_anchors

    B = target.shape[0]
    avail = torch.ones(B, C, dtype=torch.bool) if available is None else available
    d = torch.tensor([D_VOX] * B, dtype=torch.float32)
    kw.setdefault("max_per_class", 1000)
    idx, lab = sample_anchors(target, avail, d, IS_FIL, generator=torch.Generator().manual_seed(0), **kw)
    assert idx.dtype == torch.int64 and lab.dtype == torch.int64
    assert idx.ndim == 2 and idx.shape[1] == 4 and lab.shape == (idx.shape[0],)
    return idx.numpy(), lab.numpy()


def _check_particle_anchors(idx, lab, centres, B):
    for b in range(B):
        for c in (0, 1):
            got = idx[(idx[:, 0] == b) & (lab == c)][:, 1:].astype(np.float64)
            true = centres[b][c]
            assert len(got) == len(true), f"sample {b} class {c}: {len(got)} anchors for {len(true)} centres"
            for t in true:
                err = np.abs(got - t).max(axis=1)
                assert err.min() <= 0.5 + 1e-6, f"sample {b} class {c}: no anchor within 0.5 voxel of {t}"


def test_G7_C2_particle_anchors_are_centres():
    target, centres = _particle_batch()
    assert float(target.max()) < 0.99, "off-grid centres: no voxel reaches 0.99"
    idx, lab = _anchors(target)
    _check_particle_anchors(idx, lab, centres, 2)


def test_G7_C2_particle_anchors_follow_a_flip():
    target, centres = _particle_batch(seed=3)
    for axis in (0, 1, 2):
        flipped = torch.flip(target, dims=(2 + axis,))
        fc = [{c: v.copy() for c, v in cb.items()} for cb in centres]
        for cb in fc:
            for c in cb:
                cb[c][:, axis] = 63 - cb[c][:, axis]
        idx, lab = _anchors(flipped)
        _check_particle_anchors(idx, lab, fc, 2)


def test_G7_C2_min_target_is_applied():
    target, _ = _particle_batch()
    idx, _ = _anchors(target * 0.4, min_target=0.5)
    assert len(idx) == 0
    idx, _ = _anchors(target, min_target=0.999)
    assert len(idx) == 0


def test_G7_C2_unavailable_pairs_give_no_anchors():
    target, _ = _particle_batch()
    avail = torch.ones(2, C, dtype=torch.bool)
    avail[1, 0] = False
    avail[0, 1] = False
    idx, lab = _anchors(target, available=avail)
    assert not np.any((idx[:, 0] == 1) & (lab == 0))
    assert not np.any((idx[:, 0] == 0) & (lab == 1))
    assert np.any((idx[:, 0] == 0) & (lab == 0)) and np.any((idx[:, 0] == 1) & (lab == 1))


def test_G7_C2_per_class_cap_per_batch():
    target, _ = _particle_batch(n_per_class=(6, 12))
    idx, lab = _anchors(target, max_per_class=5)
    for c in (0, 1):
        n = int((lab == c).sum())
        assert 1 <= n <= 5, (c, n)


@pytest.mark.parametrize("case", ["axis", "diagonal", "offgrid"])
def test_G7_C2_filament_anchors_on_centreline_spaced(case):
    lines = {
        "axis": ((32.0, 20.0, 4.0), (32.0, 20.0, 59.0)),
        "diagonal": ((30.0, 6.0, 6.0), (30.0, 56.0, 56.0)),
        "offgrid": ((12.3, 9.6, 7.2), (50.6, 48.1, 54.7)),
    }
    tol = 0.75 if case == "offgrid" else 0.5
    p0, p1 = lines[case]
    tgt = []
    for b in range(2):
        tgt.append(H.render_line_tube(p0, p1, 2, D_VOX[2], C))
    target = torch.from_numpy(np.stack(tgt))
    idx, lab = _anchors(target)
    assert set(lab.tolist()) <= {2}
    for b in range(2):
        a = idx[(idx[:, 0] == b) & (lab == 2)][:, 1:].astype(np.float64)
        assert len(a) >= 2, f"{case}: {len(a)} anchors on a {np.linalg.norm(np.subtract(p1, p0)):.0f}-voxel tube"
        dist = H.point_segment_distance(a, p0, p1)
        assert dist.max() <= tol + 1e-6, f"{case}: anchor {dist.max():.3f} voxel from the centreline"
        pair = np.linalg.norm(a[:, None] - a[None], axis=2) + np.eye(len(a)) * 1e9
        assert pair.min() >= D_VOX[2] - 1e-6, f"{case}: anchors {pair.min():.2f} voxel apart (< one diameter)"


# --------------------------------------------------------------------------------------
# G7-C3
# --------------------------------------------------------------------------------------

def _hard(prob, target, n=64, **kw):
    from tomocpt.training.discriminative import sample_hard_negatives

    B = target.shape[0]
    d = torch.tensor([D_VOX] * B, dtype=torch.float32)
    idx, score, from_mask = sample_hard_negatives(prob, target, d, n, generator=torch.Generator().manual_seed(0), **kw)
    assert idx.dtype == torch.int64 and idx.ndim == 2 and idx.shape[1] == 4
    assert score.shape == (idx.shape[0],) and from_mask.shape == (idx.shape[0],) and from_mask.dtype == torch.bool
    return idx.numpy(), score.float().numpy(), from_mask.numpy()


def _margin(idx, centres, excl):
    """Per entry: min over labelled objects of ``distance to the centre - (1 + excl) R + 1``
    (one voxel slack for a discrete dilation). Must be >= 0."""
    out = []
    for b, z, y, x in idx:
        best = np.inf
        for c, cs in centres[b].items():
            if len(cs):
                R = D_VOX[c] / 2
                d = np.linalg.norm(cs - np.array([z, y, x]), axis=1)
                best = min(best, float((d - (1 + excl) * R + 1.0).min()))
        out.append(best)
    return np.array(out)


@pytest.mark.parametrize("excl", [0.5, 1.0])
def test_G7_C3_hard_negatives_exclusion_range_ranking(excl):
    target, centres = _particle_batch(seed=5, n_per_class=(6, 6))
    g = torch.Generator().manual_seed(1)
    prob = torch.rand(target.shape, generator=g)
    idx, score, from_mask = _hard(prob, target, n=64, min_score=0.3, max_score=0.9, exclusion_radius=excl)
    assert not from_mask.any()
    assert len(idx) == 64
    # score = max class probability at the voxel, inside [min, max], ranked descending
    pmax = prob.max(dim=1).values.numpy()
    np.testing.assert_allclose(score, pmax[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]], rtol=1e-6)
    assert score.min() >= 0.3 - 1e-6 and score.max() <= 0.9 + 1e-6
    assert np.all(np.diff(score) <= 1e-7), "hard negatives must be ranked by score (descending)"
    # never within exclusion_radius x R of a labelled object (one voxel slack for the dilation)
    margin = _margin(idx, centres, excl)
    assert np.all(margin >= -1e-6), f"hard negative inside the exclusion zone (margin {margin.min():.2f} voxel)"
    # top-n: no clearly eligible voxel (2 voxels beyond the exclusion zone) outranks a selected one
    B = target.shape[0]
    zz, yy, xx = np.meshgrid(*[np.arange(64)] * 3, indexing="ij")
    worst = score.min()
    chosen = {tuple(v) for v in idx.tolist()}
    for b in range(B):
        far = np.ones((64,) * 3, bool)
        for c, cs in centres[b].items():
            R = D_VOX[c] / 2
            for z, y, x in cs:
                far &= np.sqrt((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) > (1 + excl) * R + 2
        cand = far & (pmax[b] >= 0.3) & (pmax[b] <= 0.9) & (pmax[b] > worst + 1e-6)
        for z, y, x in zip(*np.nonzero(cand)):
            assert (b, z, y, x) in chosen, f"eligible voxel {(b, z, y, x)} with score {pmax[b, z, y, x]:.4f} " \
                                           f"> lowest selected {worst:.4f} was not selected"


def test_G7_C3_scores_outside_the_range_are_never_selected():
    target, _ = _particle_batch(seed=6)
    prob = torch.full(target.shape, 0.95)
    prob[:, :, :, :, :8] = 0.1  # low scores everywhere in a slab
    idx, score, _ = _hard(prob, target, n=32, min_score=0.3, max_score=0.9)
    assert len(idx) == 0, "no voxel has a score inside [0.3, 0.9]"


def test_G7_C3_mask_negatives_come_only_from_the_mask():
    target, centres = _particle_batch(seed=7)
    prob = torch.rand(target.shape, generator=torch.Generator().manual_seed(2))
    mask = torch.zeros(2, 1, 64, 64, 64, dtype=torch.bool)
    mask[0, 0, 40:44] = True
    idx, score, from_mask = _hard(prob, target, n=16, negative_mask=mask, n_mask=24, exclusion_radius=1.0)
    m = idx[from_mask]
    assert 1 <= len(m) <= 24
    assert np.all(m[:, 0] == 0) and np.all((m[:, 1] >= 40) & (m[:, 1] < 44)), "mask negatives outside the mask"
    assert np.all(_margin(m, centres, 1.0) >= -1e-6), "mask negative inside the exclusion zone of an object"
    h = idx[~from_mask]
    assert len(h) <= 16
    # no mask -> no mask negatives
    _, _, fm = _hard(prob, target, n=16, negative_mask=None, n_mask=24)
    assert not fm.any()
    _, _, fm = _hard(prob, target, n=16, negative_mask=mask, n_mask=0)
    assert not fm.any()
