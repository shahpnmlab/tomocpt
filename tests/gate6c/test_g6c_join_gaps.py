"""Gate 6c addendum 2: ``_join_gaps`` must stay exact and scale to noisy predictions.

A noisy map gives thousands of fragments; the first version compared all pairs of ends for each
merge and held calibration for over an hour on one tomogram. This function also builds training
labels, so a faster version must return the same polylines, in the same order, ties included."""
import time

import numpy as np
import pytest

from tomocpt.labels import filaments as LF


def _reference(polys, max_gap, R):
    """The all-pairs version, as it was at Gate 6."""
    polys = [np.asarray(p, dtype=np.float64) for p in polys if len(p) >= 2]

    def end_dir(p, at_start):
        q = p if at_start else p[::-1]
        s = LF._cumlen(q)
        sel = s <= min(2.0 * R, s[-1])
        b = q[sel] if sel.sum() >= 2 else q[:2]
        d = b[0] - b[-1]
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.zeros(3)

    merged = True
    while merged and len(polys) > 1:
        merged = False
        best = None
        for a in range(len(polys)):
            for b in range(len(polys)):
                if a == b:
                    continue
                for ea in (0, 1):
                    for eb in (0, 1):
                        pa = polys[a][0] if ea == 0 else polys[a][-1]
                        pb = polys[b][0] if eb == 0 else polys[b][-1]
                        gap = float(np.linalg.norm(pb - pa))
                        if gap > max_gap:
                            continue
                        da, db = end_dir(polys[a], ea == 0), end_dir(polys[b], eb == 0)
                        if float(da @ db) > -0.7:
                            continue
                        if gap > 1e-6:
                            gv = (pb - pa) / gap
                            if float(da @ gv) < 0.5:
                                continue
                        if best is None or gap < best[0]:
                            best = (gap, a, ea, b, eb)
        if best is not None:
            _, a, ea, b, eb = best
            pa = polys[a] if ea == 1 else polys[a][::-1]
            pb = polys[b] if eb == 0 else polys[b][::-1]
            new = np.concatenate([pa, pb], axis=0)
            polys = [p for k, p in enumerate(polys) if k not in (a, b)] + [new]
            merged = True
    return polys


def _fragments(rng, n, extent, on_grid, lengths=(2, 8)):
    """Short straight-ish pieces; ``on_grid`` keeps integer coordinates, which makes equal gaps."""
    out = []
    for _ in range(n):
        start = rng.uniform(0, extent, 3)
        direction = rng.choice([-1.0, 0.0, 1.0], 3)
        if not direction.any():
            direction[0] = 1.0
        k = int(rng.integers(lengths[0], lengths[1]))
        p = start + np.arange(k)[:, None] * direction + rng.normal(0, 0.0 if on_grid else 0.2, (k, 3))
        out.append(np.round(p) if on_grid else p)
    return out


@pytest.mark.parametrize("on_grid", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_same_as_all_pairs(seed, on_grid):
    rng = np.random.default_rng(seed)
    polys = _fragments(rng, 60, 30.0, on_grid)
    got, want = LF._join_gaps(polys, 5.0, 3.0), _reference(polys, 5.0, 3.0)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert np.array_equal(g, w)


def test_collinear_pieces_join_and_short_ones_are_dropped():
    pieces = [np.array([[i * 6.0, 0, 0], [i * 6.0 + 4.0, 0, 0]]) for i in range(5)]
    pieces.append(np.zeros((1, 3)))
    got = LF._join_gaps(pieces, 3.0, 2.0)
    assert len(got) == 1 and len(got[0]) == 10


def test_many_fragments_are_fast():
    rng = np.random.default_rng(0)
    polys = _fragments(rng, 6000, 300.0, False)
    t0 = time.perf_counter()
    LF._join_gaps(polys, 5.0, 3.0)
    assert time.perf_counter() - t0 < 10.0
