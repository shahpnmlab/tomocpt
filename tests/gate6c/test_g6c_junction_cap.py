"""Gate 6c addendum 2: a skeleton node with very many arms is not a filament crossing.

A percolated noisy region contracts into one node with thousands of arms, and pairing them by
direction is quadratic in the arms. Above ``MAX_JUNCTION_ARMS`` the arms stay unpaired."""
import numpy as np

from tomocpt.labels import filaments as LF


def _star(directions, length=40):
    size = 2 * length + 3
    skel = np.zeros((size, size, size), dtype=bool)
    c = length + 1
    for d in directions:
        d = np.asarray(d, dtype=np.float64)
        d = d / np.abs(d).max()
        for t in range(length + 1):
            z, y, x = np.round(c + t * d).astype(int)
            skel[z, y, x] = True
    return skel


def test_real_crossing_is_still_paired():
    arms = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    assert len(LF._graph_polylines(_star(arms), 3.0)) == 3


def test_cap_leaves_ordinary_junctions_alone():
    assert LF.MAX_JUNCTION_ARMS >= 16


def test_arms_above_the_cap_stay_unpaired(monkeypatch):
    arms = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    monkeypatch.setattr(LF, "MAX_JUNCTION_ARMS", 5)
    assert len(LF._graph_polylines(_star(arms), 3.0)) == 6
