"""``benchmarks.filaments.centerline_scores`` (spec public name; signature by orchestrator ruling,
31 Oct 2026): ``centerline_scores(pred_polylines, true_polylines, tolerance_vox) -> dict`` with
``"precision"``, ``"recall"``, ``"f1"``, length-weighted (precision = fraction of predicted
length within ``tolerance_vox`` of a true centreline, recall = fraction of true length within
``tolerance_vox`` of a predicted one). Polylines ``[K, 3]`` ``x, y, z`` voxels.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from gate6 import g6_helpers as H

REPO = Path(__file__).resolve().parents[2]


def _scores():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from benchmarks.filaments import centerline_scores

    return centerline_scores


def test_G6_bench_centerline_scores():
    centerline_scores = _scores()
    truth = [H.line_xyz((10.0, 20.0, 30.0), (210.0, 20.0, 30.0), n=101),   # 200 voxels
             H.line_xyz((10.0, 80.0, 30.0), (10.0, 180.0, 30.0), n=51)]    # 100 voxels
    tol = 1.0

    s = centerline_scores([t.copy() for t in truth], truth, tol)
    for k in ("precision", "recall", "f1"):
        assert s[k] == pytest.approx(1.0, abs=1e-6), (k, s)

    offset = [t + np.array([0.0, 0.0, 3.0 * tol]) for t in truth]
    s = centerline_scores(offset, truth, tol)
    for k in ("precision", "recall", "f1"):
        assert s[k] == pytest.approx(0.0, abs=1e-6), (k, s)

    # half of each true filament predicted exactly: recall 0.5 (length-weighted), precision 1
    half = [H.line_xyz((10.0, 20.0, 30.0), (110.0, 20.0, 30.0), n=51),
            H.line_xyz((10.0, 80.0, 30.0), (10.0, 130.0, 30.0), n=26)]
    s = centerline_scores(half, truth, tol)
    assert s["precision"] == pytest.approx(1.0, abs=1e-6), s
    assert s["recall"] == pytest.approx(0.5, abs=0.02), s  # the tolerance band adds <= 2 x 1 voxel of 300
    assert s["f1"] == pytest.approx(2 * 0.5 / 1.5, abs=0.02), s
