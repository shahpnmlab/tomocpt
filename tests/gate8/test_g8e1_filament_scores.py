"""Gate 8e.1, G8e-C3: filament threshold scores are identical whether the thresholds are traced in
one process or in parallel worker processes (``filament_threshold_scores(..., workers=k)``;
``workers=None`` = automatic, the calibration default; no config field, human decision 9 Nov 2026)."""
import numpy as np
import pytest

from gate6 import g6_helpers as H6


def _scene(seed=0, shape=(40, 96, 96)):
    rng = np.random.default_rng(seed)
    lines = [H6.line_xyz((5, 10, 20), (90, 80, 22)), H6.line_xyz((10, 85, 10), (85, 15, 30)),
             H6.helix_xyz(6.0, 40.0, 0.0, 12.0, origin=(10.0, 48.0, 20.0))]
    prob = H6.tube_target(shape, lines, R=3.0)
    prob = np.clip(prob * 0.9 + 0.15 * rng.random(shape, dtype=np.float32), 0, 1).astype(np.float32)
    return prob, [l[::50] for l in lines]


class _Infer:
    filament_min_length_ang = 50.0
    filament_spacing_ang = 20.0
    filament_smoothing_ang = 10.0


@pytest.mark.parametrize("workers", [2, 4, None])
def test_G8e_C3_parallel_scores_identical(workers):
    from tomocpt.predict.calibration import DEFAULT_GRID, filament_threshold_scores

    prob, true = _scene()
    serial = filament_threshold_scores(prob, true, 60.0, 10.0, _Infer(), DEFAULT_GRID, workers=1)
    par = filament_threshold_scores(prob, true, 60.0, 10.0, _Infer(), DEFAULT_GRID, workers=workers)
    assert serial.shape == (len(DEFAULT_GRID), serial.shape[1])
    assert np.array_equal(serial, par)
    assert serial[:, 0].sum() > 0, "the scene must produce traces"
