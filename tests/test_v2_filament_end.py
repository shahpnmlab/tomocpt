"""Traced filaments have no near-duplicate end point (``merge_short_end``); labels keep the legacy sampling."""
import numpy as np
import pytest

from tomocpt.labels.filaments import _trace_and_fit, fit_filament_spline

APIX, SPACING = 10.0, 40.0


def _line(length_ang):
    n = 50
    t = np.linspace(0.0, length_ang / APIX, n)
    return np.stack([10 + 0.6 * t, 10 + 0.8 * t, np.full(n, 20.0)], axis=1)  # x, y, z voxels


def _gaps(arc):
    return np.diff(np.asarray(arc, float))


@pytest.mark.parametrize("tail", [1.0, 5.0, 19.9])
def test_short_tail_is_merged(tail):
    L = 10 * SPACING + tail
    pts, tng, arc = fit_filament_spline(_line(L), APIX, 0.0, SPACING, merge_short_end=True)
    g = _gaps(arc)
    assert arc[-1] == pytest.approx(L, abs=0.5)  # still reaches the end
    np.testing.assert_allclose(g[:-1], SPACING, rtol=1e-3)
    assert SPACING * 0.999 <= g[-1] < 1.5 * SPACING
    assert len(pts) == len(tng) == len(arc) == 11
    end = _line(L)[-1]
    assert np.linalg.norm(pts[-1] - end) < 0.05


@pytest.mark.parametrize("tail", [20.1, 30.0])
def test_long_tail_is_kept(tail):
    L = 10 * SPACING + tail
    _, _, arc = fit_filament_spline(_line(L), APIX, 0.0, SPACING, merge_short_end=True)
    g = _gaps(arc)
    assert len(arc) == 12
    assert 0.5 * SPACING <= g[-1] < SPACING


def test_default_keeps_legacy_end_point():
    L = 10 * SPACING + 1.0
    _, _, arc = fit_filament_spline(_line(L), APIX, 0.0, SPACING)
    assert len(arc) == 12 and _gaps(arc)[-1] == pytest.approx(1.0, abs=0.5)


def test_short_filament_keeps_both_ends():
    _, _, arc = fit_filament_spline(_line(15.0), APIX, 0.0, SPACING, merge_short_end=True)
    assert len(arc) == 2 and arc[-1] == pytest.approx(15.0, abs=0.5)


def test_traced_filaments_have_no_short_end_gap():
    shape = (40, 80, 80)
    zz, yy, xx = np.indices(shape, dtype=np.float64)
    prob = np.zeros(shape, np.float32)
    for y0, z0, x1 in ((20.0, 20.0, 70.3), (50.0, 15.0, 61.7)):  # two straight tubes along x
        r = np.hypot(yy - y0, zz - z0)
        prob = np.maximum(prob, np.where((xx >= 5) & (xx <= x1), np.exp(-r ** 2 / 4.0), 0.0).astype(np.float32))
    fits = _trace_and_fit(prob, 0.5, 6.8, 70.0, 100.0, SPACING, 10.0)
    assert len(fits) == 2
    for pts, _, arc in fits:
        g = _gaps(arc)
        assert g.min() >= 0.5 * SPACING - 1e-6
        assert g.max() < 1.5 * SPACING
