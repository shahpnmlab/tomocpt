"""G6-C2 (``mask_to_centerlines``), G6-C3 (``render_tubes``), G6-C4 (``trace_filaments``,
``resample_polyline``), G6-C10 (``fit_filament_spline``), G6-C11 (RELION angle convention),
G6-C13 (``centerlines_to_mask``).

Truths are analytic curves; distances are exact distances from voxel centres to a polyline
(``g6_helpers``). Polylines are ``x, y, z`` native voxels; volumes and segments ``z, y, x``.
"""

import math

import numpy as np
import pytest

from gate6 import g6_helpers as H


# ------------------------------------------------------------------------------------------
# shared geometry
# ------------------------------------------------------------------------------------------

#: 250 A tube at 20 A/px: R = 6.25 voxels.
C2_APIX, C2_DIAM = 20.0, 250.0
C2_R = C2_DIAM / 2.0 / C2_APIX
C2_SHAPE = (40, 112, 256)


def c2_arc():
    """Arc of radius 170 voxels (3400 A) in the plane z = 20, 178 voxels (3560 A) long."""
    return H.arc_xyz((128.0, -100.0), 170.0, math.radians(60.0), math.radians(120.0), z=20.0, n=401)


def _check_recovered(poly, truth, mean_tol, what):
    poly = np.asarray(poly, float)
    assert poly.ndim == 2 and poly.shape[1] == 3 and len(poly) >= 2, (what, poly.shape)
    md = H.mean_dist_to_curve(poly, truth)
    assert md <= mean_tol, f"{what}: mean distance to the true centreline {md:.3f} voxel (tolerance {mean_tol})"


# ------------------------------------------------------------------------------------------
# G6-C2 mask_to_centerlines
# ------------------------------------------------------------------------------------------

def test_G6_C2_curved_tube_centreline_and_length():
    from tomocpt.labels.filaments import mask_to_centerlines

    truth = c2_arc()
    mask = open_tube_mask(C2_SHAPE, [truth], C2_R, 0.0)
    polys = mask_to_centerlines(mask, C2_APIX, C2_DIAM, 200.0)
    assert len(polys) == 1, f"one tube must give one polyline, got {len(polys)}"
    _check_recovered(polys[0], truth, 1.0, "curved tube")
    L_true, L = H.polyline_length(truth), H.polyline_length(polys[0])
    assert abs(L - L_true) <= 0.05 * L_true, f"length {L:.1f} vs true {L_true:.1f} voxels (5 %)"
    assert H.coverage(truth, polys[0], tol=1.5, trim=C2_R) >= 0.95


def test_G6_C2_uint8_mask_is_accepted():
    from tomocpt.labels.filaments import mask_to_centerlines

    truth = c2_arc()
    mask = open_tube_mask(C2_SHAPE, [truth], C2_R, 0.0).astype(np.uint8)
    polys = mask_to_centerlines(mask, C2_APIX, C2_DIAM, 200.0)
    assert len(polys) == 1
    _check_recovered(polys[0], truth, 1.0, "uint8 mask")


def test_G6_C2_two_crossing_tubes_give_two_polylines():
    """Two straight tubes crossing at 90 degrees in the plane z = 20."""
    from tomocpt.labels.filaments import mask_to_centerlines

    a = H.line_xyz((30.0, 56.0, 20.0), (220.0, 56.0, 20.0), n=201)
    b = H.line_xyz((128.0, 8.0, 20.0), (128.0, 104.0, 20.0), n=201)
    mask = open_tube_mask(C2_SHAPE, [a, b], C2_R, 0.0)
    polys = mask_to_centerlines(mask, C2_APIX, C2_DIAM, 200.0)
    assert len(polys) == 2, f"two crossing tubes must give two polylines, got {len(polys)}"
    m = H.match_polylines(polys, [a, b])
    assert sorted(m) == [0, 1], f"both polylines follow the same tube: {m}"
    for poly, k in zip(polys, m):
        truth = (a, b)[k]
        # each polyline runs through the crossing: >= 90 % of its tube (ends trimmed) within 2 voxels
        cov = H.coverage(truth, poly, tol=2.0, trim=C2_R + 2.0)
        assert cov >= 0.9, f"tube {k}: coverage {cov:.2f}"


def test_G6_C2_gap_shorter_than_half_the_diameter_gives_one_polyline():
    """The curved tube with a gap of 0.4 x diameter (5 voxels) cut across it in the middle."""
    from tomocpt.labels.filaments import mask_to_centerlines

    truth = c2_arc()
    mask = open_tube_mask(C2_SHAPE, [truth], C2_R, 0.0)
    s = H.cumulative_length(truth)
    gap = 0.4 * C2_DIAM / C2_APIX
    zz, yy, xx = np.nonzero(mask)
    pts = np.stack([xx, yy, zz], 1).astype(float)
    from scipy.spatial import cKDTree

    _, idx = cKDTree(truth).query(pts)
    mid = s[-1] / 2.0
    cut = np.abs(s[idx] - mid) <= gap / 2.0
    mask[zz[cut], yy[cut], xx[cut]] = False
    assert cut.sum() > 0
    polys = mask_to_centerlines(mask, C2_APIX, C2_DIAM, 200.0)
    assert len(polys) == 1, f"a gap of {gap:.1f} voxels (< half the diameter) must be bridged, got {len(polys)}"
    assert H.coverage(truth, polys[0], tol=1.5, trim=C2_R) >= 0.9


#: hollow 250 A tube (human decision, 31 Oct 2026): wall 50 A, lumen 150 A (DS-10490 masks are walls)
C2_R_IN = C2_R - 50.0 / C2_APIX


def open_tube_mask(shape, polylines, r_out, r_in):
    """Tube mask with **flat open ends** (no caps; as a tube cut by the tomogram border or
    ending in a segmentation): voxels at distance ``r_in <= d <= r_out`` from a (densely
    sampled) centreline whose nearest centreline point is not an end point. ``r_in = 0`` gives a
    solid tube; ``r_in > 0`` a hollow wall with a lumen of radius ``r_in``, open at both ends.
    The true centreline runs from end face to end face, so its length is the polyline length."""
    from scipy.spatial import cKDTree

    mask = np.zeros(shape, dtype=bool)
    for poly in polylines:
        dense = H.densify(poly, 0.25)
        solid = H.filled_tube_mask(shape, [dense], r_out)
        zz, yy, xx = np.nonzero(solid)
        d, idx = cKDTree(dense).query(np.stack([xx, yy, zz], 1).astype(float))
        keep = (d >= r_in) & (idx > 0) & (idx < len(dense) - 1)
        mask[zz[keep], yy[keep], xx[keep]] = True
    return mask


def test_G6_C2_hollow_curved_tube_centreline_and_length():
    """The curved tube as a hollow wall mask (outer 250 A, inner 150 A, open ends): same
    thresholds as the solid case. Margin with the scratch reference (closing at the tube radius):
    see README."""
    from tomocpt.labels.filaments import mask_to_centerlines

    truth = c2_arc()
    mask = open_tube_mask(C2_SHAPE, [truth], C2_R, C2_R_IN)
    filled = H.filled_tube_mask(C2_SHAPE, [truth], C2_R)
    assert (filled & ~mask).sum() > 0.25 * filled.sum(), "the mask must have a lumen"
    polys = mask_to_centerlines(mask, C2_APIX, C2_DIAM, 200.0)
    assert len(polys) == 1, f"one hollow tube must give one polyline, got {len(polys)}"
    _check_recovered(polys[0], truth, 1.0, "hollow curved tube")
    L_true, L = H.polyline_length(truth), H.polyline_length(polys[0])
    assert abs(L - L_true) <= 0.05 * L_true, f"length {L:.1f} vs true {L_true:.1f} voxels (5 %)"
    assert H.coverage(truth, polys[0], tol=1.5, trim=C2_R) >= 0.95


def test_G6_C2_hollow_crossing_tubes_give_two_polylines():
    from tomocpt.labels.filaments import mask_to_centerlines

    a = H.line_xyz((30.0, 56.0, 20.0), (220.0, 56.0, 20.0), n=201)
    b = H.line_xyz((128.0, 8.0, 20.0), (128.0, 104.0, 20.0), n=201)
    mask = open_tube_mask(C2_SHAPE, [a, b], C2_R, C2_R_IN)
    polys = mask_to_centerlines(mask, C2_APIX, C2_DIAM, 200.0)
    assert len(polys) == 2, f"two crossing hollow tubes must give two polylines, got {len(polys)}"
    m = H.match_polylines(polys, [a, b])
    assert sorted(m) == [0, 1], f"both polylines follow the same tube: {m}"
    for poly, k in zip(polys, m):
        truth = (a, b)[k]
        assert H.mean_dist_to_curve(poly, truth) <= 1.0
        cov = H.coverage(truth, poly, tol=2.0, trim=C2_R + 2.0)
        assert cov >= 0.9, f"tube {k}: coverage {cov:.2f}"


# ------------------------------------------------------------------------------------------
# G6-C3 render_tubes
# ------------------------------------------------------------------------------------------

def _check_profile(target, dist, R, what):
    inner = dist < R - 1
    assert inner.sum() > 100, what
    err = np.abs(target[inner] - H.profile(dist[inner], R))
    assert err.max() <= 0.02, f"{what}: profile error {err.max():.4f} for r < R - 1"
    outer = dist > R + 1
    assert np.all(target[outer] == 0), f"{what}: non-zero target at r > R + 1 (max {target[outer].max()})"


def test_G6_C3_reference_profile_is_generate_gaussian_sphere():
    """The test's profile equals ``generate_gaussian_sphere`` (CLAUDE.md 1.3) at integer R."""
    from tomocpt.labels.helpers import generate_gaussian_sphere

    for R in (5, 10, 12):
        g = generate_gaussian_sphere(R, 2 * R + 1)
        c = np.arange(-R, R + 1, dtype=float)
        zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
        r = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
        np.testing.assert_allclose(H.profile(r, R), g, atol=1e-9)


#: straight tube with irregular segment joints along X
JOINTS_X = (10.0, 13.7, 21.2, 22.0, 35.55, 50.1, 50.9, 70.33, 86.0)


def test_G6_C3_straight_tubes_profile_centreline_and_no_ripple():
    from tomocpt.labels.filaments import render_tubes

    shape, R = (48, 64, 96), 10.0
    on_grid = np.array([[x, 32.0, 24.0] for x in JOINTS_X])        # through voxel centres
    off_grid = np.array([[x, 30.3, 23.6] for x in JOINTS_X])       # 0.5 voxel off the row (z=24, y=30)
    seg = np.concatenate([H.segments_from_polyline(on_grid, 0, 2 * R),
                          H.segments_from_polyline(off_grid, 1, 2 * R)])
    out = render_tubes(seg, shape, 2)
    assert out.shape == (2,) + shape and out.dtype == np.float32
    for c, poly in enumerate((on_grid, off_grid)):
        d = H.distance_field(shape, [poly], reach=R + 3)
        _check_profile(out[c], d, R, f"straight tube {c}")
    xs = np.arange(10, 87)
    row = out[0, 24, 32, xs]
    assert np.all(np.abs(row - 1.0) <= 0.02), f"centreline value {row.min():.4f}..{row.max():.4f}"
    assert row.max() - row.min() < 0.02, "ripple along the centreline"
    row = out[1, 24, 30, xs]  # constant distance 0.5 from the off-grid line
    assert row.max() - row.min() < 0.02, f"ripple along the off-grid tube {row.max() - row.min():.4f}"


def test_G6_C3_helical_tube_profile():
    from tomocpt.labels.filaments import render_tubes

    shape, R = (40, 40, 112), 5.0
    dense = H.helix_xyz(12.0, 48.0, 0.0, 4 * np.pi, origin=(8.0, 20.0, 20.0))
    poly = H.resample_by_length(dense, 2.0)  # coarse joints every 2 voxels
    out = render_tubes(H.segments_from_polyline(poly, 0, 2 * R), shape, 1)
    d = H.distance_field(shape, [poly], reach=R + 3)
    _check_profile(out[0], d, R, "helical tube")
    near = d < 0.1
    if near.any():
        assert np.all(np.abs(out[0][near] - 1.0) <= 0.02)


def test_G6_C3_crossing_tubes_max_combine_and_round_caps():
    from tomocpt.labels.filaments import render_tubes

    shape, R = (32, 64, 64), 6.0
    a = np.array([[8.0, 20.0, 16.0], [40.0, 30.0, 16.0], [56.0, 44.0, 15.0]])
    b = np.array([[30.0, 6.0, 16.5], [30.0, 58.0, 16.5]])
    seg = np.concatenate([H.segments_from_polyline(a, 0, 2 * R), H.segments_from_polyline(b, 0, 2 * R)])
    out = render_tubes(seg, shape, 1)
    d = H.distance_field(shape, [a, b], reach=R + 3)  # union = max of the profiles
    _check_profile(out[0], d, R, "crossing tubes")


def test_G6_C3_empty_segments():
    from tomocpt.labels.filaments import render_tubes

    out = render_tubes(np.zeros((0, 8), np.float32), (16, 16, 16), 3)
    assert out.shape == (3, 16, 16, 16) and not out.any()


# ------------------------------------------------------------------------------------------
# G6-C4 trace_filaments, resample_polyline
# ------------------------------------------------------------------------------------------

def test_G6_C4_trace_recovers_centreline_and_drops_short():
    from tomocpt.labels.filaments import trace_filaments

    long_ = c2_arc()
    short = H.line_xyz((200.0, 95.0, 20.0), (202.0, 95.0, 20.0))  # 40 A core: < 200 A after tracing
    prob = np.maximum(H.tube_target(C2_SHAPE, [long_], C2_R), H.tube_target(C2_SHAPE, [short], C2_R))
    polys = trace_filaments(prob, 0.3, C2_APIX, C2_DIAM, 200.0, 40.0)
    assert len(polys) == 1, f"expected the long filament only, got {len(polys)} polylines"
    _check_recovered(polys[0], long_, 0.5, "traced filament")
    L_true = H.polyline_length(long_)
    assert H.polyline_length(polys[0]) >= 0.85 * L_true
    assert H.mean_dist_to_curve(polys[0], short) > 20.0


def test_G6_C4_min_length_is_applied():
    """The same map with min_length_ang above the long filament's length gives nothing."""
    from tomocpt.labels.filaments import trace_filaments

    long_ = c2_arc()
    prob = H.tube_target(C2_SHAPE, [long_], C2_R)
    L_ang = H.polyline_length(long_) * C2_APIX
    assert trace_filaments(prob, 0.3, C2_APIX, C2_DIAM, 200.0, 40.0)
    assert len(trace_filaments(prob, 0.3, C2_APIX, C2_DIAM, 1.5 * L_ang, 40.0)) == 0


@pytest.mark.parametrize("spacing", [3.0, 4.7])
def test_G6_C4_resample_polyline_spacing(spacing):
    from tomocpt.labels.filaments import resample_polyline

    dense = H.helix_xyz(12.0, 48.0, 0.0, 4 * np.pi, origin=(8.0, 20.0, 20.0), n=8001)
    out = np.asarray(resample_polyline(dense, spacing), float)
    assert out.ndim == 2 and out.shape[1] == 3
    step = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.all(np.abs(step[:-1] / spacing - 1.0) < 0.01), f"spacing {step.min():.4f}..{step.max():.4f}"
    assert step[-1] <= spacing * 1.01
    np.testing.assert_allclose(out[0], dense[0], atol=1e-6)
    assert H.max_dist_to_curve(out, dense) < 0.01


def test_G6_C4_resample_polyline_irregular_straight_line():
    from tomocpt.labels.filaments import resample_polyline

    poly = np.array([[x, 2.0 + 0.5 * x, 7.0 - 0.25 * x] for x in JOINTS_X])
    out = np.asarray(resample_polyline(poly, 2.5), float)
    step = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.all(np.abs(step[:-1] / 2.5 - 1.0) < 0.01)
    assert H.max_dist_to_curve(out, poly) < 1e-6
    L = H.polyline_length(poly)
    assert len(out) in (int(L // 2.5) + 1, int(L // 2.5) + 2)


# ------------------------------------------------------------------------------------------
# G6-C10 fit_filament_spline
# ------------------------------------------------------------------------------------------

HELIX_R_ANG, HELIX_PITCH_ANG, HELIX_LEN_ANG = 300.0, 1000.0, 1692.0


def helix_ang(n=200001):
    """Helix of radius 300 A (axis X, pitch 1000 A), 1692 A long, in A."""
    per_rad = math.hypot(HELIX_PITCH_ANG / (2 * np.pi), HELIX_R_ANG)
    return H.helix_xyz(HELIX_R_ANG, HELIX_PITCH_ANG, 0.0, HELIX_LEN_ANG / per_rad,
                       origin=(100.0, 500.0, 500.0), n=n)


def _check_fit_output(pts, tng, arc, spacing):
    pts, tng, arc = (np.asarray(a, float) for a in (pts, tng, arc))
    K = len(pts)
    assert K >= 3 and pts.shape == (K, 3) and tng.shape == (K, 3) and arc.shape == (K,)
    np.testing.assert_allclose(np.linalg.norm(tng, axis=1), 1.0, atol=1e-6)
    assert arc[0] == pytest.approx(0.0, abs=1e-6)
    da = np.diff(arc)
    assert np.all(np.abs(da[:-1] / spacing - 1.0) < 0.01), f"arc spacing {da.min():.3f}..{da.max():.3f}"
    assert 0 < da[-1] <= spacing * 1.01
    dots = (tng[1:] * tng[:-1]).sum(1)
    assert np.all(dots > 0), "tangent sign flips along the filament"
    return pts, tng, arc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_G6_C10_noisy_helix(seed):
    """6.8 A/px, samples every voxel with N(0, 1 voxel) noise per coordinate; smoothing = the
    noise RMS (sqrt(3) voxels in A). Margin: a reference splprep fit gives mean 0.32-0.39 voxel
    over 10 seeds (threshold 0.5).

    Why this smoothing (accepted by the orchestrator): FITPACK drives the residual to
    ``s = n x smoothing^2``, so a smoothing below the noise RMS fits the noise and one above it
    cuts into the 300 A helix (1.2 x the noise RMS already gives ~0.7 voxel). The 0.5-voxel
    criterion holds only with the smoothing matched to the noise."""
    from tomocpt.labels.filaments import fit_filament_spline

    apix, spacing = 6.8, 40.0
    truth = helix_ang()
    samples = H.resample_by_length(truth / apix, 1.0)
    noisy = samples + np.random.default_rng(seed).standard_normal(samples.shape)
    pts, tng, arc = _check_fit_output(*fit_filament_spline(noisy, apix, math.sqrt(3.0) * apix, spacing), spacing)
    md = H.mean_dist_to_curve(pts * apix, truth) / apix
    assert md <= 0.5, f"mean distance {md:.3f} voxel"
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1) * apix
    assert np.all(np.abs(chord[:-1] / spacing - 1.0) < 0.01)
    fd = pts[2:] - pts[:-2]
    fd /= np.linalg.norm(fd, axis=1, keepdims=True)
    assert np.all(np.abs((fd * tng[1:-1]).sum(1)) > 0.95), "tangents do not follow the curve"


def test_G6_C10_same_curve_at_two_voxel_sizes():
    """Noise-free samples (one per voxel) at 6.8 and 13.6 A/px, smoothing 2 A: same count and
    the same points in A within 0.5 x 13.6 A (a reference splprep fit: 43 points both, max
    difference 0.3 A; L / spacing = 42.5, far from an integer).

    Why 2 A (accepted by the orchestrator): with the default 10 A the fitted length grows with the
    smoothing and differs between the two samplings (45 vs 44 points in a reference fit), so the
    count criterion is only meaningful with light smoothing."""
    from tomocpt.labels.filaments import fit_filament_spline

    truth, spacing = helix_ang(), 40.0
    res = {}
    for apix in (6.8, 13.6):
        poly = H.resample_by_length(truth / apix, 1.0)
        res[apix] = _check_fit_output(*fit_filament_spline(poly, apix, 2.0, spacing), spacing)
    a, b = res[6.8][0] * 6.8, res[13.6][0] * 13.6
    assert len(a) == len(b), f"{len(a)} vs {len(b)} points"
    d = np.linalg.norm(a - b, axis=1)
    assert d.max() <= 0.5 * 13.6, f"max difference {d.max():.2f} A"
    np.testing.assert_allclose(res[6.8][2][:-1], res[13.6][2][:-1], atol=1e-3)  # the last point may be the end
    assert abs(res[6.8][2][-1] - res[13.6][2][-1]) <= 0.5 * 13.6


@pytest.mark.parametrize("n", [2, 3])
def test_G6_C10_few_points(n):
    """Fewer than 4 points (k = min(3, n - 1)): a straight 200 A line sampled every 40 A."""
    from tomocpt.labels.filaments import fit_filament_spline

    poly = H.line_xyz((5.0, 6.0, 7.0), (5.0 + 12.0, 6.0 + 16.0, 7.0), n=n)  # 20 voxels = 200 A at 10 A/px
    pts, tng, arc = (np.asarray(a, float) for a in fit_filament_spline(poly, 10.0, 10.0, 40.0))
    assert len(pts) in (6, 7)
    assert H.max_dist_to_curve(pts, poly) < 0.05
    np.testing.assert_allclose(np.abs(tng @ np.array([0.6, 0.8, 0.0])), 1.0, atol=1e-3)


# ------------------------------------------------------------------------------------------
# G6-C11 RELION angle convention
# ------------------------------------------------------------------------------------------

def test_G6_C11_reference_matrix_is_self_consistent():
    """The test's own reference: for the obviously-correct (tilt, psi) the RELION matrix maps Z
    to d and its transpose maps d to Z; the matrix is a rotation; rot does not move the axis."""
    rng = np.random.default_rng(1)
    ez = np.array([0.0, 0.0, 1.0])
    for _ in range(200):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        A = H.relion_euler_matrix(0.0, *H.reference_tilt_psi(d))
        np.testing.assert_allclose(A @ A.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(A) == pytest.approx(1.0)
        np.testing.assert_allclose(A @ ez, d, atol=1e-12)
        np.testing.assert_allclose(A.T @ d, ez, atol=1e-12)
        np.testing.assert_allclose(H.relion_euler_matrix(rng.uniform(0, 360), *H.reference_tilt_psi(d)) @ ez,
                                   d, atol=1e-12)


def test_G6_C11_random_directions_map_to_the_filament_axis():
    from tomocpt.labels.filaments import tangents_to_relion_angles

    rng = np.random.default_rng(0)
    d = rng.standard_normal((500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tilt, psi = (np.asarray(a, float) for a in tangents_to_relion_angles(d))
    assert tilt.shape == (500,) and psi.shape == (500,)
    ez = np.array([0.0, 0.0, 1.0])
    for k in range(len(d)):
        A = H.relion_euler_matrix(0.0, tilt[k], psi[k])
        v = A @ ez
        s = 1.0 if np.dot(v, d[k]) >= 0 else -1.0
        np.testing.assert_allclose(v, s * d[k], atol=1e-6, err_msg=f"direction {d[k]}")
        np.testing.assert_allclose(A.T @ (s * d[k]), ez, atol=1e-6)


def test_G6_C11_landmarks():
    from tomocpt.labels.filaments import tangents_to_relion_angles

    tilt, _ = tangents_to_relion_angles(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    for t in np.asarray(tilt, float):
        assert min(abs(t), abs(abs(t) - 180.0)) < 1e-6, f"filament along Z: tilt {t}"
    ang = np.radians(np.linspace(0, 360, 37))
    xy = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
    tilt, _ = tangents_to_relion_angles(xy)
    np.testing.assert_allclose(np.asarray(tilt, float), 90.0, atol=1e-6)


# ------------------------------------------------------------------------------------------
# G6-C13 centerlines_to_mask
# ------------------------------------------------------------------------------------------

def test_G6_C13_centerlines_to_mask_radius():
    from tomocpt.labels.filaments import centerlines_to_mask

    shape = (40, 112, 256)
    polys = [c2_arc(), H.line_xyz((30.0, 95.0, 18.0), (220.0, 100.0, 22.0), n=40)]
    mask = centerlines_to_mask(polys, shape, C2_APIX, C2_DIAM)
    assert isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and mask.shape == shape
    assert set(np.unique(mask).tolist()) <= {0, 1}
    d = H.distance_field(shape, polys, reach=C2_R + 3)
    assert np.all(mask[d <= C2_R - 1] == 1), "voxels within R - 1 of the centreline are not in the mask"
    assert np.all(mask[d > C2_R + 1] == 0), "voxels beyond R + 1 are in the mask"
    assert mask[d <= C2_R - 1].size > 1000
