"""G5-C5 (scale pair geometry), G5-C6 (corresponding points), G5-C11 (clean target),
G5-C4 (masked input voxels are exactly 0).

Contract assumed here (README.md): ``make_scale_pair(source, centre, s_a, s_b, apix_ref, rng)``
takes a numpy ``(Z, Y, X)`` source and a ``(z, y, x)`` centre in source voxels; a view covers
``64 * s`` source voxels around the centre, so one view voxel is ``s`` source voxels.
``corresponding_points`` returns two ``[n, 3]`` arrays (or tensors) of ``(z, y, x)`` positions in
**view voxel coordinates** (voxel ``i`` of a view is at coordinate ``i``), after the flips.
``make_training_pair`` takes ``source``, ``apix_ref``, ``ssl_cfg`` and ``rng`` by keyword.
"""

import math

import numpy as np
import pytest

from gate5 import g5_helpers as H

SRC_SHAPE = (160, 160, 160)
CENTRE = (80.0, 77.0, 83.0)


def _scale_pair(src, s_a, s_b, seed=7):
    from tomocpt.networks.ssl.views import make_scale_pair

    np.random.seed(seed)
    return make_scale_pair(src, CENTRE, s_a, s_b, H.APIX_REF, np.random.default_rng(seed))


# --------------------------------------------------------------------------------------
# G5-C5
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("s_a,s_b", [(1.0, 1.6), (1.45, 1.0), (1.3, 1.9)])
def test_G5_C5_target_and_blob_at_the_view_centre(s_a, s_b):
    src = H.gaussian_blob_volume(SRC_SHAPE, CENTRE, sigma=2.0)
    view_a, view_b, target, _maps = _scale_pair(src, s_a, s_b)
    assert H.to_scalar(target) == pytest.approx(math.log2(s_a / s_b), abs=1e-6)
    ca, cb = H.blob_centroid(view_a), H.blob_centroid(view_b)
    # "View centre": the voxel the source centre maps to. Either convention is accepted:
    # 32 (= 64 // 2, voxel 0 at the crop origin) or 31.5 (geometric centre of 0..63).
    for c in (ca, cb):
        assert np.all((c >= 31.0 - 1e-6) & (c <= 32.5 + 1e-6)), (ca, cb)
    assert np.max(np.abs(ca - cb)) <= 0.5, (ca, cb)


# --------------------------------------------------------------------------------------
# G5-C6
# --------------------------------------------------------------------------------------

#: Marker positions (source voxels, z y x) relative to CENTRE, inside both views.
MARKERS = np.array([[-12, -11, -13], [12, -12, 10], [-11, 13, 12], [13, 12, -12],
                    [0, 0, 0], [6, -5, 9], [-7, 8, -4], [10, 3, 0]], dtype=np.float64)


def _augment(view, seed):
    from tomocpt.networks.ssl.views import wedge_preserving_augment

    v = H.to_np(view).astype(np.float32)
    out, _tgt, flips = wedge_preserving_augment(v.copy(), v.copy(), np.random.default_rng(seed))
    return H.flip_axes_of(out, v), flips


def _pick_flips(view, avoid=None):
    """First augmentation seed whose flips are not empty (and differ from ``avoid``)."""
    for seed in range(200):
        axes, flips = _augment(view, seed)
        assert axes is not None
        if axes and axes != avoid:
            return axes, flips
    raise AssertionError("wedge_preserving_augment never flipped")


def _fit_affine(P, Q):
    """Least-squares ``Q = P @ A.T + t``; returns ``(A, t, max residual)``."""
    X = np.hstack([P, np.ones((len(P), 1))])
    sol, *_ = np.linalg.lstsq(X, Q, rcond=None)
    A, t = sol[:3].T, sol[3]
    return A, t, float(np.abs(X @ sol - Q).max())


@pytest.mark.parametrize("s_a,s_b", [(1.0, 1.7), (1.5, 1.15)])
def test_G5_C6_corresponding_points_land_on_the_same_marker(s_a, s_b):
    from tomocpt.networks.ssl.views import corresponding_points

    P = np.asarray(CENTRE) + MARKERS
    views = []
    maps = None
    for k, p in enumerate(P):
        src = H.gaussian_blob_volume(SRC_SHAPE, p, sigma=1.5, seed=k)
        va, vb, _t, m = _scale_pair(src, s_a, s_b)
        maps = m if maps is None else maps  # the maps do not depend on the volume content
        views.append((H.to_np(va), H.to_np(vb)))

    axes_a, flips_a = _pick_flips(views[0][0])
    axes_b, flips_b = _pick_flips(views[0][1], avoid=axes_a)
    Qa = np.array([H.blob_centroid(np.flip(va, axes_a)) for va, _ in views])
    Qb = np.array([H.blob_centroid(np.flip(vb, axes_b)) for _, vb in views])

    # Each view is an affine image of the source: scale 1/s, flipped axes negated.
    for Q, s, axes in ((Qa, s_a, axes_a), (Qb, s_b, axes_b)):
        A, _t, res = _fit_affine(P, Q)
        assert res < 0.3, f"marker positions are not an affine image of the source (residual {res})"
        want = np.diag([(-1.0 if a in axes else 1.0) / s for a in range(3)])
        np.testing.assert_allclose(A, want, atol=0.03 / s)

    Aa, ta, _ = _fit_affine(P, Qa)
    Ab, tb, _ = _fit_affine(P, Qb)
    pts_a, pts_b = corresponding_points(128, maps, flips_a, flips_b, np.random.default_rng(3))
    pts_a, pts_b = H.points_np(pts_a), H.points_np(pts_b)
    assert pts_a.shape == pts_b.shape == (128, 3)
    for pts in (pts_a, pts_b):
        assert np.all((pts >= -0.5) & (pts <= 63.5)), "anchors must lie inside both views"
    src_pts = np.linalg.solve(Aa, (pts_a - ta).T).T
    pred_b = src_pts @ Ab.T + tb
    err = np.abs(pred_b - pts_b).max(axis=1)
    assert err.max() <= 0.5, f"{int((err > 0.5).sum())} of 128 anchor pairs do not correspond (max {err.max():.2f} voxel)"


# --------------------------------------------------------------------------------------
# G5-C11 and the make_training_pair contract
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def source():
    return H.make_source(11)


def _pair(source, seed=3, **kw):
    kw.setdefault("n_anchors", 32)
    return H.training_pair(source, H.ssl_cfg(version=2, **kw), seed)


def test_G5_T52_make_training_pair_keys_and_shapes(source):
    out = _pair(source, degradation_p=0.8)
    assert set(H.PAIR_KEYS) <= set(out), set(H.PAIR_KEYS) - set(out)
    for k in ("input_a", "target_a", "mask_a", "input_b", "target_b", "mask_b"):
        assert H.to_np(out[k]).shape == (64, 64, 64), k
    for k in ("pts_a", "pts_b"):
        assert H.points_np(out[k]).shape == (32, 3), k
    t = H.to_scalar(out["target_log2"])
    a, b = H.to_scalar(out["apix_a"]), H.to_scalar(out["apix_b"])
    assert abs(t) <= 1.0 + 1e-6  # scale_range_log2 = 1
    assert math.log2(a / b) == pytest.approx(t, abs=1e-5)
    for apix in (a, b):
        assert H.APIX_REF - 1e-5 <= apix <= 2 * H.APIX_REF + 1e-5


def test_G5_C11_target_is_the_clean_view(source):
    clean = _pair(source, degradation_p=0.0, inversion_p=0.0)
    deg = _pair(source, degradation_p=1.0, inversion_p=0.0)
    for k in ("target_a", "target_b"):
        np.testing.assert_array_equal(H.to_np(deg[k]), H.to_np(clean[k]), err_msg=k)
    # The input is degraded: it differs from the target outside the mask.
    visible = ~H.to_np(deg["mask_a"]).astype(bool)
    diff = np.abs(H.to_np(deg["input_a"]) - H.to_np(deg["target_a"]))[visible]
    assert np.mean(diff > 1e-4) > 0.5


def test_G5_C11_inversion_applies_to_view_and_target(source):
    plain = _pair(source, degradation_p=0.0, inversion_p=0.0)
    inv = _pair(source, degradation_p=0.0, inversion_p=1.0)
    for k in ("target_a", "target_b"):
        r = np.corrcoef(H.to_np(inv[k]).ravel(), H.to_np(plain[k]).ravel())[0, 1]
        assert r < -0.99, (k, r)
    vis = ~H.to_np(inv["mask_a"]).astype(bool)
    r = np.corrcoef(H.to_np(inv["input_a"])[vis], H.to_np(inv["target_a"])[vis])[0, 1]
    assert r > 0.5, "the input and its target must have the same polarity"


# --------------------------------------------------------------------------------------
# G5-C4 (input side)
# --------------------------------------------------------------------------------------

def test_G5_C4_masked_input_voxels_are_exactly_zero(source):
    for seed in (3, 4):
        out = _pair(source, seed=seed, degradation_p=1.0)
        for v in ("a", "b"):
            m = H.to_np(out[f"mask_{v}"]).astype(bool)
            assert 0.05 < m.mean() < 0.5, m.mean()  # mask ratio U(0.15, 0.4), blocks may overlap
            x = H.to_np(out[f"input_{v}"])
            assert np.all(x[m] == 0.0)
