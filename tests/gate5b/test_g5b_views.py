"""G5b-C2: independent noise between views (``decorrelate_view_noise``) and the clean target.

Contract assumed here (README.md): ``decorrelate_view_noise(view, ssl_cfg, rng)`` takes a numpy
float32 ``(Z, Y, X)`` view, an ``SSLConfig`` and a ``numpy.random.Generator``; it returns the
decorrelated view (numpy or torch, leading singleton axes allowed) of the same shape and does not
modify its input. It applies an isotropic Fourier low-pass with cutoff
``U(view_lowpass_min, view_lowpass_max)`` x Nyquist (``|k| / 0.5 cycles per voxel``) and adds
independent white noise at ``U(view_noise_snr_min, view_noise_snr_max)`` relative to the view's
own std.

Measurement of "voxel-wise correlation < 0.1" on white noise (deviation from the literal
criterion, README.md): with the spec's own design the two views share the white-noise content
below the smaller cutoff, so the full-band correlation of a correct implementation is
(c_min/c_max)^1.5 * snr/(1+snr) ~ 0.2-0.45 (measured on a reference implementation: 0.20-0.42),
never < 0.1. What the design removes is the shared *high-frequency* noise (the shortcut): the
tests require correlation < 0.1 in the band above 0.65 x Nyquist (reference: |r| < 0.005; before
the fix: ~1) and < 0.7 over the full band (before the fix: ~1).
"""

import numpy as np
import pytest

from gate5 import g5_helpers as H
from gate5b import g5b_helpers as HB

HF = 0.65  # above view_lowpass_max = 0.6: only independent noise remains there


def _decorrelate(view, cfg, seed):
    from tomocpt.networks.ssl.views import decorrelate_view_noise

    out = decorrelate_view_noise(view, cfg, np.random.default_rng(seed))
    out = H.to_np(out)
    assert out.shape == view.shape
    assert np.isfinite(out).all()
    return out


def _white(seed, shape=(64, 64, 64)):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _blobs(seed, shape=(64, 64, 64), n=12, sigma=3.0):
    """Dark Gaussian blobs (structure mostly below 0.25 x Nyquist)."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij"))
    v = np.zeros(shape)
    for _ in range(n):
        c = rng.uniform(8, min(shape) - 8, 3)
        v -= np.exp(-((grid - c[:, None, None, None]) ** 2).sum(0) / (2 * sigma ** 2))
    return v.astype(np.float32)


@pytest.mark.parametrize("seed", range(5))
def test_G5b_C2_white_noise_views_decorrelate(seed):
    cfg = H.resolved(H.ssl_cfg(version=2))
    src = _white(seed)
    a = _decorrelate(src.copy(), cfg, 100 + seed)
    b = _decorrelate(src.copy(), cfg, 200 + seed)
    r_full = HB.corr(a, b)
    r_hf = HB.corr(HB.band(a, lo=HF), HB.band(b, lo=HF))
    assert abs(r_hf) < 0.1, r_hf       # reference ~0.003
    assert r_full < 0.7, r_full        # reference 0.20-0.42 (before: 1.0)


@pytest.mark.parametrize("seed", range(3))
def test_G5b_C2_structure_survives(seed):
    """Low-passed structure (<= 0.25 x Nyquist, below the smallest cutoff 0.35) correlates > 0.7
    between the two views and with the source (reference: >= 0.97)."""
    cfg = H.resolved(H.ssl_cfg(version=2))
    src = _blobs(seed)
    a = _decorrelate(src.copy(), cfg, 100 + seed)
    b = _decorrelate(src.copy(), cfg, 200 + seed)
    la, lb, ls = (HB.band(v, hi=0.25) for v in (a, b, src))
    assert HB.corr(la, lb) > 0.7
    assert HB.corr(la, ls) > 0.7
    assert HB.corr(lb, ls) > 0.7


def test_G5b_C2_cutoff_and_snr_are_the_config_values():
    """``view_lowpass_min = max = 0.4`` and a huge SNR: below the cutoff the output is the input,
    above it nothing of the input is left."""
    cfg = H.resolved(H.ssl_cfg(version=2, view_lowpass_min=0.4, view_lowpass_max=0.4,
                               view_noise_snr_min=1e4, view_noise_snr_max=1e4))
    src = _white(7)
    out = _decorrelate(src.copy(), cfg, 3)
    assert HB.corr(HB.band(out, hi=0.35), HB.band(src, hi=0.35)) > 0.99
    assert abs(HB.corr(HB.band(out, lo=0.45), HB.band(src, lo=0.45))) < 0.05


def test_G5b_C2_input_not_modified():
    cfg = H.resolved(H.ssl_cfg(version=2))
    src = _white(1)
    keep = src.copy()
    _decorrelate(src, cfg, 0)
    np.testing.assert_array_equal(src, keep)


# --------------------------------------------------------------------------------------
# inside make_training_pair
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def white_source():
    return H.robust_normalise(np.random.default_rng(5).standard_normal((128,) * 3))


@pytest.mark.parametrize("seed", range(3))
def test_G5b_C2_training_pair_inputs_decorrelated_at_s1(white_source, seed):
    """Pure white-noise source, both views at ``s = 1`` (``scale_range_log2 = 0``), no Fourier
    degradation, no inversion. The two targets are the same clean crop (up to flips; correlation
    > 0.999), so the inputs share the location voxel for voxel. Before the fix the inputs differ
    only by flips and the intensity jitter of D (correlation ~0.99); after it they must decorrelate
    (both-visible voxels; reference implementation: full band 0.20-0.32, above 0.65 x Nyquist ~0)."""
    cfg = H.ssl_cfg(version=2, n_anchors=32, degradation_p=0.0, inversion_p=0.0, scale_range_log2=0.0)
    p = H.training_pair(white_source, cfg, seed)
    axes, r_t = HB.best_flip(p["target_a"], p["target_b"])
    assert r_t > 0.999, f"targets are not the same clean view (r = {r_t})"
    f = lambda k: np.flip(H.to_np(p[k]), axes)  # noqa: E731
    ia, ib = H.to_np(p["input_a"]), f("input_b")
    vis = ~H.to_np(p["mask_a"]).astype(bool) & ~f("mask_b").astype(bool)
    assert vis.mean() > 0.2
    r_full = HB.corr(ia[vis], ib[vis])
    r_hf = HB.corr(HB.band(ia, lo=HF)[vis], HB.band(ib, lo=HF)[vis])
    assert r_full < 0.7, r_full
    assert abs(r_hf) < 0.1, r_hf


def test_G5b_C2_reconstruction_target_unchanged_bitwise():
    """The decorrelation acts on the inputs only: changing its settings leaves ``target_a`` and
    ``target_b`` bitwise identical (same seeds) and changes the inputs."""
    src = H.make_source(11)
    base = dict(n_anchors=32, degradation_p=0.8)
    p1 = H.training_pair(src, H.ssl_cfg(version=2, **base), 3)
    p2 = H.training_pair(src, H.ssl_cfg(version=2, view_lowpass_min=0.5, view_lowpass_max=0.55,
                                        view_noise_snr_min=2.0, view_noise_snr_max=3.0, **base), 3)
    for k in ("target_a", "target_b"):
        np.testing.assert_array_equal(H.to_np(p1[k]), H.to_np(p2[k]), err_msg=k)
    vis = ~H.to_np(p1["mask_a"]).astype(bool)
    assert np.mean(np.abs(H.to_np(p1["input_a"]) - H.to_np(p2["input_a"]))[vis] > 1e-4) > 0.5


def test_G5b_C1_superseded_G5_C11_polarity():
    """Replaces G5-C11's ``corr(input, target) > 0.5`` (human ruling, 31 Oct 2026): with the
    Gate 5b decorrelation at the spec defaults a correct implementation gives ~0.29. The purpose,
    input and target share polarity under inversion, is kept: r > 0.1 for both polarities, and
    inversion flips the targets (r < -0.99)."""
    src = H.make_source(11)
    plain = H.training_pair(src, H.ssl_cfg(version=2, n_anchors=32, degradation_p=0.0, inversion_p=0.0), 0)
    inv = H.training_pair(src, H.ssl_cfg(version=2, n_anchors=32, degradation_p=0.0, inversion_p=1.0), 0)
    for k in ("target_a", "target_b"):
        assert HB.corr(H.to_np(inv[k]), H.to_np(plain[k])) < -0.99, k
    for p in (plain, inv):
        for v in ("a", "b"):
            vis = ~H.to_np(p[f"mask_{v}"]).astype(bool)
            r = HB.corr(H.to_np(p[f"input_{v}"])[vis], H.to_np(p[f"target_{v}"])[vis])
            assert r > 0.1, (v, r)
