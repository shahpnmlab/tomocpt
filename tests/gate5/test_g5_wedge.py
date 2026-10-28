"""G5-C2 (cause check), G5-C3 (wedge kept), G5-C9 (noise equalisation), G5-C12 (degrade stats).

``wedge_power_fraction`` (``utils.wedge``) is the fraction of Fourier power inside the missing
region of the canonical orientation (tilt axis Y, 60 deg). Periodic volumes (whole wedged noise
volumes and their flips) are measured with it directly; views cut out of a larger volume are
measured with ``g5_helpers.view_wedge_fraction`` (window + 10 deg margin, see its docstring).
"""

import numpy as np
import pytest

from gate5 import g5_helpers as H
from utils.wedge import wedge_power_fraction

N_DRAWS = 50


# --------------------------------------------------------------------------------------
# G5-C2: cause check (pure numpy; documents why the v1 rotation task is ill-posed)
# --------------------------------------------------------------------------------------

def test_G5_C2_rot90_about_z_moves_the_wedge_rot180_and_flips_do_not():
    v = H.wedged_noise((64, 64, 64), seed=0)
    assert wedge_power_fraction(v) < 1e-6
    for k in (1, 3):  # 90 and 270 deg about Z = in the (Y, X) plane
        assert wedge_power_fraction(np.rot90(v, k, axes=(1, 2))) > 0.2
    assert wedge_power_fraction(np.rot90(v, 2, axes=(1, 2))) < 0.02
    for axis in range(3):
        assert wedge_power_fraction(np.flip(v, axis)) < 0.02
    assert wedge_power_fraction(np.flip(v, (0, 1, 2))) < 0.02


def test_G5_C2_view_metric_separates_rotated_from_correct_crops():
    """Self-check of the crop metric used in G5-C3."""
    src = H.wedged_noise((128, 128, 128), seed=1)
    crop = src[32:96, 30:94, 34:98]
    assert H.view_wedge_fraction(crop) < 0.005
    assert H.view_wedge_fraction(np.rot90(crop, 1, axes=(1, 2))) > 0.1


# --------------------------------------------------------------------------------------
# G5-C3: wedge-preserving augmentation, scale pairs and degradation keep the wedge
# --------------------------------------------------------------------------------------

def test_G5_C3_wedge_preserving_augment_keeps_the_wedge():
    from tomocpt.networks.ssl.views import wedge_preserving_augment

    v = H.wedged_noise((64, 64, 64), seed=2)
    rng = np.random.default_rng(0)
    patterns = set()
    for _ in range(N_DRAWS):
        out, tgt, _flips = wedge_preserving_augment(v.copy(), v.copy() * 2.0, rng)
        axes = H.flip_axes_of(out, v)
        assert axes is not None, "the view must be a pure flip of the input (no other rotation)"
        assert H.flip_axes_of(H.to_np(tgt) / 2.0, v) == axes, "the target gets the same flips as the view"
        patterns.add(axes)
        assert wedge_power_fraction(H.to_np(out)) < 0.02
    assert len(patterns) >= 4, f"flips look non-random: {patterns}"


def test_G5_C3_make_scale_pair_keeps_the_wedge():
    from tomocpt.networks.ssl.views import make_scale_pair

    src = H.wedged_noise((128, 128, 128), seed=3)
    rng = np.random.default_rng(1)
    worst = 0.0
    for i in range(N_DRAWS):
        s_a, s_b = H.log_uniform(rng, 1.0, 2.0), H.log_uniform(rng, 1.0, 2.0)
        r = int(np.floor(64 - 32 * max(s_a, s_b)))
        centre = tuple(float(64 + rng.integers(-r, r + 1)) for _ in range(3))
        np.random.seed(i)
        view_a, view_b, _t, _maps = make_scale_pair(src, centre, s_a, s_b, H.APIX_REF,
                                                    np.random.default_rng(100 + i))
        for v in (view_a, view_b):
            assert H.to_np(v).shape == (64, 64, 64)
            worst = max(worst, H.view_wedge_fraction(v))
    assert worst < 0.02, worst


def test_G5_C3_degrade_with_only_the_wedge_keeps_the_wedge():
    from tomocpt.networks.ssl.views import degrade

    cfg = H.ssl_cfg(version=2, use_fourier_aug=False, use_mw_aug=True, degradation_p=1.0)
    v = H.wedged_noise((64, 64, 64), seed=4)
    rng = np.random.default_rng(2)
    changed = 0
    for i in range(N_DRAWS):
        np.random.seed(i)
        out = H.to_np(degrade(v.copy(), cfg, rng))
        assert out.shape == (64, 64, 64)
        assert wedge_power_fraction(out) < 0.02
        changed += not np.allclose(out, v, atol=1e-4)
    assert changed > 0, "degrade never changed the view"


# --------------------------------------------------------------------------------------
# G5-C9: noise equalisation
# --------------------------------------------------------------------------------------

def _check_equalised(a, b):
    from tomocpt.networks.ssl.views import equalise_noise

    a2, b2 = equalise_noise(a, b)
    sa, sb = H.shell_noise_std(a2), H.shell_noise_std(b2)
    assert abs(sa / sb - 1.0) < 0.10, (sa, sb)


def test_G5_C9_equalise_noise_on_white_noise_of_different_levels():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((64, 64, 64)).astype(np.float32)
    b = (0.3 * rng.standard_normal((64, 64, 64))).astype(np.float32)
    assert H.shell_noise_std(a) / H.shell_noise_std(b) > 3.0
    np.random.seed(0)
    _check_equalised(a, b)
    np.random.seed(1)
    _check_equalised(b.copy(), a.copy())


def test_G5_C9_equalise_noise_on_a_scale_pair_of_pure_noise():
    from tomocpt.networks.ssl.views import make_scale_pair

    src = np.random.default_rng(6).standard_normal((128, 128, 128)).astype(np.float32)
    np.random.seed(0)
    view_a, view_b, _t, _maps = make_scale_pair(src, (64.0, 64.0, 64.0), 1.0, 1.9, H.APIX_REF,
                                                np.random.default_rng(0))
    a, b = H.to_np(view_a).astype(np.float32), H.to_np(view_b).astype(np.float32)
    np.random.seed(1)
    _check_equalised(a, b)


# --------------------------------------------------------------------------------------
# G5-C12: degrade keeps the mean and the standard deviation
# --------------------------------------------------------------------------------------

def test_G5_C12_degrade_keeps_mean_and_std():
    from tomocpt.networks.ssl.views import degrade

    cfg = H.ssl_cfg(version=2, degradation_p=1.0)
    v = (0.3 + 0.4 * np.random.default_rng(7).standard_normal((64, 64, 64))).astype(np.float32)
    rng = np.random.default_rng(3)
    changed = 0
    for i in range(10):
        np.random.seed(i)
        out = H.to_np(degrade(v.copy(), cfg, rng))
        np.testing.assert_allclose(out.mean(), v.mean(), rtol=0.02)
        np.testing.assert_allclose(out.std(), v.std(), rtol=0.02)
        changed += not np.allclose(out, v, atol=1e-4)
    assert changed > 0, "degrade never changed the view"
