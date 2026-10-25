"""Wedge orientation and output scale of the Fourier augmentation in tomocpt's frame.

Two defects found in review of the membrain-seg port:

1. membrain-seg transposes tomograms to ``(x, y, z)`` on load; tomocpt keeps
   mrcfile's ``(z, y, x)``. The reference mask, used unchanged, keeps the wedge
   around Z and removes the frequencies a real tomogram measures (around X).
2. The reference returns zero mean and unit standard deviation. tomocpt chunks
   have a standard deviation of about 0.3 after ``robust_normalization``, so every
   degraded chunk came back about 3x larger in amplitude than every clean one.
"""

import numpy as np
import pytest
import torch
import torchio as tio

from tomocpt.dataManager.fourier_augmentations import (
    RandomFourierDegradation,
    oriented_wedge_mask,
    wedge_mask,
)

N = 64


def _freq_grid(n=N):
    k = np.fft.fftshift(np.fft.fftfreq(n)) * n
    return np.meshgrid(k, k, k, indexing="ij")  # (kz, ky, kx) for a (z, y, x) array


def _real_tomogram_measured_region(max_tilt_deg=60.0):
    """Coefficients a tomogram with tilt axis Y and tilt range +/- max_tilt measures."""
    kz, _, kx = _freq_grid()
    return np.degrees(np.arctan2(np.abs(kz), np.abs(kx))) <= max_tilt_deg


def _robust_norm(x):
    p5, p95 = np.percentile(x, [5, 95])
    return np.clip((x - np.median(x)) / (p95 - p5), -3, 3).astype(np.float32)


def test_zyx_mask_keeps_the_x_axis_and_removes_the_z_axis():
    keep = oriented_wedge_mask((N, N, N), 45.0, array_order="zyx", zero_dc=False)
    c = N // 2
    assert keep[c, c, c + 10], "a coefficient on the kx axis must be kept"
    assert not keep[c + 10, c, c], "a coefficient on the kz (beam) axis must be removed"


def test_zyx_mask_agrees_with_a_real_tomogram_wedge():
    measured = _real_tomogram_measured_region(60.0)
    good = oriented_wedge_mask((N, N, N), 45.0, array_order="zyx", zero_dc=False)
    wrong = wedge_mask((N, N, N), 45.0, zero_dc=False)
    frac_good = (good & measured).sum() / measured.sum()
    frac_wrong = (wrong & measured).sum() / measured.sum()
    assert frac_good > 0.65
    assert frac_wrong < 0.40  # documents the defect this wrapper fixes


def test_zyx_mask_is_constant_along_the_tilt_axis():
    keep = oriented_wedge_mask((N, N, N), 45.0, array_order="zyx", zero_dc=False)
    assert (keep == keep[:, :1, :]).all()


def test_xyz_order_is_the_reference_mask():
    assert np.array_equal(
        oriented_wedge_mask((N, N, N), 45.0, array_order="xyz"), wedge_mask((N, N, N), 45.0)
    )


def test_non_cubic_shape_is_preserved():
    assert oriented_wedge_mask((32, 48, 64), 45.0).shape == (32, 48, 64)


def test_unknown_array_order_is_rejected():
    with pytest.raises(ValueError):
        oriented_wedge_mask((N, N, N), 45.0, array_order="yxz")
    with pytest.raises(ValueError):
        RandomFourierDegradation(array_order="yxz")


def test_degraded_output_has_no_power_in_the_real_missing_wedge():
    """Wedge only, blend covering the whole patch: nothing left where real data have nothing."""
    np.random.seed(0)
    t = RandomFourierDegradation(
        missing_wedge_prob=1.0,
        amplitude_prob=0.0,
        loc=(0.5, 0.5),
        scale=(1e6, 1e6),
    )
    x = np.random.default_rng(0).standard_normal((N, N, N)).astype(np.float32)
    y = t._degrade_patch(x, t._randomize())
    power = np.abs(np.fft.fftshift(np.fft.fftn(y))) ** 2
    missing = ~_real_tomogram_measured_region(60.0)
    assert power[missing].sum() / power.sum() < 0.01


@pytest.mark.parametrize("amp", [0.0, 1.0])
def test_output_keeps_the_input_mean_and_std(amp):
    np.random.seed(1)
    rng = np.random.default_rng(1)
    t = RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=amp)
    for _ in range(5):
        x = _robust_norm(rng.standard_normal((N, N, N)))
        y = t._degrade_patch(x, t._randomize())
        assert y.std() == pytest.approx(x.std(), rel=0.02)
        assert y.mean() == pytest.approx(x.mean(), abs=1e-3)


def test_reference_scaling_is_still_available():
    np.random.seed(2)
    x = _robust_norm(np.random.default_rng(2).standard_normal((N, N, N)))
    t = RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=0.0, match_input_stats=False)
    y = t._degrade_patch(x, t._randomize())
    assert y.std() == pytest.approx(1.0, rel=0.05)


def test_constant_patch_stays_constant():
    np.random.seed(3)
    t = RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0)
    x = np.full((N, N, N), 0.25, dtype=np.float32)
    y = t._degrade_patch(x, t._randomize())
    assert np.allclose(y, 0.25)


def test_transform_through_torchio_keeps_scale():
    np.random.seed(4)
    x = _robust_norm(np.random.default_rng(4).standard_normal((N, N, N)))
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=torch.from_numpy(x)[None]))
    out = RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0)(subject)
    assert float(out["input_data"].data.std()) == pytest.approx(float(x.std()), rel=0.02)
