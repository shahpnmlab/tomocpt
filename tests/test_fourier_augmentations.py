"""Tests for the Fourier and intensity augmentations ported from membrain-seg.

Test 1 (port equivalence) is the load-bearing one: it is the only check that the
vendored numerics actually match the originals, and it can only run while the
reference checkout is present.
"""

import numpy as np
import pytest
import torch
import torchio as tio

from conftest import CHUNK_SHAPE, requires_membrain

from tomocpt.dataManager.dataloading import build_training_transforms
from tomocpt.dataManager.filter_utils import hypotenuse_ndim, rotational_kernel
from tomocpt.dataManager.fourier_augmentations import (
    RandomFourierDegradation,
    fft_patch_to_real,
    generate_gaussian_kernel,
    normalize_and_fft_patch,
    wedge_mask,
)
from tomocpt.dataManager.intensity_augmentations import (
    RandomBrightness,
    RandomBrightnessGradient,
    RandomContrast,
    RandomLocalGamma,
)
from tomocpt.defaultConfigs.train_config import AugmentationConfig


# --------------------------------------------------------------------------
# 1. Port equivalence
# --------------------------------------------------------------------------


@requires_membrain
@pytest.mark.parametrize("shape", [(64, 64, 64), (32, 32, 32), (16, 16, 16)])
@pytest.mark.parametrize("angle", [0.0, 15.0, 30.0, 45.0, 60.0, 88.0, 90.0])
def test_wedge_mask_matches_reference(membrain_ref, shape, angle):
    """The vendored wedge mask is bit-identical to membrain-seg's.

    ``taper_width=0`` is the exact-reference path. This is what makes the
    deliberate deviations elsewhere trustworthy: the geometry itself was not
    quietly "improved" in transit.
    """
    ours = wedge_mask(shape, angle, taper_width=0.0)
    theirs = membrain_ref.wedge_mask(shape, angle)
    assert ours.dtype == theirs.dtype
    assert np.array_equal(ours, theirs)


@requires_membrain
@pytest.mark.parametrize("shape", [(64, 64, 64), (31, 64, 17), (16, 20, 24)])
def test_rotational_kernel_matches_reference(membrain_ref, shape):
    """The vendored rotational kernel is bit-identical to membrain-seg's."""
    arr = np.random.default_rng(0).random(40)
    assert np.array_equal(
        rotational_kernel(arr, shape), membrain_ref.rotational_kernel(arr, shape)
    )


@requires_membrain
def test_hypotenuse_ndim_matches_reference(membrain_ref):
    """The vendored radial-distance helper is bit-identical to membrain-seg's."""
    for shape in [(13, 9, 7), (64, 64, 64), (8, 8)]:
        axes = np.ogrid[tuple(slice(0, s) for s in shape)]
        assert np.array_equal(
            hypotenuse_ndim(axes), membrain_ref.hypotenuse_ndim(axes)
        )
        assert np.array_equal(
            hypotenuse_ndim(axes, offset=0), membrain_ref.hypotenuse_ndim(axes, offset=0)
        )


@requires_membrain
def test_synthesized_spectrum_matches_reference(membrain_ref):
    """``get_line_plot`` matches the reference for an identical random state."""
    np.random.seed(1234)
    _, ours = __import__(
        "tomocpt.dataManager.fourier_augmentations", fromlist=["get_line_plot"]
    ).get_line_plot(32, smooth_sigma=3.0, step_sigma=2.0, offset=5.0)
    np.random.seed(1234)
    _, theirs = membrain_ref.get_line_plot(
        32, smooth_sigma=3.0, step_sigma=2.0, offset=5.0
    )
    assert np.array_equal(ours, theirs)


# --------------------------------------------------------------------------
# 2. Round-trip identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((64, 64, 64), id="even-cubic"),
        pytest.param((32, 48, 16), id="even-anisotropic"),
        pytest.param((31, 31, 31), id="odd-cubic"),
        pytest.param((31, 33, 17), id="odd-anisotropic"),
    ],
)
def test_fft_round_trip_is_identity(shape, rng):
    """FFT then inverse FFT returns the input, for even *and* odd sizes.

    ``normalize_and_fft_patch`` rescales to ``[0, 1]``, so the round trip is
    compared against that rescaled patch rather than the raw input.

    The odd cases are the point: the reference inverts its ``fftshift`` with
    another ``fftshift``, which is correct only when every axis has even length.
    At ``CHUNK_SIZE=64`` the bug is invisible; on any odd axis it scrambles the
    volume. These parametrizations fail against the unfixed reference.
    """
    patch = rng.standard_normal(shape).astype(np.float32)
    expected = patch - patch.min()
    expected /= expected.max()

    result = fft_patch_to_real(normalize_and_fft_patch(patch))

    assert result.shape == shape
    np.testing.assert_allclose(result, expected, atol=1e-5)


# --------------------------------------------------------------------------
# 3. Wedge geometry and the angle convention
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("angle", "expected_fraction"),
    [(0.0, 0.0), (30.0, 0.288), (45.0, 0.508), (60.0, 0.712), (90.0, 1.0)],
)
def test_keep_angle_is_the_angle_retained(angle, expected_fraction):
    """``keep_angle`` counts degrees KEPT, not degrees removed.

    The reference parameter is named ``missing_angle`` while its own docstring
    describes angles to keep, and production passes ``(45, 45)``. This pins the
    convention empirically: the retained fraction rises monotonically with the
    angle, 0 keeps nothing and 90 keeps everything. Hence the tomocpt name
    ``keep_angle_range``.
    """
    mask = wedge_mask(CHUNK_SHAPE, angle, zero_dc=False)
    assert mask.mean() == pytest.approx(expected_fraction, abs=0.01)


def test_keep_fraction_is_monotonic_in_angle():
    """Widening the retained angle never removes coefficients."""
    fractions = [wedge_mask(CHUNK_SHAPE, a, zero_dc=False).mean() for a in range(0, 91, 10)]
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))


def test_mask_is_a_wedge_not_a_cone():
    """The mask is constant along axis 1, the tilt axis.

    A mask that varied along the tilt axis would be a missing *cone*, a
    different and physically wrong degradation.
    """
    mask = wedge_mask(CHUNK_SHAPE, 45.0, zero_dc=False)
    for i in range(1, CHUNK_SHAPE[1]):
        assert np.array_equal(mask[:, 0, :], mask[:, i, :])


def test_dc_handling_matches_reference_and_is_recoverable():
    """The reference *removes* DC; ``zero_dc=False`` retains it.

    ``wedge_mask`` sets the centre of the removal mask to 1 immediately before
    inverting it, so the DC coefficient is discarded rather than kept - the
    inverse of what the surrounding code reads as intending. It is preserved
    under the default for port fidelity, and it is harmless in practice because
    the output is re-normalized to zero mean afterwards, which discards DC
    anyway. This test documents both halves so a future change is deliberate.
    """
    centre = tuple(s // 2 for s in CHUNK_SHAPE)
    assert not wedge_mask(CHUNK_SHAPE, 45.0, zero_dc=True)[centre]
    assert wedge_mask(CHUNK_SHAPE, 45.0, zero_dc=False)[centre]


def test_taper_softens_the_wedge_boundary():
    """A non-zero taper replaces the hard edge with a graded ramp.

    The hard edge causes Gibbs ringing along the wedge boundary, which is a
    shortcut feature a picking model can latch onto.
    """
    hard = wedge_mask(CHUNK_SHAPE, 45.0, taper_width=0.0, zero_dc=False)
    soft = wedge_mask(CHUNK_SHAPE, 45.0, taper_width=20.0, zero_dc=False)

    assert soft.dtype == np.float32
    assert soft.min() >= 0.0 and soft.max() <= 1.0
    # Intermediate values exist only in the tapered version.
    assert not np.any((hard > 0) & (hard < 1))
    assert np.count_nonzero((soft > 0.01) & (soft < 0.99)) > 0
    # The taper straddles the boundary: it keeps more than the hard mask
    # somewhere and less somewhere else.
    assert np.any(soft > hard) and np.any(soft < hard)


def test_wedge_symmetry_under_axis_reflection():
    """The wedge is symmetric under reflection of the slope axis.

    See :func:`test_known_friedel_asymmetry_at_the_wedge_boundary` for the
    boundary voxels where symmetry does *not* hold.
    """
    mask = wedge_mask(CHUNK_SHAPE, 45.0, zero_dc=False)
    flip = np.arange(CHUNK_SHAPE[0])[::-1]
    assert np.array_equal(mask, mask[flip, :, :])


# --------------------------------------------------------------------------
# 4. Real, finite output
# --------------------------------------------------------------------------


def test_output_is_real_and_finite(image_tensor):
    """No NaN or Inf survives the transform, under every branch combination."""
    for mw, amp in [(True, True), (True, False), (False, True)]:
        transform = RandomFourierDegradation(
            missing_wedge_aug=mw,
            amplitude_aug=amp,
            missing_wedge_prob=1.0 if mw else 0.0,
            amplitude_prob=1.0 if amp else 0.0,
        )
        subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
        out = transform(subject)["input_data"].data
        assert torch.isfinite(out).all(), f"non-finite output for mw={mw} amp={amp}"
        assert out.dtype == torch.float32
        assert not torch.is_complex(out)


def test_reclip_restores_the_input_range(image_tensor):
    """``reclip`` puts the output back inside tomocpt's +/- 3 input range.

    The reference ends by normalizing to zero mean and unit standard deviation,
    a different distribution from the hard-clipped one ``robust_normalization``
    feeds in.
    """
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    out = RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0, reclip=True)(
        subject
    )["input_data"].data
    assert out.min() >= -3.0 and out.max() <= 3.0

    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    unclipped = RandomFourierDegradation(
        missing_wedge_prob=1.0, amplitude_prob=1.0, reclip=False
    )(subject)["input_data"].data
    assert torch.isfinite(unclipped).all()


def test_known_friedel_asymmetry_at_the_wedge_boundary():
    """Characterises a known defect carried over from the reference.

    The wedge is constructed about ``mean(arange(64)) == 31.5`` while
    ``fftshift`` places DC at index 32, and the strict ``<`` / ``>`` comparisons
    break ties differently on the two boundary lines. The result is that the
    boundary voxels are not Friedel-symmetric, so the inverse FFT is not exactly
    real and ``np.real`` silently discards a residual of a few percent - far
    above float32 noise.

    This is asserted rather than fixed because the fix would change
    ``wedge_mask`` and invalidate the port-equivalence test above. If a future
    change centres the wedge correctly, this test fails and should be replaced
    with a strict symmetry assertion.
    """
    mask = wedge_mask(CHUNK_SHAPE, 45.0, zero_dc=False)
    flip = np.arange(CHUNK_SHAPE[0])[::-1]
    asymmetric = np.count_nonzero(mask != mask[np.ix_(flip, flip, flip)])
    assert 0 < asymmetric / mask.size < 0.05, "asymmetry outside the known range"

    patch = np.random.default_rng(0).standard_normal(CHUNK_SHAPE).astype(np.float32)
    spectrum = normalize_and_fft_patch(patch)
    spectrum[~mask] = 0.0
    inverse = np.fft.ifftn(np.fft.ifftshift(spectrum))
    residual = np.abs(inverse.imag).max() / np.abs(inverse.real).max()
    assert 0.01 < residual < 0.15, f"imaginary residual {residual} outside known range"


# --------------------------------------------------------------------------
# 5. The label is never touched
# --------------------------------------------------------------------------


def test_fourier_transform_leaves_label_bit_identical(subject, label_tensor):
    """The paper's core requirement: degrade the image, keep the label exact."""
    out = RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0)(subject)
    assert torch.equal(out["target_data"].data, label_tensor)
    assert not torch.equal(out["input_data"].data, subject["input_data"].data.clone())


@pytest.mark.parametrize(
    "transform",
    [
        RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0),
        RandomBrightnessGradient(),
        RandomLocalGamma(),
        RandomBrightness(),
        RandomContrast(),
    ],
    ids=["fourier", "brightness_gradient", "local_gamma", "brightness", "contrast"],
)
def test_every_intensity_transform_leaves_label_bit_identical(
    transform, image_tensor, label_tensor
):
    """Every ported transform is an ``IntensityTransform`` and skips LabelMaps.

    TorchIO enforces this structurally rather than by a ``keys=["image"]``
    convention, but the guarantee is worth asserting: this is the thing most
    likely to break silently during a restructure.
    """
    subject = tio.Subject(
        input_data=tio.ScalarImage(tensor=image_tensor.clone()),
        target_data=tio.LabelMap(tensor=label_tensor.clone()),
    )
    out = transform(subject)
    assert torch.equal(out["target_data"].data, label_tensor)
    assert not torch.equal(out["input_data"].data, image_tensor)


def test_full_pipeline_leaves_label_bit_identical(image_tensor, label_tensor):
    """The whole composed pipeline, minus spatial transforms, preserves the label.

    The affine and elastic transforms are switched off because they are
    *supposed* to move the label; everything downstream of them is not.
    """
    cfg = AugmentationConfig(
        affine_p=0.0,
        elastic_blur_p=0.0,
        use_mw_aug=True,
        use_fourier_aug=True,
        missing_wedge_prob=1.0,
        amplitude_prob=1.0,
        brightness_gradient_p=1.0,
        local_gamma_p=1.0,
        brightness_p=1.0,
        contrast_p=1.0,
        noise_p=1.0,
        gamma_p=1.0,
        bias_field_p=1.0,
    )
    subject = tio.Subject(
        input_data=tio.ScalarImage(tensor=image_tensor.clone()),
        target_data=tio.LabelMap(tensor=label_tensor.clone()),
    )
    out = build_training_transforms(cfg)(subject)
    assert torch.equal(out["target_data"].data, label_tensor)
    assert not torch.equal(out["input_data"].data, image_tensor)


# --------------------------------------------------------------------------
# 6. No input mutation
# --------------------------------------------------------------------------


def test_normalize_and_fft_patch_does_not_mutate_its_argument(rng):
    """The reference rescaled a view of the caller's tensor in place."""
    patch = rng.standard_normal(CHUNK_SHAPE).astype(np.float32)
    original = patch.copy()
    normalize_and_fft_patch(patch)
    assert np.array_equal(patch, original)


@pytest.mark.parametrize(
    "transform",
    [
        RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0),
        RandomBrightnessGradient(),
        RandomLocalGamma(),
        RandomBrightness(),
        RandomContrast(),
    ],
    ids=["fourier", "brightness_gradient", "local_gamma", "brightness", "contrast"],
)
def test_transforms_do_not_mutate_the_source_tensor(transform, image_tensor):
    """The caller's tensor is unchanged after the transform returns."""
    original = image_tensor.clone()
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor))
    transform(subject)
    assert torch.equal(image_tensor, original)


# --------------------------------------------------------------------------
# 7. Shape and dtype invariance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0),
        RandomBrightnessGradient(),
        RandomLocalGamma(),
        RandomBrightness(),
        RandomContrast(),
    ],
    ids=["fourier", "brightness_gradient", "local_gamma", "brightness", "contrast"],
)
def test_shape_and_dtype_are_preserved(transform, image_tensor):
    """``(1, 64, 64, 64)`` float32 in, ``(1, 64, 64, 64)`` float32 out."""
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    out = transform(subject)["input_data"].data
    assert out.shape == image_tensor.shape == (1,) + CHUNK_SHAPE
    assert out.dtype == torch.float32


def test_gaussian_kernel_shape_and_range():
    """The shared Gaussian kernel spans the patch with values in ``(0, 1]``."""
    kernel = generate_gaussian_kernel(CHUNK_SHAPE, (10.0, 30.0), (-0.5, 1.5))
    assert kernel.shape == CHUNK_SHAPE
    assert kernel.min() > 0.0 and kernel.max() <= 1.0


# --------------------------------------------------------------------------
# 8. Determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_the_same_output(image_tensor):
    """Randomness comes from ``numpy.random``, so seeding it reproduces a run."""
    outputs = []
    for _ in range(2):
        np.random.seed(4242)
        torch.manual_seed(4242)
        subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
        transform = RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0)
        outputs.append(transform(subject)["input_data"].data)
    assert torch.equal(outputs[0], outputs[1])


def test_different_seeds_give_different_output(image_tensor):
    """Guards against a transform that silently does nothing."""
    outputs = []
    for seed in (1, 2):
        np.random.seed(seed)
        torch.manual_seed(seed)
        subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
        transform = RandomFourierDegradation(missing_wedge_prob=1.0, amplitude_prob=1.0)
        outputs.append(transform(subject)["input_data"].data)
    assert not torch.equal(outputs[0], outputs[1])


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------


def test_defaults_reproduce_the_historical_pipeline():
    """Default config must not change any existing run.

    Compared against a verbatim copy of the pre-change hardcoded pipeline.
    """
    historical = tio.Compose(
        [
            tio.RandomAffine(degrees=45, default_pad_value="otsu", p=0.8),
            tio.OneOf(
                {tio.RandomElasticDeformation(): 0.1, tio.RandomBlur(std=1): 0.1},
                p=0.75,
            ),
        ]
    )
    built = build_training_transforms(AugmentationConfig())

    assert [type(t) for t in built.transforms] == [type(t) for t in historical.transforms]
    assert [t.probability for t in built.transforms] == [
        t.probability for t in historical.transforms
    ]
    assert built.transforms[0].degrees == historical.transforms[0].degrees
    assert (
        built.transforms[0].default_pad_value
        == historical.transforms[0].default_pad_value
    )

    def by_type(one_of, cls):
        return next(k for k in one_of.transforms_dict if isinstance(k, cls))

    assert (
        by_type(built.transforms[1], tio.RandomBlur).std_ranges
        == by_type(historical.transforms[1], tio.RandomBlur).std_ranges
    )
    assert (
        by_type(built.transforms[1], tio.RandomElasticDeformation).max_displacement
        == by_type(historical.transforms[1], tio.RandomElasticDeformation).max_displacement
    )


def test_fourier_degradation_runs_after_all_spatial_transforms():
    """Ordering is the entire point of the augmentation.

    Running the wedge after the rotations makes it land at a random orientation
    relative to the specimen while the label stays geometrically correct.
    Putting it earlier silently destroys the augmentation while still training
    normally.
    """
    cfg = AugmentationConfig(
        use_mw_aug=True,
        use_fourier_aug=True,
        brightness_gradient_p=0.3,
        local_gamma_p=0.3,
        brightness_p=0.3,
        contrast_p=0.3,
    )
    names = [type(t).__name__ for t in build_training_transforms(cfg).transforms]
    spatial = {"RandomAffine", "OneOf"}
    index = names.index("RandomFourierDegradation")
    assert spatial.issubset(set(names[:index]))
    assert not spatial & set(names[index + 1 :])


def test_fourier_transform_absent_when_both_switches_are_off():
    """Neither switch set means no Fourier transform in the pipeline at all."""
    names = [type(t).__name__ for t in build_training_transforms(AugmentationConfig()).transforms]
    assert "RandomFourierDegradation" not in names


@pytest.mark.parametrize(
    ("mw", "fourier"), [(True, False), (False, True), (True, True)]
)
def test_switches_are_independent(mw, fourier):
    """``use_mw_aug`` and ``use_fourier_aug`` control their halves separately."""
    cfg = AugmentationConfig(use_mw_aug=mw, use_fourier_aug=fourier)
    transform = next(
        t
        for t in build_training_transforms(cfg).transforms
        if isinstance(t, RandomFourierDegradation)
    )
    assert transform.missing_wedge_aug is mw
    assert transform.amplitude_aug is fourier


def test_disabled_transform_is_a_no_op(image_tensor):
    """With both augmentations off the image passes through untouched."""
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    out = RandomFourierDegradation(missing_wedge_aug=False, amplitude_aug=False)(subject)
    assert torch.equal(out["input_data"].data, image_tensor)
