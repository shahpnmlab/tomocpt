"""Contrast inversion, so the picker is invariant to tomogram polarity.

Tomograms come with particles either dark on light or light on dark, depending
on CTF handling and reconstruction convention. Without this augmentation a
picker learns the sign of the density as a feature and transfers badly across
the two.
"""

import numpy as np
import pytest
import torch
import torchio as tio

from conftest import CHUNK_SHAPE

from tomocpt.dataManager.dataloading import build_training_transforms
from tomocpt.dataManager.intensity_augmentations import RandomContrastInversion
from tomocpt.defaultConfigs.train_config import AugmentationConfig


def test_inversion_reflects_about_the_mean(image_tensor):
    """Output is the input reflected about its own mean."""
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    out = RandomContrastInversion(p=1.0)(subject)["input_data"].data

    expected = 2.0 * image_tensor.mean() - image_tensor
    torch.testing.assert_close(out, expected)


def test_inversion_preserves_the_mean(image_tensor):
    """Reflection leaves the mean untouched, unlike a naive negation."""
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    out = RandomContrastInversion(p=1.0)(subject)["input_data"].data

    assert out.mean().item() == pytest.approx(image_tensor.mean().item(), abs=1e-5)
    assert out.std().item() == pytest.approx(image_tensor.std().item(), rel=1e-5)


def test_inversion_flips_polarity(image_tensor):
    """Bright voxels become dark and vice versa: correlation is -1."""
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    out = RandomContrastInversion(p=1.0)(subject)["input_data"].data

    a = (image_tensor - image_tensor.mean()).flatten()
    b = (out - out.mean()).flatten()
    correlation = torch.dot(a, b) / (a.norm() * b.norm())
    assert correlation.item() == pytest.approx(-1.0, abs=1e-5)


def test_inversion_is_its_own_inverse(image_tensor):
    """Applying it twice returns the original, so the augmentation is symmetric.

    This is what lets it be used at any probability. The reference transform
    always inverts and relies on being applied exactly twice to balance out.
    """
    transform = RandomContrastInversion(p=1.0)
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    once = transform(subject)
    twice = transform(once)["input_data"].data

    torch.testing.assert_close(twice, image_tensor, atol=1e-5, rtol=1e-5)


def test_inversion_shifts_the_range_by_exactly_twice_the_mean(image_tensor):
    """Range moves by ``2 * mean``, which is negligible for zero-centred input.

    Reflecting about a non-zero mean cannot preserve the range exactly - that
    would need plain negation, which instead flips the mean. For tomocpt's
    inputs the mean is ~0 so the overshoot past the +/- 3 clip is ~1e-4, far too
    small to justify a re-clip that would cost the involution property.
    """
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    out = RandomContrastInversion(p=1.0)(subject)["input_data"].data

    shift = 2.0 * image_tensor.mean().item()
    assert out.min().item() == pytest.approx(shift - image_tensor.max().item(), abs=1e-5)
    assert out.max().item() == pytest.approx(shift - image_tensor.min().item(), abs=1e-5)

    overshoot = max(-3.0 - out.min().item(), out.max().item() - 3.0, 0.0)
    assert overshoot <= abs(shift) + 1e-6
    assert overshoot < 0.01, "input is not zero-centred enough for this to be safe"

    assert out.shape == image_tensor.shape
    assert out.dtype == torch.float32


def test_inversion_leaves_the_label_untouched(image_tensor, label_tensor):
    """The whole point: same particles, opposite density sign, same target."""
    subject = tio.Subject(
        input_data=tio.ScalarImage(tensor=image_tensor.clone()),
        target_data=tio.LabelMap(tensor=label_tensor.clone()),
    )
    out = RandomContrastInversion(p=1.0)(subject)

    assert torch.equal(out["target_data"].data, label_tensor)
    assert not torch.equal(out["input_data"].data, image_tensor)


def test_probability_zero_is_a_no_op(image_tensor):
    """``p=0`` leaves the image exactly alone."""
    subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
    out = RandomContrastInversion(p=0.0)(subject)["input_data"].data
    assert torch.equal(out, image_tensor)


def test_probability_is_respected(image_tensor):
    """At ``p=0.5`` roughly half the samples come back inverted."""
    transform = RandomContrastInversion(p=0.5)
    torch.manual_seed(0)
    inverted = 0
    for _ in range(200):
        subject = tio.Subject(input_data=tio.ScalarImage(tensor=image_tensor.clone()))
        out = transform(subject)["input_data"].data
        if not torch.equal(out, image_tensor):
            inverted += 1
    assert 70 < inverted < 130, f"{inverted}/200 inverted, expected roughly half"


def test_absent_from_the_pipeline_by_default():
    """Off by default, so existing runs are unchanged."""
    names = [
        type(t).__name__
        for t in build_training_transforms(AugmentationConfig()).transforms
    ]
    assert "RandomContrastInversion" not in names


def test_inversion_precedes_the_other_intensity_transforms():
    """Polarity is a property of the tomogram, so it is established first.

    It must in particular precede ``RandomLocalGamma``, which raises the image
    to a power after rescaling to ``[0, 1]`` and is therefore not symmetric
    about the mean - applying it before the inversion would model a
    gamma response on the wrong polarity.
    """
    cfg = AugmentationConfig(
        contrast_inversion_p=0.5,
        brightness_gradient_p=0.3,
        local_gamma_p=0.3,
        brightness_p=0.3,
        contrast_p=0.3,
    )
    names = [type(t).__name__ for t in build_training_transforms(cfg).transforms]
    index = names.index("RandomContrastInversion")

    for later in ["RandomBrightnessGradient", "RandomLocalGamma", "RandomBrightness", "RandomContrast"]:
        assert names.index(later) > index, f"{later} runs before contrast inversion"


def test_inversion_runs_after_the_fourier_degradation():
    """Still downstream of every spatial transform and of the wedge."""
    cfg = AugmentationConfig(
        contrast_inversion_p=0.5, use_mw_aug=True, use_fourier_aug=True
    )
    names = [type(t).__name__ for t in build_training_transforms(cfg).transforms]
    assert names.index("RandomContrastInversion") > names.index("RandomFourierDegradation")


def test_full_pipeline_with_inversion_preserves_the_label(image_tensor, label_tensor):
    """End to end, with spatial transforms off so the label should not move."""
    cfg = AugmentationConfig(
        affine_p=0.0,
        elastic_blur_p=0.0,
        contrast_inversion_p=1.0,
        use_mw_aug=True,
        use_fourier_aug=True,
        missing_wedge_prob=1.0,
        amplitude_prob=1.0,
    )
    subject = tio.Subject(
        input_data=tio.ScalarImage(tensor=image_tensor.clone()),
        target_data=tio.LabelMap(tensor=label_tensor.clone()),
    )
    out = build_training_transforms(cfg)(subject)

    assert torch.equal(out["target_data"].data, label_tensor)
    assert torch.isfinite(out["input_data"].data).all()
