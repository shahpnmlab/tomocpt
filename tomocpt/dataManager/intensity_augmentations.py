"""Localized intensity augmentations for cryo-ET subvolumes.

Ported from membrain-seg's ``segmentation/dataloading/transforms.py`` (themselves
reimplementations from the batchgenerators package), restructured as
:class:`torchio.IntensityTransform` subclasses so TorchIO guarantees that only
:class:`torchio.ScalarImage` data is touched and every :class:`torchio.LabelMap`
is skipped.

Parameter defaults are the values used at membrain-seg's actual call site
(``segmentation/dataloading/memseg_augmentation.py``), not the class defaults.

Several augmentations in the reference pipeline are deliberately *not* ported,
because TorchIO already ships an equivalent and this pipeline is TorchIO
end-to-end:

==================================== ==========================
Reference                            TorchIO equivalent
==================================== ==========================
``SimulateLowResolutionTransform``   :class:`torchio.RandomAnisotropy`
Gaussian noise                       :class:`torchio.RandomNoise`
Global gamma                         :class:`torchio.RandomGamma`
Smooth multiplicative field          :class:`torchio.RandomBiasField`
==================================== ==========================

:class:`torchio.RandomBiasField` is a *multiplicative* smooth field and is
complementary to :class:`RandomBrightnessGradient`, which is additive - they are
not redundant with one another.
"""

from typing import Callable, Sequence, Tuple, Union

import numpy as np
import torch
import torchio as tio

from tomocpt.dataManager.fourier_augmentations import (
    generate_gaussian_kernel,
    run_interpolation,
    sample_scalar,
)

__all__ = [
    "RandomContrastInversion",
    "RandomBrightnessGradient",
    "RandomLocalGamma",
    "RandomBrightness",
    "RandomContrast",
]


def _default_gradient_scale(img_shape: Sequence[int], axis: int) -> float:
    """Sample the call-site Gaussian blob width for one axis.

    Note this is *not* the width used by
    :class:`~tomocpt.dataManager.fourier_augmentations.RandomFourierDegradation`:
    the brightness-gradient and local-gamma call sites use the tighter
    ``exp(U(log(n // 6), log(n)))``, giving a blob smaller than the patch, which
    is what makes these augmentations local.
    """
    return float(
        np.exp(np.random.uniform(np.log(img_shape[axis] // 6), np.log(img_shape[axis])))
    )


def _default_max_strength(image, kernel) -> float:
    """Sample a signed brightness-gradient strength, avoiding near-zero values."""
    if np.random.uniform() < 0.5:
        return float(np.random.uniform(-5, -1))
    return float(np.random.uniform(1, 5))


def _default_gamma() -> float:
    """Sample a local gamma exponent well away from the identity at 1.0."""
    if np.random.uniform() < 0.5:
        return float(np.random.uniform(0.01, 0.8))
    return float(np.random.uniform(1.5, 4))


class RandomContrastInversion(tio.IntensityTransform):
    """Randomly flip the contrast polarity of the image.

    Tomograms come in both conventions - particles dark on a light background
    and light on a dark one - depending on CTF handling and reconstruction
    convention. A picker trained on one polarity learns the sign of the density
    as a feature and transfers badly to the other. Applying this makes the
    network invariant to the convention instead.

    The label is deliberately left alone, so the network must locate the same
    particles whichever way the density runs.

    The image is reflected about its own mean rather than simply negated. For
    tomocpt's zero-centred inputs the two are equivalent, but reflecting is
    correct for any input whose mean is not already zero, and it leaves the mean
    untouched. Reflection is its own inverse, so applying it twice is a no-op and
    the augmentation stays symmetric at any probability - unlike the reference's
    ``RandAdjustContrastWithInversionAndStats``, which always inverts and relies
    on being applied exactly twice to balance out.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`torchio.IntensityTransform`, including ``p``, which
        is the probability of inverting. ``p=0.5`` gives an even mix of both
        polarities.

    Notes
    -----
    Reflection preserves the mean and standard deviation exactly, and shifts the
    range by exactly ``2 * mean``: a ``+/- 3`` clipped input comes back in
    ``[2 * mean - 3, 2 * mean + 3]``. tomocpt's inputs are zero-centred by
    ``robust_normalization``, so in practice that overshoot is on the order of
    1e-4 and no re-clipping is applied - clipping would cost the involution
    property, which is worth more than four decimal places of range.

    Plain negation would instead preserve the range exactly and flip the sign of
    the mean. The two differ only by ``2 * mean``; preserving the distribution's
    shape is the more useful guarantee.
    """

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Invert every scalar image about its mean; skip every label map."""
        for image in self.get_images(subject):
            data = image.data
            image.set_data(2.0 * data.mean() - data)
        return subject


class RandomBrightnessGradient(tio.IntensityTransform):
    """Add a scaled Gaussian brightness blob to the image.

    The blob centre is drawn at random and may fall outside the patch, so the
    result ranges from a gentle one-sided gradient to a localized bright or dark
    spot.

    Parameters
    ----------
    scale : callable or tuple, optional
        Per-axis blob width, sampled with ``(spatial_shape, axis)``.
    loc : callable or tuple, optional
        Per-axis blob centre in units of the spatial shape.
    max_strength : callable or tuple or float, optional
        Peak amplitude of the added blob, sampled with ``(image, kernel)``.
    mean_centered : bool, optional
        Subtract the kernel mean before scaling, making the gradient
        zero-mean rather than purely additive.
    **kwargs
        Forwarded to :class:`torchio.IntensityTransform`, including ``p``.
    """

    def __init__(
        self,
        scale: Union[Callable, Tuple[float, float]] = _default_gradient_scale,
        loc: Union[Callable, Tuple[float, float]] = (-0.5, 1.5),
        max_strength: Union[Callable, Tuple[float, float], float] = _default_max_strength,
        mean_centered: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.loc = loc
        self.max_strength = max_strength
        self.mean_centered = mean_centered

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Add a brightness blob to every scalar image; skip every label map."""
        for image in self.get_images(subject):
            data = image.data
            out = data.clone()
            for c in range(data.shape[0]):
                patch = np.asarray(data[c], dtype=np.float32)
                kernel = generate_gaussian_kernel(patch.shape, self.scale, self.loc)
                if self.mean_centered:
                    kernel = kernel - kernel.mean()
                max_kernel_val = max(np.max(np.abs(kernel)), 1e-8)
                strength = sample_scalar(self.max_strength, patch, kernel)
                kernel = kernel / max_kernel_val * strength
                # Out of place: the reference does ``image += kernel``, which
                # mutates the caller's tensor.
                out[c] = torch.from_numpy((patch + kernel).astype(np.float32))
            image.set_data(out)
        return subject


class RandomLocalGamma(tio.IntensityTransform):
    """Apply a gamma correction within a soft Gaussian blob.

    The patch is rescaled to ``[0, 1]``, raised to a random power, blended back
    through the kernel, and restored to its original range - so intensities
    outside the blob are untouched and the global dynamic range is preserved.

    Parameters
    ----------
    scale : callable or tuple, optional
        Per-axis blob width, sampled with ``(spatial_shape, axis)``.
    loc : callable or tuple, optional
        Per-axis blob centre in units of the spatial shape.
    gamma : callable or tuple, optional
        Gamma exponent, sampled with no arguments.
    **kwargs
        Forwarded to :class:`torchio.IntensityTransform`, including ``p``.
    """

    def __init__(
        self,
        scale: Union[Callable, Tuple[float, float]] = _default_gradient_scale,
        loc: Union[Callable, Tuple[float, float]] = (-0.5, 1.5),
        gamma: Union[Callable, Tuple[float, float]] = _default_gamma,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.loc = loc
        self.gamma = gamma

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply a local gamma to every scalar image; skip every label map."""
        for image in self.get_images(subject):
            data = image.data
            out = data.clone()
            for c in range(data.shape[0]):
                patch = np.asarray(data[c], dtype=np.float32)
                kernel = generate_gaussian_kernel(patch.shape, self.scale, self.loc)
                mn, mx = float(patch.min()), float(patch.max())
                scaled = (patch - mn) / max(mx - mn, 1e-8)
                modified = np.power(scaled, sample_scalar(self.gamma))
                blended = run_interpolation(scaled, modified, kernel) * (mx - mn) + mn
                out[c] = torch.from_numpy(blended.astype(np.float32))
            image.set_data(out)
        return subject


class RandomBrightness(tio.IntensityTransform):
    """Shift the whole image by a Gaussian-distributed constant.

    Parameters
    ----------
    mu : float, optional
        Mean of the additive constant.
    sigma : float, optional
        Standard deviation of the additive constant.
    **kwargs
        Forwarded to :class:`torchio.IntensityTransform`, including ``p``.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Shift every scalar image; skip every label map."""
        for image in self.get_images(subject):
            add_const = float(np.random.normal(loc=self.mu, scale=self.sigma))
            image.set_data(image.data + add_const)
        return subject


class RandomContrast(tio.IntensityTransform):
    """Scale the image about its mean by a random contrast factor.

    Parameters
    ----------
    contrast_range : Tuple[float, float], optional
        Range to draw the multiplicative contrast factor from.
    preserve_range : bool, optional
        Clamp the result back into the input's original min/max range.
    **kwargs
        Forwarded to :class:`torchio.IntensityTransform`, including ``p``.
    """

    def __init__(
        self,
        contrast_range: Tuple[float, float] = (0.5, 2.0),
        preserve_range: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Rescale contrast of every scalar image; skip every label map."""
        for image in self.get_images(subject):
            data = image.data
            factor = float(
                np.random.uniform(self.contrast_range[0], self.contrast_range[1])
            )
            mean = data.mean()
            out = mean + factor * (data - mean)
            if self.preserve_range:
                out = out.clamp(min=data.min(), max=data.max())
            image.set_data(out)
        return subject
