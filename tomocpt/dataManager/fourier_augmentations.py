"""Fourier-domain augmentation for cryo-ET subvolumes.

Ported from membrain-seg's ``FourierAugmentations`` branch
(``segmentation/dataloading/fourier_augmentations.py``), restructured as a
:class:`torchio.IntensityTransform` so that TorchIO guarantees structurally that
only :class:`torchio.ScalarImage` data is touched and every
:class:`torchio.LabelMap` is skipped.

The transform combines three effects, all of which matter:

1. **Missing wedge** - zero a wedge of Fourier coefficients, imitating the
   restricted tilt range of a tomogram.
2. **Amplitude spectrum randomization** - synthesize a radially averaged
   amplitude spectrum from a Gaussian-smoothed random walk, expand it to a 3D
   rotational kernel and multiply it into the FFT. This imitates varied
   CTF / defocus / dose-filtering conditions.
3. **Localized application** - blend the degraded patch back into the original
   through a soft Gaussian blob, so a patch is partly degraded and partly clean.
   As in the reference, drawing the wedge *forces* the blend on, so the wedge is
   always localized. That is deliberate.

The label is never modified, which is the whole point of the augmentation: the
network is asked to infer structure from context in wedge-impacted regions.
"""

from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
import numpy.fft as fft
import torch
import torchio as tio
from scipy.ndimage import gaussian_filter1d

from tomocpt.dataManager.filter_utils import rotational_kernel

__all__ = [
    "sample_scalar",
    "generate_gaussian_kernel",
    "wedge_mask",
    "get_line_plot",
    "normalize_and_fft_patch",
    "fft_patch_to_real",
    "run_interpolation",
    "RandomFourierDegradation",
]


def sample_scalar(value: Union[Tuple, List, Callable, Any], *args: Any) -> Any:
    """Sample a scalar from a range, or compute it with a callable.

    Implementation from the batchgenerators package, via membrain-seg.

    Parameters
    ----------
    value : tuple or list or callable or Any
        A ``(low, high)`` range to draw uniformly from, a callable invoked with
        ``*args``, or a plain value returned as-is.
    *args : Any
        Extra arguments forwarded to ``value`` when it is callable.

    Returns
    -------
    Any
        The sampled or computed scalar.
    """
    if isinstance(value, (tuple, list)):
        return np.random.uniform(value[0], value[1])
    elif callable(value):
        return value(*args)
    else:
        return value


def generate_gaussian_kernel(
    img_shape: Sequence[int],
    scale: Union[Tuple, List, Callable, Any],
    loc: Union[Tuple, List, Callable, Any],
) -> np.ndarray:
    """Build an anisotropic Gaussian blob over an image grid.

    The blob centre ``loc`` is expressed in units of the image shape and may fall
    outside the image, which is what lets the blob cover the patch only
    partially.

    This helper is shared by :class:`RandomFourierDegradation` and the intensity
    augmentations, rather than being duplicated per transform as in the
    reference.

    Parameters
    ----------
    img_shape : Sequence[int]
        Spatial shape of the patch, without a channel axis.
    scale : tuple or list or callable or Any
        Per-axis Gaussian width, sampled via :func:`sample_scalar` with
        ``(img_shape, axis_index)``.
    loc : tuple or list or callable or Any
        Per-axis blob centre in units of the image shape, sampled the same way.

    Returns
    -------
    np.ndarray
        Kernel of shape ``img_shape`` with values in ``(0, 1]``.

    Notes
    -----
    Unlike the reference, no leading channel axis is added: callers here operate
    on a single 3D patch at a time, so the extra axis would only broadcast back
    out again.
    """
    n_dim = len(img_shape)
    scale = [sample_scalar(scale, img_shape, i) for i in range(n_dim)]
    loc = [sample_scalar(loc, img_shape, i) for i in range(n_dim)]
    loc = np.array(loc) * np.array(img_shape)
    coords = [
        np.linspace(-loc[i], img_shape[i] - loc[i], img_shape[i]) for i in range(n_dim)
    ]
    meshgrid = np.meshgrid(*coords, indexing="ij")
    kernel = np.exp(-0.5 * sum((meshgrid[i] / scale[i]) ** 2 for i in range(n_dim)))
    return kernel


def wedge_mask(
    shape: Tuple[int, int, int],
    angle: float,
    taper_width: float = 0.0,
    zero_dc: bool = True,
) -> np.ndarray:
    """Build a missing-wedge *keep* mask for an ``fftshift``-ed 3D spectrum.

    ``angle`` is the half-opening of the retained double wedge, measured from the
    first axis in the (axis 0, axis 2) plane: ``90`` keeps everything and ``0``
    keeps (almost) nothing. Axis 1 is the tilt axis - the mask is constant along
    it, which is what makes this a wedge rather than a cone.

    Parameters
    ----------
    shape : Tuple[int, int, int]
        Shape of the spectrum.
    angle : float
        Half-opening angle in degrees of the region to keep.
    taper_width : float, optional
        Full width in degrees of a raised-cosine ramp straddling the wedge
        boundary. ``0`` (the default) reproduces the reference's hard binary
        mask exactly. A hard edge causes Gibbs ringing along the wedge boundary,
        which is a shortcut feature a picking model can latch onto, so a small
        non-zero taper is preferable once port equivalence has been established.
    zero_dc : bool, optional
        Whether the central (DC) coefficient is removed. ``True`` is the
        reference behaviour.

    Returns
    -------
    np.ndarray
        Boolean keep mask when ``taper_width == 0``, otherwise a float32 keep
        weight in ``[0, 1]``.

    Notes
    -----
    The hard-mask branch is a slope-based construction on the index grid, copied
    from the reference deliberately unaltered: it is equivalent to an angular
    threshold only because the spectrum is ``fftshift``-ed first, and the port
    equivalence test compares against the original.

    The reference sets the centre voxel of the *removal* mask to 1 immediately
    before inverting it, so the DC coefficient ends up zeroed rather than kept.
    That reads like an inverted intent, but it is harmless here because the
    output is re-normalized to zero mean afterwards, so the DC term is discarded
    either way. It is preserved under ``zero_dc=True`` to keep the port faithful.
    """
    # Create the 3D coordinate grid
    x_coords, _, z_coords = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    x_center = np.mean(x_coords)
    z_center = np.mean(z_coords)

    if taper_width > 0:
        # Soft angular ramp. atan2 of the distances from the centre reproduces
        # the same boundary as the slope construction below, but continuously.
        phi = np.degrees(
            np.arctan2(np.abs(z_coords - z_center), np.abs(x_coords - x_center))
        )
        inner = max(angle - taper_width / 2.0, 0.0)
        outer = angle + taper_width / 2.0
        keep = np.clip((outer - phi) / (outer - inner), 0.0, 1.0)
        keep = 0.5 * (1.0 - np.cos(np.pi * keep))
        keep = keep.astype("f4")
        if zero_dc:
            keep[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 0.0
        return keep

    angle_radians = np.radians(angle)
    m = np.tan(angle_radians)  # slope
    b = z_center  # intercept

    def above_wedge(x):
        return m * x + b

    x_left_side = x_coords[: int(x_coords.shape[0] / 2), :, :]
    x_right_side = x_coords[int(x_coords.shape[0] / 2) :, :, :]

    x_right_above_wedge = above_wedge(x_right_side - x_center)
    x_left_above_wedge = above_wedge(x_center - x_left_side)
    x_right_below_wedge = above_wedge(x_center - x_right_side)
    x_left_below_wedge = above_wedge(x_left_side - x_center)

    x_above_wedge = np.concatenate((x_left_above_wedge, x_right_above_wedge), axis=0)
    x_below_wedge = np.concatenate((x_left_below_wedge, x_right_below_wedge), axis=0)
    above_wedge_mask = x_above_wedge < z_coords
    below_wedge_mask = x_below_wedge > z_coords

    removal_mask = above_wedge_mask + below_wedge_mask
    removal_mask = removal_mask > 0
    if zero_dc:
        # Reference indexes the middle axis with ``shape[2]``; corrected to
        # ``shape[1]`` here. Identical for cubic patches such as CHUNK_SIZE=64.
        removal_mask[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1
    return ~removal_mask


def get_line_plot(
    n_points: int, smooth_sigma: float, step_sigma: float, offset: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize a fake radially averaged amplitude spectrum.

    A Gaussian-smoothed random walk, offset and rectified. Feeding this through
    :func:`~tomocpt.dataManager.filter_utils.rotational_kernel` yields a 3D
    amplitude envelope standing in for an arbitrary CTF / dose-filter response.

    Parameters
    ----------
    n_points : int
        Number of radial bins.
    smooth_sigma : float
        Standard deviation of the Gaussian smoothing kernel.
    step_sigma : float
        Standard deviation of the random-walk step sizes.
    offset : float
        Offset added before taking the absolute value.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The x values and the corresponding non-negative y values.
    """
    x = np.linspace(0, n_points - 1, n_points)
    y = np.cumsum(np.random.randn(n_points) * step_sigma)
    y_smoothed = gaussian_filter1d(y, sigma=smooth_sigma)
    return x, np.abs(y_smoothed + offset)


def normalize_and_fft_patch(patch: np.ndarray) -> np.ndarray:
    """Rescale a patch to ``[0, 1]`` and return its shifted FFT.

    Parameters
    ----------
    patch : np.ndarray
        Real-valued 2D or 3D patch. Not modified.

    Returns
    -------
    np.ndarray
        Complex ``fftshift``-ed spectrum.

    Notes
    -----
    The reference rescales in place, and its ``patch`` is a view into the input
    tensor, so calling it mutated the caller's data. A copy is taken here.
    """
    patch = np.array(patch, dtype=np.float32, copy=True)
    patch -= patch.min()
    peak = patch.max()
    patch /= peak if peak > 0 else 1.0
    return fft.fftshift(fft.fftn(patch))


def fft_patch_to_real(fft_patch: np.ndarray) -> np.ndarray:
    """Invert a shifted spectrum back to a real-space patch.

    Parameters
    ----------
    fft_patch : np.ndarray
        Complex ``fftshift``-ed spectrum.

    Returns
    -------
    np.ndarray
        Real-space patch.

    Notes
    -----
    The reference undoes the shift with ``fftshift``; the correct inverse is
    ``ifftshift``. The two coincide only when every axis has even length, so the
    reference is silently correct at ``CHUNK_SIZE=64`` and silently wrong
    otherwise.
    """
    fft_patch = fft.ifftshift(fft_patch)
    return np.real(fft.ifftn(fft_patch))


def run_interpolation(
    img: np.ndarray, img_modified: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    """Blend a modified patch back into the original through a soft kernel.

    Parameters
    ----------
    img : np.ndarray
        Original patch.
    img_modified : np.ndarray
        Degraded patch.
    kernel : np.ndarray
        Blend weights in ``[0, 1]``, broadcastable to the patch shape.

    Returns
    -------
    np.ndarray
        ``img`` where the kernel is 0, ``img_modified`` where it is 1.
    """
    return img + kernel * (img_modified - img)


def _default_scale(img_shape: Sequence[int], axis: int) -> float:
    """Sample the production kernel width for one axis.

    This is the call-site value from membrain-seg's augmentation pipeline, not
    the class default in the reference module. It yields a blob comparable to or
    larger than the patch, so the degradation covers most of a 64^3 chunk with a
    gentle falloff. The class default (``np.log(x[y] // 6)``) would give a much
    tighter blob and a materially different augmentation.
    """
    return float(
        np.exp(
            np.random.uniform(
                np.log(img_shape[axis]) * 0.75, np.log(img_shape[axis]) * 1.5
            )
        )
    )


def _sample_range(value: Union[float, Tuple[float, float]]) -> float:
    """Draw from a ``(low, high)`` range, or pass a scalar straight through."""
    if isinstance(value, (tuple, list)):
        return float(np.random.uniform(value[0], value[1]))
    return float(value)


class RandomFourierDegradation(tio.IntensityTransform):
    """Randomly degrade a subvolume in Fourier space, leaving labels untouched.

    Applies a missing wedge and/or an amplitude-spectrum randomization, blended
    back into the original through a soft Gaussian blob. Must run **after** all
    spatial transforms so that the wedge lands at a random orientation relative
    to the specimen while the label stays geometrically correct.

    Parameters
    ----------
    amplitude_aug : bool, optional
        Enable amplitude-spectrum randomization.
    missing_wedge_aug : bool, optional
        Enable the missing wedge.
    smooth_sigma_range : float or Tuple[float, float], optional
        Gaussian smoothing sigma for the synthesized amplitude spectrum.
    step_sigma_range : float or Tuple[float, float], optional
        Random-walk step sigma for the synthesized amplitude spectrum.
    offset_range : float or Tuple[float, float], optional
        Offset for the synthesized amplitude spectrum.
    keep_angle_range : float or Tuple[float, float], optional
        Half-opening angle in degrees of the retained wedge; see
        :func:`wedge_mask`. ``90`` keeps everything, ``0`` keeps nothing.
    missing_wedge_prob : float, optional
        Probability of applying the wedge.
    amplitude_prob : float, optional
        Probability of applying the amplitude randomization.
    sample_kernel_prob : float, optional
        Probability of localizing the degradation when the wedge did *not* fire.
        Whenever the wedge fires the blend is forced on regardless.
    scale : callable or tuple, optional
        Per-axis Gaussian blob width, sampled with ``(img_shape, axis)``.
    loc : callable or tuple, optional
        Per-axis blob centre in units of the image shape.
    taper_width : float, optional
        Raised-cosine taper width in degrees on the wedge boundary; see
        :func:`wedge_mask`. Defaults to ``0``, i.e. exact reference behaviour.
    reclip : bool, optional
        Re-apply tomocpt's hard clip at ``+/- clip_value`` after the transform.
        The reference ends by normalizing to zero mean and unit standard
        deviation, which is a different distribution from the clipped one
        ``robust_normalization`` feeds in.
    clip_value : float, optional
        Clip bound used when ``reclip`` is set.
    **kwargs
        Forwarded to :class:`torchio.IntensityTransform`.

    Notes
    -----
    Randomness is drawn from ``numpy.random`` rather than TorchIO's generator,
    matching the reference; seed ``numpy.random`` to reproduce a run.
    """

    def __init__(
        self,
        amplitude_aug: bool = True,
        missing_wedge_aug: bool = True,
        smooth_sigma_range: Union[float, Tuple[float, float]] = (2.0, 4.0),
        step_sigma_range: Union[float, Tuple[float, float]] = (0.1, 4.0),
        offset_range: Union[float, Tuple[float, float]] = (2.0, 10.0),
        keep_angle_range: Union[float, Tuple[float, float]] = (45.0, 45.0),
        missing_wedge_prob: float = 0.5,
        amplitude_prob: float = 0.5,
        sample_kernel_prob: float = 0.5,
        scale: Union[Callable, Tuple[float, float]] = _default_scale,
        loc: Union[Callable, Tuple[float, float]] = (-0.5, 1.5),
        taper_width: float = 0.0,
        reclip: bool = True,
        clip_value: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude_aug = amplitude_aug
        self.missing_wedge_aug = missing_wedge_aug
        self.smooth_sigma_range = smooth_sigma_range
        self.step_sigma_range = step_sigma_range
        self.offset_range = offset_range
        self.keep_angle_range = keep_angle_range
        self.missing_wedge_prob = missing_wedge_prob
        self.amplitude_prob = amplitude_prob
        self.sample_kernel_prob = sample_kernel_prob
        self.scale = scale
        self.loc = loc
        self.taper_width = taper_width
        self.reclip = reclip
        self.clip_value = clip_value

    def _randomize(self) -> dict:
        """Draw the per-call augmentation parameters.

        Returns
        -------
        dict
            The sampled parameters and the three on/off decisions.

        Notes
        -----
        The reference tests ``smooth_sigma_range`` when deciding how to sample
        ``offset`` and ``missing_angle`` - copy-paste from the block above it.
        Each range is tested against itself here.
        """
        do_mw = bool(np.random.random() < self.missing_wedge_prob)
        do_amp = bool(np.random.random() < self.amplitude_prob)
        do_kernel = bool(np.random.random() < self.sample_kernel_prob)
        # The wedge is always localized: the network must see patches that are
        # partly degraded and partly clean. Deliberate, per the reference.
        if do_mw:
            do_kernel = True
        return {
            "smooth_sigma": _sample_range(self.smooth_sigma_range),
            "step_sigma": _sample_range(self.step_sigma_range),
            "offset": _sample_range(self.offset_range),
            "keep_angle": _sample_range(self.keep_angle_range),
            "do_mw": do_mw,
            "do_amp": do_amp,
            "do_kernel": do_kernel,
        }

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Degrade every scalar image in the subject; skip every label map."""
        if not (self.missing_wedge_aug or self.amplitude_aug):
            return subject

        params = self._randomize()
        if not (params["do_mw"] or params["do_amp"]):
            return subject

        for image in self.get_images(subject):
            data = image.data
            out = data.clone()
            for c in range(data.shape[0]):
                patch = np.asarray(data[c], dtype=np.float32)
                out[c] = torch.from_numpy(self._degrade_patch(patch, params))
            image.set_data(out)
        return subject

    def _degrade_patch(self, patch: np.ndarray, params: dict) -> np.ndarray:
        """Apply the Fourier degradation to a single 3D patch."""
        fft_patch = normalize_and_fft_patch(patch)

        if self.amplitude_aug and params["do_amp"]:
            _, fake_spectrum = get_line_plot(
                n_points=int(patch.shape[0] / 2),
                smooth_sigma=params["smooth_sigma"],
                step_sigma=params["step_sigma"],
                offset=params["offset"],
            )
            fft_patch = fft_patch * rotational_kernel(fake_spectrum, patch.shape)

        if self.missing_wedge_aug and params["do_mw"]:
            keep = wedge_mask(
                patch.shape, params["keep_angle"], taper_width=self.taper_width
            )
            if keep.dtype == bool:
                fft_patch[~keep] = 0.0
            else:
                fft_patch = fft_patch * keep

        real_patch = _standardize(fft_patch_to_real(fft_patch))

        if params["do_kernel"]:
            kernel = generate_gaussian_kernel(patch.shape, self.scale, self.loc)
            real_patch = run_interpolation(_standardize(patch), real_patch, kernel)

        real_patch = _standardize(real_patch)
        if self.reclip:
            real_patch = np.clip(real_patch, -self.clip_value, self.clip_value)
        return np.ascontiguousarray(real_patch, dtype=np.float32)


def _standardize(arr: np.ndarray) -> np.ndarray:
    """Return a zero-mean, unit-standard-deviation copy of ``arr``."""
    arr = np.asarray(arr, dtype=np.float32) - np.mean(arr)
    std = float(np.std(arr))
    return arr / (std if std > 0 else 1.0)
