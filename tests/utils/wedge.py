"""Missing-wedge helpers for tests (written at Gate 1, frozen afterwards).

Conventions
-----------
* Arrays are ``(Z, Y, X)``.
* Masks are in the **unshifted** ``np.fft.fftn`` layout (DC at index 0); use
  ``np.fft.fftshift`` to view them centred.
* Frequencies are physical (``np.fft.fftfreq``, cycles per voxel, Nyquist = 0.5), so
  the wedge angle is the true angle for cubic voxels even when the array is not a cube.
* ``max_angle_deg`` is the maximum tilt angle of the (simulated) tilt series: a
  coefficient is **measured** when its angle from the in-plane frequency axis is at most
  ``max_angle_deg``. ``max_angle_deg = 90`` measures everything; ``60`` is a typical
  +/-60 degree tilt series and leaves a 60 degree-wide double wedge around ``kz`` empty.

Tilt axes
---------
* ``"y"`` (default, the usual tomogram geometry): the wedge lives in the ``kz-kx`` plane
  and is constant along ``ky``. A coefficient ``(kz, ky, kx)`` is measured iff
  ``atan2(|kz|, |kx|) <= max_angle_deg``.
* ``"x"``: the wedge lives in the ``kz-ky`` plane and is constant along ``kx``.
  Measured iff ``atan2(|kz|, |ky|) <= max_angle_deg``.

The line ``kz = k_inplane = 0`` (which includes the DC term) counts as measured.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["missing_wedge_mask", "apply_missing_wedge", "wedge_power_fraction"]


def _freq_grids(shape: Sequence[int]):
    if len(shape) != 3:
        raise ValueError(f"expected a 3D shape (Z, Y, X), got {tuple(shape)}")
    fz = np.fft.fftfreq(int(shape[0]))[:, None, None]
    fy = np.fft.fftfreq(int(shape[1]))[None, :, None]
    fx = np.fft.fftfreq(int(shape[2]))[None, None, :]
    return fz, fy, fx


def missing_wedge_mask(shape, max_angle_deg: float = 60.0, tilt_axis: str = "y") -> np.ndarray:
    """Boolean mask of the **measured** Fourier coefficients (True = measured, kept).

    The complement (``~mask``) is the missing region. Unshifted FFT layout.
    """
    fz, fy, fx = _freq_grids(shape)
    if tilt_axis == "y":
        inplane = fx
    elif tilt_axis == "x":
        inplane = fy
    else:
        raise ValueError(f"tilt_axis must be 'y' or 'x', got {tilt_axis!r}")
    angle = np.degrees(np.arctan2(np.abs(fz), np.abs(inplane)))  # 0 on the in-plane axis, 90 on kz
    measured = angle <= float(max_angle_deg) + 1e-9
    return np.broadcast_to(measured, tuple(int(s) for s in shape)).copy()


def apply_missing_wedge(vol: np.ndarray, max_angle_deg: float = 60.0, tilt_axis: str = "y") -> np.ndarray:
    """Zero the Fourier coefficients outside the tilt range. Returns a real array of ``vol``'s dtype
    (float32 for integer input)."""
    vol = np.asarray(vol)
    mask = missing_wedge_mask(vol.shape, max_angle_deg=max_angle_deg, tilt_axis=tilt_axis)
    out = np.fft.ifftn(np.fft.fftn(vol) * mask).real
    dtype = vol.dtype if np.issubdtype(vol.dtype, np.floating) else np.float32
    return out.astype(dtype, copy=False)


def wedge_power_fraction(vol: np.ndarray, max_angle_deg: float = 60.0, tilt_axis: str = "y",
                         kmin: float = 0.05) -> float:
    """Fraction of Fourier power inside the **missing** region of the given orientation.

    Coefficients with ``|k| < kmin * Nyquist`` are excluded from both numerator and
    denominator (the low-frequency core is ill-defined in angle). Near 0 for a volume that
    is correctly wedged in that orientation; about the missing-region volume fraction
    (~1/3 for 60 degrees) for a white-noise volume. Returns 0.0 if no power remains.
    """
    vol = np.asarray(vol, dtype=np.float64)
    fz, fy, fx = _freq_grids(vol.shape)
    radius = np.sqrt(fz ** 2 + fy ** 2 + fx ** 2) / 0.5  # in units of Nyquist
    keep = np.broadcast_to(radius >= kmin, vol.shape)
    power = np.abs(np.fft.fftn(vol)) ** 2
    missing = ~missing_wedge_mask(vol.shape, max_angle_deg=max_angle_deg, tilt_axis=tilt_axis)
    total = float(power[keep].sum())
    if total <= 0.0:
        return 0.0
    return float(power[keep & missing].sum() / total)
