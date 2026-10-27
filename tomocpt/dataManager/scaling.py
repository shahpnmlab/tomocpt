"""Voxel size and scale rule of data pipeline v2 (CLAUDE.md Section 3).

Every function takes its values as arguments (invariant 3): nothing here reads
``mainConfig``. Gate 4 uses :func:`plan_levels` to plan inference.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import mrcfile
import torch

from tomocpt.defaultConfigs.scale_config import ResampleMethod
from tomocpt.logger import get_logger

logging = get_logger()

#: Relative tolerance of the band comparisons (spec: 1e-6).
BAND_RTOL = 1e-6


def read_apix(mrc_path: Union[str, Path], override: Optional[float] = None) -> float:
    """Voxel size in A of an MRC file (``mrc.voxel_size.x``), or ``override`` when given.

    Raises ValueError (naming ``apix_override``) if the header value is <= 0. Logs a warning
    if it is exactly 1.0, which is often an unset header.
    """
    if override is not None:
        override = float(override)
        if not override > 0:
            raise ValueError(f"apix_override must be > 0, got {override}")
        return override
    with mrcfile.open(str(mrc_path), permissive=True, header_only=True) as mrc:
        apix = float(mrc.voxel_size.x)
    if not apix > 0:
        raise ValueError(
            f"The MRC header of {mrc_path} gives a voxel size of {apix} A. Set apix_override "
            "to the correct voxel size in A."
        )
    if apix == 1.0:
        logging.warning(
            f"The MRC header of {mrc_path} gives a voxel size of exactly 1.0 A, which is often an "
            "unset header. Set apix_override if this is wrong."
        )
    return apix


def _l_eff(length: float, apix_native: float, band_min: float) -> float:
    return max(float(length), float(apix_native) * float(band_min))


def _band_rule(apix_native: float, lengths: Sequence[float], band_max: float) -> float:
    return max(float(apix_native), max(float(x) for x in lengths) / float(band_max))


def _group_indices(apix_native: float, lengths: Sequence[float], band_min: float,
                   band_max: float) -> List[List[int]]:
    """Greedy grouping of ``lengths`` (indices), sorted by L_eff (spec T2.1)."""
    if len(lengths) == 0:
        raise ValueError("At least one class diameter is needed to choose a working voxel size")
    for x in lengths:
        if not float(x) > 0:
            raise ValueError(f"Class diameters must be > 0, got {x}")
    if not (band_min > 0 and band_max >= band_min):
        raise ValueError(f"Invalid band [{band_min}, {band_max}]")
    ratio = float(band_max) / float(band_min)
    order = sorted(range(len(lengths)), key=lambda i: (_l_eff(lengths[i], apix_native, band_min), i))
    groups: List[List[int]] = []
    current: List[int] = []
    for i in order:
        if current:
            first = _l_eff(lengths[current[0]], apix_native, band_min)
            if _l_eff(lengths[i], apix_native, band_min) / first > ratio * (1.0 + 1e-12):
                groups.append(current)
                current = []
        current.append(i)
    groups.append(current)
    return groups


def group_lengths(apix_native: float, lengths_ang: Sequence[float], band_min: float,
                  band_max: float) -> List[Tuple[float, List[float]]]:
    """Split class diameters into groups that each fit the band at one working voxel size.

    Sorted by ``L_eff = max(L, apix_native * band_min)``; a group grows while
    ``max(L_eff) / min(L_eff) <= band_max / band_min``. Returns ``(apix_work, lengths)`` per
    group (original lengths, sorted), with ``apix_work`` from the band rule.
    """
    out = []
    for g in _group_indices(apix_native, lengths_ang, band_min, band_max):
        lengths = sorted(float(lengths_ang[i]) for i in g)
        out.append((_band_rule(apix_native, lengths, band_max), lengths))
    return out


def choose_working_apix(apix_native: float, lengths_ang: Sequence[float], band_min: float,
                        band_max: float) -> float:
    """Band rule: ``max(apix_native, max(L) / band_max)``, never finer than native.

    Raises ValueError if no single voxel size keeps every class inside the band (or below it
    at native size); :func:`group_lengths` then gives the groups.
    """
    groups = group_lengths(apix_native, lengths_ang, band_min, band_max)
    if len(groups) > 1:
        raise ValueError(
            f"No single working voxel size fits class diameters {sorted(float(x) for x in lengths_ang)} A "
            f"at {apix_native} A/px in the band [{band_min}, {band_max}] voxels; they need "
            f"{len(groups)} levels (see group_lengths / plan_levels)."
        )
    return groups[0][0]


class Level(NamedTuple):
    """One working voxel size of a tomogram.

    ``assigned``: class indices (class-table rows) read at this level (each class is assigned
    to exactly one level). ``available``: every class with ``lo_c <= d_vox <= band_max`` here.
    """
    apix_work: float
    assigned: List[int]
    available: List[int]


def class_lower_limit(diameter_ang: float, apix_native: float, band_min: float) -> float:
    """``lo_c = min(band_min, diameter / apix_native)`` (CLAUDE.md Section 3)."""
    return min(float(band_min), float(diameter_ang) / float(apix_native))


def in_band(diameter_ang: float, apix_native: float, apix_work: float, band_min: float,
            band_max: float, rtol: float = BAND_RTOL) -> bool:
    """True if ``lo_c <= diameter / apix_work <= band_max`` (relative tolerance ``rtol``)."""
    d = float(diameter_ang) / float(apix_work)
    lo = class_lower_limit(diameter_ang, apix_native, band_min)
    return lo * (1.0 - rtol) <= d <= float(band_max) * (1.0 + rtol)


def _diameter(row) -> float:
    if hasattr(row, "diameter_ang"):
        return float(row.diameter_ang)
    if isinstance(row, dict):
        return float(row["diameter_ang"])
    return float(row)


def plan_levels(apix_native: float, class_table: Sequence, band_min: float, band_max: float) -> List[Level]:
    """One :class:`Level` per group of :func:`group_lengths`, sorted by ``apix_work``.

    ``class_table`` rows are ``ClassSpec`` (or anything with ``diameter_ang``, or plain
    diameters). Indices refer to class-table rows.
    """
    diam = [_diameter(r) for r in class_table]
    levels = []
    for g in _group_indices(apix_native, diam, band_min, band_max):
        apix_work = _band_rule(apix_native, [diam[i] for i in g], band_max)
        available = [c for c, d in enumerate(diam) if in_band(d, apix_native, apix_work, band_min, band_max)]
        levels.append(Level(apix_work=float(apix_work), assigned=sorted(g), available=available))
    levels.sort(key=lambda lv: lv.apix_work)
    return levels


def working_shape(native_shape_zyx: Sequence[int], apix_native: float, apix_work: float) -> Tuple[int, int, int]:
    """``int(s * apix_native / apix_work)`` per axis (the v1 rounding)."""
    f = float(apix_native) / float(apix_work)
    return tuple(int(int(s) * f) for s in native_shape_zyx)


def coordinate_map(native_shape_zyx: Sequence[int], work_shape_zyx: Sequence[int],
                   method: Union[ResampleMethod, str] = ResampleMethod.fourier):
    """Exact per-axis map from native to working voxel coordinates of the resample that
    :func:`resample_volume` performs: ``work = native * scale + offset``.

    ``fourier``: ``scale = N_out / N_in``, ``offset = 0`` (voxel 0 stays at the origin).
    ``trilinear`` (``align_corners=False``): ``scale = N_out / N_in``,
    ``offset = 0.5 * scale - 0.5``. Equal shapes (no resample) give scale 1, offset 0.
    Returns ``(scale_zyx, offset_zyx)``, two float64 arrays of length 3.
    """
    import numpy as np

    n_in = np.asarray([int(s) for s in native_shape_zyx], dtype=np.float64)
    n_out = np.asarray([int(s) for s in work_shape_zyx], dtype=np.float64)
    if n_in.shape != (3,) or n_out.shape != (3,):
        raise ValueError(f"coordinate_map expects 3D shapes, got {tuple(native_shape_zyx)} -> {tuple(work_shape_zyx)}")
    if np.array_equal(n_in, n_out):
        return np.ones(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    scale = n_out / n_in
    method = ResampleMethod(getattr(method, "value", method))
    if method == ResampleMethod.trilinear:
        return scale, 0.5 * scale - 0.5
    return scale, np.zeros(3, dtype=np.float64)


def _fourier_resample_axis(vol: torch.Tensor, dim: int, n_out: int) -> torch.Tensor:
    """Crop or zero-pad the spectrum of a real tensor along one axis (scipy.signal.resample
    convention for the Nyquist bin); keeps the mean and voxel 0 at the origin."""
    n_in = vol.shape[dim]
    if n_out == n_in:
        return vol
    spec = torch.fft.rfft(vol, dim=dim)
    n_keep = min(n_in, n_out) // 2 + 1
    spec = spec.narrow(dim, 0, n_keep)
    m = min(n_in, n_out)
    if m % 2 == 0:
        # The bin at m/2 stands for the pair +-m/2.
        idx = [slice(None)] * spec.ndim
        idx[dim] = slice(m // 2, m // 2 + 1)
        spec = spec.clone()
        spec[tuple(idx)] *= 2.0 if n_out < n_in else 0.5
    out = torch.fft.irfft(spec, n=n_out, dim=dim)
    return out * (float(n_out) / float(n_in))


def _fourier_resample(vol: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
    out = vol.to(torch.float32)
    # Shrink the axis that shrinks most first (less memory for the later axes).
    dims = sorted(range(3), key=lambda d: target_shape[d] / out.shape[d])
    for d in dims:
        out = _fourier_resample_axis(out, d, int(target_shape[d]))
    return out.contiguous()


def resample_volume(vol: torch.Tensor, target_shape: Sequence[int],
                    method: Union[ResampleMethod, str] = ResampleMethod.fourier) -> torch.Tensor:
    """Resample a ``[D, H, W]`` tensor to ``target_shape``.

    ``trilinear`` calls the legacy :func:`preprocessing.resize_volume` unchanged. ``fourier``
    crops (anti-aliasing) or zero-pads the centred spectrum, keeps the mean, handles odd and
    even sizes and returns a real tensor. Output voxel ``j`` samples input position
    ``j * N_in / N_out`` (voxel 0 stays at the origin). Runs on the device of ``vol`` and
    falls back to the CPU on CUDA OOM.
    """
    from tomocpt.dataManager.preprocessing import resize_volume

    target_shape = tuple(int(s) for s in target_shape)
    if len(target_shape) != 3 or vol.ndim != 3:
        raise ValueError(f"resample_volume expects a 3D volume and a 3D shape, got {tuple(vol.shape)} -> {target_shape}")
    method = ResampleMethod(getattr(method, "value", method))
    if method == ResampleMethod.trilinear:
        return resize_volume(vol, target_shape)
    if vol.device.type == "cuda":
        try:
            return _fourier_resample(vol, target_shape)
        except torch.cuda.OutOfMemoryError:
            logging.warning("CUDA out of memory during Fourier resampling; falling back to the CPU.")
            gc.collect()
            torch.cuda.empty_cache()
            return _fourier_resample(vol.cpu(), target_shape)
    return _fourier_resample(vol, target_shape)


def jitter_range(apix_native: float, diameters_ang: Sequence[float], band_min: float,
                 band_max: float) -> Tuple[float, float]:
    """Range of working voxel sizes that keeps every class in ``diameters_ang`` within
    ``[lo_c, band_max]`` and is never finer than native. ``lo >= hi`` means empty."""
    lo = max([float(apix_native)] + [float(d) / float(band_max) for d in diameters_ang])
    hi = min(float(d) / class_lower_limit(d, apix_native, band_min) for d in diameters_ang)
    return lo, hi
