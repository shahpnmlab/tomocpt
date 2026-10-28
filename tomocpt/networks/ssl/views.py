"""SSL v2 views (Gate 5): scale pairs, noise equalisation, wedge-preserving augmentation,
degradation and the training pair.

Arrays are ``(Z, Y, X)`` numpy float32 volumes; the missing wedge of the data has its tilt
axis along Y. ``rng`` is always a ``numpy.random.Generator``.

Geometry. A view of scale ``s`` around a source centre ``c`` (source voxels) has 64^3 voxels;
view voxel ``i`` samples the source at ``c + (i - 32) * s`` (per axis), so the centre maps to
view coordinate 32 and one view voxel spans ``s`` source voxels (``apix_view = apix_ref * s``).
The view is computed with a separable, Hann-windowed sinc kernel whose cutoff is 0.9 of the
*view* Nyquist frequency: anti-aliased Fourier downsampling with the same final low-pass for
every ``s`` (``s = 1`` is not special), evaluated at the exact (fractional) positions. The
kernel is separable and symmetric, so it multiplies the spectrum by a real, even product
``H(kz) H(ky) H(kx)`` and never moves power into the missing wedge.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

VIEW = 64
HALF = VIEW // 2
#: cutoff of the final low-pass, cycles per view voxel (0.9 of the view Nyquist frequency)
LOWPASS = 0.45
#: half-width of the windowed-sinc kernel, in view voxels
KERNEL_HALF = 8.0
#: Fourier shell (fraction of Nyquist) used to estimate the noise level of a view
NOISE_SHELL = (0.8, 0.9)
#: mask blocks (voxels) and mask ratio range of the reconstruction task
MASK_BLOCK = (8, 16)
MASK_RATIO = (0.15, 0.4)
#: intensity part of D: x -> a x + b + N(0, sigma)
INTENSITY_A = (0.8, 1.2)
INTENSITY_B = (-0.1, 0.1)
NOISE_SIGMA = (0.0, 0.1)

__all__ = ["make_scale_pair", "equalise_noise", "wedge_preserving_augment", "degrade",
           "corresponding_points", "make_training_pair"]


# --------------------------------------------------------------------------------------
# scale pairs
# --------------------------------------------------------------------------------------

def _reflect(idx: np.ndarray, n: int) -> np.ndarray:
    """Symmetric (edge-repeating) reflection of integer indices into ``[0, n)``."""
    m = np.mod(idx, 2 * n)
    return np.where(m < n, m, 2 * n - 1 - m)


def _axis_operator(n_src: int, centre: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(W [64, K], source indices [K])`` of one axis: ``view = W @ source[indices]``."""
    u = centre + (np.arange(VIEW, dtype=np.float64) - HALF) * s
    reach = KERNEL_HALF * s
    k = np.arange(int(math.floor(u[0] - reach)), int(math.ceil(u[-1] + reach)) + 1)
    t = (k[None, :] - u[:, None]) / s  # offset in view voxels
    w = np.where(np.abs(t) < KERNEL_HALF, 0.5 * (1.0 + np.cos(np.pi * t / KERNEL_HALF)), 0.0)
    h = np.sinc(2.0 * LOWPASS * t) * w
    h /= h.sum(axis=1, keepdims=True)  # keep the mean (DC gain 1)
    return h.astype(np.float32), _reflect(k, n_src)


def _resample_view(source: np.ndarray, centre: Sequence[float], s: float) -> np.ndarray:
    ops = [_axis_operator(n, float(c), float(s)) for n, c in zip(source.shape, centre)]
    (wz, iz), (wy, iy), (wx, ix) = ops
    sub = np.asarray(source[np.ix_(iz, iy, ix)], dtype=np.float32)
    out = np.tensordot(wz, sub, axes=(1, 0))      # (64, Ky, Kx)
    out = np.tensordot(out, wy, axes=(1, 1))      # (64, Kx, 64)
    out = np.tensordot(out, wx, axes=(1, 1))      # (64, 64, 64) = (z, y, x)
    return np.ascontiguousarray(out, dtype=np.float32)


def make_scale_pair(source, centre, s_a: float, s_b: float, apix_ref: float, rng=None):
    """Two views of ``source`` around ``centre`` (source voxels, ``(z, y, x)``) at scales
    ``s_a`` and ``s_b`` (``apix_view = apix_ref * s``).

    Returns ``(view_a, view_b, target_log2, coord_maps)`` with ``target_log2 = log2(s_a / s_b)``.
    ``coord_maps`` describes the geometry only (not the content); pass it to
    :func:`corresponding_points`. Source voxels outside the volume are reflected.
    ``rng`` is unused (the views are deterministic); it is accepted for a uniform signature.
    """
    source = np.asarray(source)
    if source.ndim != 3:
        raise ValueError(f"make_scale_pair expects a 3D source, got shape {source.shape}")
    centre = tuple(float(c) for c in centre)
    view_a = _resample_view(source, centre, s_a)
    view_b = _resample_view(source, centre, s_b)
    maps = {"centre": centre, "s_a": float(s_a), "s_b": float(s_b), "apix_ref": float(apix_ref)}
    return view_a, view_b, float(math.log2(float(s_a) / float(s_b))), maps


# --------------------------------------------------------------------------------------
# noise equalisation
# --------------------------------------------------------------------------------------

def _radius_grid(shape) -> np.ndarray:
    """Radial frequency in units of Nyquist, unshifted FFT layout."""
    k = [np.fft.fftfreq(n) for n in shape]
    return np.sqrt(k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2) / 0.5


def _shell_power(F: np.ndarray, shell: np.ndarray, n: int) -> float:
    return float(np.mean(np.abs(F[shell]) ** 2) / n)


def _support(F: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Where a view holds data in Fourier space (~1) versus a missing region such as a wedge (0):
    power over its radial mean, clipped to ``[0, 1]``, per Fourier voxel. Not smoothed: at small
    radii a smoothing kernel would blur the wedge and fill it at low frequencies, which would
    raise the low-frequency power of the view that receives the noise (a scale cue)."""
    P = np.abs(F) ** 2
    bins = np.minimum((r * F.shape[0] / 2.0).astype(np.int64), int(F.shape[0]))
    radial = np.bincount(bins.ravel(), weights=P.ravel()) / np.maximum(np.bincount(bins.ravel()), 1)
    ref = radial[bins]
    return np.clip(P / np.maximum(ref, 1e-30), 0.0, 1.0)


def equalise_noise(view_a, view_b, rng: Optional[np.random.Generator] = None):
    """Add noise to the quieter view until the noise std of both views, estimated from the
    Fourier shell 0.8-0.9 Nyquist, matches.

    The added noise is white noise passed through the same final low-pass as the views (an
    ``s = 1`` resampling) and restricted to the Fourier support of the quieter view, so it
    does not fill that view's missing wedge (a filled wedge would tell the coarser view
    apart). Returns ``(view_a, view_b)``.
    """
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2 ** 31))
    a = np.asarray(view_a, dtype=np.float32)
    b = np.asarray(view_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"views differ in shape: {a.shape} vs {b.shape}")
    r = _radius_grid(a.shape)
    shell = (r >= NOISE_SHELL[0]) & (r <= NOISE_SHELL[1])
    n = a.size
    Fa, Fb = np.fft.fftn(a - a.mean()), np.fft.fftn(b - b.mean())
    pa, pb = _shell_power(Fa, shell, n), _shell_power(Fb, shell, n)
    # Always draw the noise, so that the random stream does not depend on the content. The
    # white cube has a margin of the kernel reach, so the low-pass never reflects at its border
    # (reflected, mirror-correlated borders would mark the view that received the noise).
    margin = int(math.ceil(KERNEL_HALF)) + 1
    white = rng.standard_normal(tuple(n + 2 * margin for n in a.shape)).astype(np.float32)
    if pa == pb:
        return a.copy(), b.copy()
    quiet_is_a = pa < pb
    Fq = Fa if quiet_is_a else Fb
    deficit = abs(pa - pb)
    noise = _resample_view(white, tuple(margin + x // 2 for x in a.shape), 1.0)  # same low-pass as the views
    G = np.fft.fftn(noise) * np.sqrt(_support(Fq, r))
    got = _shell_power(G, shell, n)
    if got <= 0:
        return a.copy(), b.copy()
    noise = np.real(np.fft.ifftn(G * math.sqrt(deficit / got))).astype(np.float32)
    if quiet_is_a:
        return a + noise, b.copy()
    return a.copy(), b + noise


# --------------------------------------------------------------------------------------
# flips, degradation, correspondences
# --------------------------------------------------------------------------------------

def _flip_axes(flips) -> Tuple[int, ...]:
    return tuple(a for a in range(3) if flips[a])


def wedge_preserving_augment(view, target, rng: np.random.Generator):
    """Random flip of each axis (probability 0.5), applied to the view and its target.

    Flips keep the double wedge (tilt axis Y) in place; no other rotation is used.
    Returns ``(view, target, flips)``, ``flips`` = three bools (z, y, x)."""
    flips = tuple(bool(rng.random() < 0.5) for _ in range(3))
    axes = _flip_axes(flips)
    view = np.asarray(view)
    target = np.asarray(target)
    if axes:
        view = np.flip(view, axes)
        target = np.flip(target, axes)
    return np.ascontiguousarray(view), np.ascontiguousarray(target), flips


def degrade(view, ssl_cfg, rng: np.random.Generator):
    """The Fourier part of the degradation ``D``: missing wedge (``keep_angle`` uniform in
    ``[keep_angle_min, keep_angle_max]``, oriented for ``(z, y, x)``) and amplitude-spectrum
    randomisation, each applied with probability ``degradation_p`` (and enabled by
    ``use_mw_aug`` / ``use_fourier_aug``), through
    ``RandomFourierDegradation._degrade_patch``. Keeps the view's mean and std.

    Every random number comes from ``rng`` (the global ``numpy.random`` state, which
    ``_degrade_patch`` draws from, is seeded from ``rng`` and restored afterwards)."""
    from tomocpt.dataManager.fourier_augmentations import RandomFourierDegradation

    view = np.asarray(view, dtype=np.float32)
    tf = RandomFourierDegradation(
        amplitude_aug=bool(ssl_cfg.use_fourier_aug), missing_wedge_aug=bool(ssl_cfg.use_mw_aug),
        keep_angle_range=(float(ssl_cfg.keep_angle_min), float(ssl_cfg.keep_angle_max)),
        array_order="zyx", match_input_stats=True, reclip=False,
    )
    p = float(ssl_cfg.degradation_p)
    # Fixed number of draws, whatever the outcome.
    u_mw, u_amp, u_kernel = rng.random(3)
    params = {
        "smooth_sigma": float(rng.uniform(*tf.smooth_sigma_range)),
        "step_sigma": float(rng.uniform(*tf.step_sigma_range)),
        "offset": float(rng.uniform(*tf.offset_range)),
        "keep_angle": float(rng.uniform(*tf.keep_angle_range)),
    }
    seed = int(rng.integers(0, 2 ** 32))
    do_mw = bool(ssl_cfg.use_mw_aug) and u_mw < p
    do_amp = bool(ssl_cfg.use_fourier_aug) and u_amp < p
    if not (do_mw or do_amp):
        return view.copy()
    # As in RandomFourierDegradation: a wedge is always blended locally.
    params.update(do_mw=do_mw, do_amp=do_amp, do_kernel=bool(do_mw or u_kernel < tf.sample_kernel_prob))
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return tf._degrade_patch(view, params)
    finally:
        np.random.set_state(state)


def corresponding_points(n: int, coord_maps: Dict, flips_a, flips_b, rng: np.random.Generator):
    """``n`` source points inside the region both views cover, mapped to each view.

    Returns ``(pts_a, pts_b)``, float32 ``[n, 3]``, ``(z, y, x)`` in view voxel coordinates
    (voxel ``i`` at ``i``) after the flips."""
    c = np.asarray(coord_maps["centre"], dtype=np.float64)
    s_a, s_b = float(coord_maps["s_a"]), float(coord_maps["s_b"])
    s_min = min(s_a, s_b)
    lo = c - HALF * s_min
    hi = c + (VIEW - 1 - HALF) * s_min
    u = rng.uniform(lo, hi, size=(int(n), 3))
    out = []
    for s, flips in ((s_a, flips_a), (s_b, flips_b)):
        p = (u - c) / s + HALF
        for axis in _flip_axes(flips):
            p[:, axis] = (VIEW - 1) - p[:, axis]
        out.append(np.ascontiguousarray(p, dtype=np.float32))
    return out[0], out[1]


# --------------------------------------------------------------------------------------
# training pair
# --------------------------------------------------------------------------------------

def _random_mask(rng: np.random.Generator, shape=(VIEW,) * 3) -> np.ndarray:
    """Blocks of 8-16 voxels until the masked fraction reaches ``U(0.15, 0.4)``. True = masked."""
    ratio = rng.uniform(*MASK_RATIO)
    mask = np.zeros(shape, dtype=bool)
    target = ratio * mask.size
    count = 0
    while count < target:
        size = rng.integers(MASK_BLOCK[0], MASK_BLOCK[1] + 1, size=3)
        start = [int(rng.integers(0, n - b + 1)) for n, b in zip(shape, size)]
        sl = tuple(slice(st, st + int(b)) for st, b in zip(start, size))
        mask[sl] = True
        count = int(mask.sum())
    return mask


def _robust_affine(v: np.ndarray) -> Tuple[float, float]:
    """``(median, p95 - p5)`` of the robust normalisation."""
    p5, med, p95 = np.quantile(v, [0.05, 0.5, 0.95])
    scale = float(p95 - p5)
    return float(med), (scale if scale > 0 else 1.0)


def _resolve(ssl_cfg):
    fn = getattr(ssl_cfg, "resolved", None)
    return fn() if callable(fn) else ssl_cfg


def _draw_centre(shape, s_max: float, rng: np.random.Generator) -> Tuple[float, ...]:
    centre = []
    for n in shape:
        lo, hi = HALF * s_max, n - HALF * s_max
        u = rng.random()
        centre.append(float(lo + u * (hi - lo)) if hi > lo else n / 2.0)
    return tuple(centre)


_WORKER_THREADS_LIMITED = False


def _limit_threads_in_worker() -> None:
    """In a DataLoader worker, use one BLAS thread: many workers each running a full-width
    OpenBLAS pool oversubscribe the CPU and starve the training loop."""
    global _WORKER_THREADS_LIMITED
    if _WORKER_THREADS_LIMITED:
        return
    import torch.utils.data

    if torch.utils.data.get_worker_info() is None:
        return
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # threadpoolctl missing or no supported pool: keep the defaults
        pass
    _WORKER_THREADS_LIMITED = True


def make_training_pair(source, apix_ref: float, ssl_cfg, rng: np.random.Generator,
                       centre: Optional[Sequence[float]] = None, s_a: Optional[float] = None,
                       s_b: Optional[float] = None) -> dict:
    """One SSL v2 training item from a source chunk (``apix_ref`` Å per source voxel).

    Order per view: crop and Fourier-resample -> noise equalisation (both views) -> flips
    (view and target) -> polarity (view and target) -> D (view only: Fourier degradation, then
    ``a x + b`` and Gaussian noise) -> robust normalisation (affine from the clean view, shared
    by input and target) -> masking (input only, fill 0).

    The random draws of D and of the mask come from child generators seeded at a fixed point,
    so D never shifts the draws that build the views and targets.

    Returns ``input_*``, ``target_*`` (float32 ``[1, 64, 64, 64]``), ``mask_*`` (bool, True =
    masked), ``pts_*`` (float32 ``[n_anchors, 3]``), ``target_log2``, ``apix_a``, ``apix_b``
    (float32 scalars), ``flips_*`` (int64 ``[3]``).
    """
    _limit_threads_in_worker()
    cfg = _resolve(ssl_cfg)
    source = np.asarray(source)
    seeds = rng.integers(0, 2 ** 63 - 1, size=2)
    rng_d = np.random.default_rng(int(seeds[0]))
    rng_m = np.random.default_rng(int(seeds[1]))

    log_max = float(cfg.scale_range_log2) * math.log(2.0)
    u = rng.random(2)
    s_a = float(math.exp(u[0] * log_max)) if s_a is None else float(s_a)
    s_b = float(math.exp(u[1] * log_max)) if s_b is None else float(s_b)
    c_draw = _draw_centre(source.shape, max(s_a, s_b), rng)
    centre = c_draw if centre is None else tuple(float(c) for c in centre)

    view_a, view_b, target_log2, maps = make_scale_pair(source, centre, s_a, s_b, apix_ref, rng)
    view_a, view_b = equalise_noise(view_a, view_b, rng)

    out = {}
    flips = {}
    for name, view in (("a", view_a), ("b", view_b)):
        view, target, fl = wedge_preserving_augment(view, view.copy(), rng)
        if rng.random() < float(cfg.inversion_p):
            view, target = -view, -target
        flips[name] = fl
        # D (input only), from its own generator.
        x = degrade(view, cfg, rng_d)
        a = rng_d.uniform(*INTENSITY_A)
        b = rng_d.uniform(*INTENSITY_B)
        sigma = rng_d.uniform(*NOISE_SIGMA)
        x = a * x + b + sigma * rng_d.standard_normal(x.shape).astype(np.float32)
        # One affine normalisation, computed on the clean view, for input and target.
        med, scale = _robust_affine(target)
        target = np.clip((target - med) / scale, -3.0, 3.0).astype(np.float32)
        x = np.clip((x - med) / scale, -3.0, 3.0).astype(np.float32)
        mask = _random_mask(rng_m)
        x[mask] = 0.0
        out[f"input_{name}"] = np.ascontiguousarray(x[None], dtype=np.float32)
        out[f"target_{name}"] = np.ascontiguousarray(target[None], dtype=np.float32)
        out[f"mask_{name}"] = np.ascontiguousarray(mask[None])
        out[f"flips_{name}"] = np.asarray(fl, dtype=np.int64)

    pts_a, pts_b = corresponding_points(int(cfg.n_anchors), maps, flips["a"], flips["b"], rng)
    out.update(
        pts_a=pts_a, pts_b=pts_b,
        target_log2=np.float32(target_log2),
        apix_a=np.float32(apix_ref * s_a), apix_b=np.float32(apix_ref * s_b),
    )
    return out
