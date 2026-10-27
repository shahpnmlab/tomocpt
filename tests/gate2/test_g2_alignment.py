"""G2-C20: target alignment after resampling (exact coordinate map).

``working_shape`` truncates (``int(N * f)``), so the resampled volume's real scale is
``N_out / N_in`` per axis, not ``apix_native / apix_work``. Points must follow the map of the
resample that was run (``scaling.coordinate_map``), and zoom augmentation must map points with
the map of the zoom that was run.

Dataset for (b) and (c): native (64, 128, 128) at 7 A/px. Class 1 (300 A, one point, no
density) sets apix_work = 300/32 = 9.375 A, working shape (47, 95, 95): N_out/N_in = 0.734,
0.742, 0.742 against f = 0.7467 (the approximate map is off by up to ~0.8 voxel at the far
edge). Class 2 (100 A, d_vox 10.7) marks 27 small Gaussian blobs (sigma 1.2 native voxels,
cosine-tapered to exactly 0 at r = 3.5) on a 3 x 3 x 3 grid of positions. Z positions include
60 (3 voxels from the far edge; Z is padded 47 -> 64, so the default "skip" border policy keeps
it). Y/X far positions are 118 (10 voxels from the edge, the closest "skip" keeps for d_vox 10.7).
Fewer than 5 % of voxels are non-zero, so robust normalisation takes its ``data - mean`` branch
(no clipping) and the background is a constant.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gate2 import g2_helpers as H
from utils.tiny import applied_main_config

NATIVE = (64, 128, 128)
APIX = 7.0
APIX_WORK = 9.375
DIAM = {1: 300.0, 2: 100.0}
ZS, YS, XS = (10.0, 32.3, 60.0), (12.0, 64.6, 118.0), (14.0, 70.2, 118.0)
BLOBS_ZYX = np.array([(z, y, x) for z in ZS for y in YS for x in XS])
SIGMA, TAPER = 1.2, 3.5


def _blob_volume(shape=NATIVE, centres=BLOBS_ZYX):
    vol = np.zeros(shape, dtype=np.float64)
    for c in centres:
        lo = [max(0, int(np.floor(v - TAPER)) - 1) for v in c]
        hi = [min(s, int(np.ceil(v + TAPER)) + 2) for v, s in zip(c, shape)]
        zz, yy, xx = np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij")
        r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
        w = np.where(r < TAPER, 0.5 * (1 + np.cos(np.pi * r / TAPER)), 0.0)
        vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += np.exp(-r ** 2 / (2 * SIGMA ** 2)) * w
    return vol.astype(np.float32)


def _ref_map(n_in, n_out, method):
    scale = np.asarray(n_out, dtype=np.float64) / np.asarray(n_in, dtype=np.float64)
    if method == "trilinear":
        return scale, 0.5 * scale - 0.5
    return scale, np.zeros(3)


def _centroid(vol, around, half, bg, sign=1.0):
    c = np.round(np.asarray(around)).astype(int)
    lo = np.maximum(c - half, 0)
    hi = np.minimum(c + half + 1, vol.shape)
    sub = (vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64) - bg) * sign
    grids = np.meshgrid(*[np.arange(a, b, dtype=np.float64) for a, b in zip(lo, hi)], indexing="ij")
    return np.array([(g * sub).sum() / sub.sum() for g in grids])


# ------------------------------------------------------------------------------------------
# (a) coordinate_map
# ------------------------------------------------------------------------------------------

def test_G2_C20_coordinate_map_values():
    from tomocpt.dataManager.scaling import coordinate_map
    from tomocpt.defaultConfigs.scale_config import ResampleMethod

    n_in, n_out = (64, 128, 127), (47, 95, 94)
    scale = np.array(n_out, dtype=np.float64) / np.array(n_in, dtype=np.float64)
    s, o = coordinate_map(n_in, n_out, ResampleMethod.fourier)
    np.testing.assert_allclose(np.asarray(s, dtype=np.float64), scale, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(o, dtype=np.float64), 0.0, atol=1e-12)
    s, o = coordinate_map(n_in, n_out, ResampleMethod.trilinear)
    np.testing.assert_allclose(np.asarray(s, dtype=np.float64), scale, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(o, dtype=np.float64), 0.5 * scale - 0.5, rtol=1e-12, atol=1e-12)
    for method in (ResampleMethod.fourier, ResampleMethod.trilinear):
        s, o = coordinate_map(n_in, n_in, method)
        assert len(s) == len(o) == 3
        np.testing.assert_allclose(np.asarray(s, dtype=np.float64), 1.0, atol=0)
        np.testing.assert_allclose(np.asarray(o, dtype=np.float64), 0.0, atol=0)


def test_G2_C20_points_to_working_with_coord_map():
    import pandas as pd

    from tomocpt.labels.helpers import points_to_working
    from tomocpt.labels.multiclass import build_class_table

    table = build_class_table(DIAM, {1: "big", 2: "blob"})
    df = pd.DataFrame({"x": [118.0, 14.0], "y": [64.6, 12.0], "z": [60.0, 10.0], "class_number": [2, 1],
                       "diameter_ang": [100.0, 300.0]})
    cmap = (np.array([0.5, 0.75, 0.8]), np.array([0.1, -0.2, 0.3]))
    p = np.asarray(points_to_working(df, APIX, APIX_WORK, table, coord_map=cmap), dtype=np.float64)
    np.testing.assert_allclose(p[0, :3], [60.0 * 0.5 + 0.1, 64.6 * 0.75 - 0.2, 118.0 * 0.8 + 0.3], rtol=1e-6)
    np.testing.assert_allclose(p[:, 3], [1, 0])
    np.testing.assert_allclose(p[:, 4], [100.0 / APIX_WORK, 300.0 / APIX_WORK], rtol=1e-6)
    q = np.asarray(points_to_working(df, APIX, APIX_WORK, table), dtype=np.float64)
    np.testing.assert_allclose(q[0, :3], np.array([60.0, 64.6, 118.0]) * APIX / APIX_WORK, rtol=1e-6)


# ------------------------------------------------------------------------------------------
# (b) stored points vs resampled blob centroids
# ------------------------------------------------------------------------------------------

def _inputs(root):
    from tomocpt.labels.multiclass import build_class_table

    vol = _blob_volume()
    assert (vol > 0).mean() < 0.05
    big = {1: np.array([[64.0, 64.0, 32.0]])}  # (x, y, z); no density, sets apix_work only
    coords = dict(big)
    coords[2] = BLOBS_ZYX[:, ::-1].copy()
    tomos = [dict(name=f"TS_{i + 1:02d}", volume=vol, apix=APIX, shape=NATIVE, coords=coords) for i in range(2)]
    df = H.write_v2_inputs(root, tomos, DIAM)
    return tomos, df, build_class_table(DIAM, {1: "big", 2: "blob"})


@pytest.fixture(scope="module")
def aligned_chunks(tmp_path_factory):
    from tomocpt.defaultConfigs.scale_config import ResampleMethod

    root = tmp_path_factory.mktemp("g2_c20")
    tomos, df, table = _inputs(root)
    out = {}
    with applied_main_config(None):
        for method in ("fourier", "trilinear"):
            out[method] = H.run_v2_chunking(df, root / f"chunks_{method}", table,
                                            resample=ResampleMethod(method))
    return dict(tomos=tomos, df=df, table=table, chunks=out)


def _working_volume(method):
    """Normalised (v1), resampled with resample_volume, padded as v1: the volume the chunker cuts."""
    from tomocpt.dataManager.preprocessing import robust_normalization, symmetrize_and_pad
    from tomocpt.dataManager.scaling import resample_volume, working_shape
    from tomocpt.defaultConfigs.scale_config import ResampleMethod

    ws = tuple(int(s) for s in working_shape(NATIVE, APIX, APIX_WORK))
    norm = robust_normalization(_blob_volume())
    w = resample_volume(torch.from_numpy(norm), ws, ResampleMethod(method))
    w, _ = symmetrize_and_pad(w, 64)
    pad = np.array([H.v1_pad_front(s) for s in ws], dtype=np.float64)
    return w.numpy().astype(np.float64), ws, pad


@pytest.mark.parametrize("method, tol", [("fourier", 0.10), ("trilinear", 0.15)])
def test_G2_C20_stored_points_sit_on_resampled_blobs(aligned_chunks, method, tol):
    from tomocpt.dataManager.scaling import working_shape

    ws = tuple(int(s) for s in working_shape(NATIVE, APIX, APIX_WORK))
    assert ws == (47, 95, 95)
    for n_in, n_out in zip(NATIVE, ws):
        assert abs(n_out / n_in - APIX / APIX_WORK) > 1e-3

    vol, ws, pad = _working_volume(method)
    bg = float(np.median(vol))
    scale, offset = _ref_map(NATIVE, ws, method)
    centroids = np.array([_centroid(vol, c * scale + offset + pad, 4, bg) for c in BLOBS_ZYX])
    # the approximate map misses the far blobs by more than the tolerance (the test can tell)
    approx = BLOBS_ZYX * (APIX / APIX_WORK) + pad
    assert np.abs(approx - centroids).max() > 3 * tol

    chunks = aligned_chunks["chunks"][method]
    starts = [H.grid_starts(s) for s in vol.shape]
    n_checked, far_checked = 0, set()
    for d in H.chunk_dirs(chunks):
        for f in H.points_files(d):
            p = np.load(f).astype(np.float64)
            p = p[p[:, 3] == 1]          # blob class
            if not len(p):
                continue
            best = (np.inf, None, None)
            for oz in starts[0]:
                for oy in starts[1]:
                    for ox in starts[2]:
                        q = p[:, :3] + np.array([oz, oy, ox])
                        dist = np.linalg.norm(q[:, None, :] - centroids[None, :, :], axis=2)
                        err = dist.min(axis=1).max()
                        if err < best[0]:
                            best = (err, q, dist.argmin(axis=1))
            err, q, idx = best
            assert err <= tol, (f, err)
            n_checked += len(q)
            far_checked |= {int(i) for i in idx if BLOBS_ZYX[i, 0] == 60.0 or BLOBS_ZYX[i, 1] == 118.0
                            or BLOBS_ZYX[i, 2] == 118.0}
    assert n_checked >= 27
    assert len(far_checked) >= 10


# ------------------------------------------------------------------------------------------
# (c) zoom
# ------------------------------------------------------------------------------------------

def test_G2_C20_zoomed_target_peak_sits_on_the_zoomed_blob(aligned_chunks):
    """zoom_aug_max=1.5, spatial augmentation off. There is no public way to fix s, so s is read
    per sample as apix_in / apix_work (as in C16); for every drawn s, each interior blob-class
    target peak (centroid of the target around its local maximum) lies within 0.15 voxel of the
    centroid of the zoomed blob in input_data. Contrast inversion is handled by the sign of
    the blob."""
    from tomocpt.defaultConfigs.scale_config import ScaleConfig, ResampleMethod

    chunks = aligned_chunks["chunks"]["fourier"]
    data = H.make_data(chunks, augmentation=H.no_spatial_aug(),
                       scale_cfg=ScaleConfig(zoom_aug_max=1.5, resample=ResampleMethod.fourier), batch_size=1)
    n, s_seen = 0, []
    for _ in range(4):
        for b in data.train_dataloader():
            s = APIX_WORK / float(b["apix_work"][0, 0])
            s_seen.append(s)
            t = b["target_data"][0, 1].numpy().astype(np.float64)
            v = b["input_data"][0, 0].numpy().astype(np.float64)
            bg = float(np.median(v))
            R = float(b["class_d_vox"][0, 1]) / 2.0
            half_t, half_v = int(np.ceil(R)) + 1, int(np.ceil(TAPER * 0.75 * s)) + 2
            for flat in np.argsort(t.ravel())[::-1][:400]:
                peak = np.array(np.unravel_index(flat, t.shape))
                if t[tuple(peak)] < 0.9:
                    break
                half = max(half_t, half_v)
                if np.any(peak - half < 0) or np.any(peak + half >= 64):
                    continue
                lo, hi = peak - 1, peak + 2
                if t[tuple(peak)] < t[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].max():
                    continue
                ct = _centroid(t, peak, half_t, 0.0)
                sign = 1.0 if v[tuple(peak)] > bg else -1.0
                cv = _centroid(v, ct, half_v, bg, sign)
                assert np.linalg.norm(ct - cv) <= 0.15, (s, ct, cv)
                n += 1
    assert max(s_seen) > 1.05
    assert n >= 5
