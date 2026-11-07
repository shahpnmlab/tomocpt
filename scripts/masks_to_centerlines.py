#!/usr/bin/env python
"""Filament centrelines from binary segmentation masks, by template matching.

Written for DS-10490 (cryoET data portal), whose actin and microtubule masks have no per-filament
identity and touch in bundles and at crossings. Skeletons of such masks merge neighbours and make false
links. Here oriented templates follow each filament straight through: crossing strands differ in
direction, and a traced strand is marked as used, so its neighbour gets its own line.

1. A first skeleton trace gives the radial profile of the mask around an axis. It picks the template: a solid
   cylinder for solid filaments, a ring for tubes whose mask marks only the wall. ``--rmin``, ``--rmax`` override.
2. A bank of oriented templates (FFT convolution, on the GPU if there is one) gives seeds and directions.
3. From the strongest unused seed, a line is walked both ways with limited turning. Where the template leaves
   the volume, and at the filament ends, the line goes straight on to the end of the mask.

Output: one IMOD model per mask, with the mask's stem, one open contour per filament; use the
directory as ``prepData.filament_coords``::

    python scripts/masks_to_centerlines.py --mask_dir data/filaments/10490/masks/actin \\
        --out_dir data/filaments/10490/centerlines/actin --diameter_ang 70 --runs sample_1,sample_2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tomocpt.labels import filaments as LF  # noqa: E402

N_DIRECTIONS = 150
STEP_VOX = 3.0
CONE_DEG = 20.0
LATERAL_VOX = 2
MAX_TEMPLATE_POINTS = 1200
MIN_IN_VOLUME = 0.85  # below this fraction of the template in the volume, the score is not trusted
FINAL_SMOOTHING_ANG = 20.0


def hemisphere(n):
    i = np.arange(n) + 0.5
    z = i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def basis(d):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.8 else np.array([0, 1.0, 0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def template(rmin, rmax, length):
    """Template points ``[N, 3]``: along the axis, then the two axes across it."""
    g = np.arange(-int(rmax) - 1, int(rmax) + 2, 1.0)
    u, v = np.meshgrid(g, g, indexing="ij")
    r = np.hypot(u, v)
    keep = (r <= rmax) & (r >= rmin)
    t = np.arange(-length / 2, length / 2 + 0.01, 1.0)
    return np.stack([np.repeat(t, keep.sum()), np.tile(u[keep], len(t)), np.tile(v[keep], len(t))], 1)


def oriented(t, d):
    u, v = basis(d)
    return t[:, 0:1] * d + t[:, 1:2] * u + t[:, 2:3] * v


def score(vol, x, points):
    """``(mean of vol under the template, fraction of the template in the volume)``; x, y, z voxels."""
    q = np.round(x + points).astype(int)[:, ::-1]
    ok = ((q >= 0) & (q < np.array(vol.shape))).all(-1)
    n = int(ok.sum())
    if not n:
        return 0.0, 0.0
    return float(vol[q[ok, 0], q[ok, 1], q[ok, 2]].mean()), n / len(q)


def radial_profile(mask, polylines, rmax):
    """Fill of the mask by whole-voxel distance from the polylines (x, y, z)."""
    tree = cKDTree(np.concatenate(polylines))
    z, y, x = np.meshgrid(*[np.arange(s) for s in mask.shape], indexing="ij")
    grid = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)[::3]
    dist = tree.query(grid, distance_upper_bound=rmax + 1, workers=-1)[0]
    ok = np.isfinite(dist)
    bins = dist[ok].astype(int)
    n = int(rmax) + 2
    return np.bincount(bins, mask.ravel()[::3][ok], n) / np.maximum(np.bincount(bins, minlength=n), 1)


def template_bank(mask, points, directions, device):
    """Best response and direction index per voxel."""
    shape = mask.shape
    spectrum = torch.fft.rfftn(torch.as_tensor(mask, dtype=torch.float32, device=device))
    best = torch.zeros(shape, device=device)
    arg = torch.zeros(shape, dtype=torch.int16, device=device)
    for i, d in enumerate(directions):
        p = np.unique(np.round(oriented(points, d)).astype(int)[:, ::-1], axis=0)
        kernel = torch.zeros(shape, device=device)
        kernel[p[:, 0] % shape[0], p[:, 1] % shape[1], p[:, 2] % shape[2]] = 1.0 / len(p)
        response = torch.fft.irfftn(spectrum * torch.conj(torch.fft.rfftn(kernel)), s=shape)
        better = response > best
        best = torch.where(better, response, best)
        arg = torch.where(better, torch.full_like(arg, i), arg)
    return best.cpu().numpy(), arg.cpu().numpy()


def _fan(cone):
    out = [np.array([0, 0, 1.0])]
    for theta in (cone / 2, cone):
        s, c = np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))
        for phi in np.deg2rad(np.arange(0, 360, 45)):
            out.append(np.array([s * np.cos(phi), s * np.sin(phi), c]))
    return np.array(out)


def trace_mask(mask, diameter_ang, apix, rmin, rmax, length, tau, device):
    """Polylines (x, y, z voxels) of one binary mask."""
    full = template(rmin, rmax, length)
    directions = hemisphere(N_DIRECTIONS)
    response, best_dir = template_bank(mask, full, directions, device)
    points = full
    if len(points) > MAX_TEMPLATE_POINTS:
        points = points[np.random.default_rng(0).choice(len(points), MAX_TEMPLATE_POINTS, replace=False)]
    slab = template(rmin, rmax, 2.0)
    fan = _fan(CONE_DEG)
    lateral = [(a, b) for a in range(-LATERAL_VOX, LATERAL_VOX + 1) for b in range(-LATERAL_VOX, LATERAL_VOX + 1)]
    diameter_vox = diameter_ang / apix
    stale_limit = max(2, int(2 * diameter_vox / STEP_VOX))
    free = mask.copy()
    shape_xyz = np.array(mask.shape[::-1])

    def walk(x, d):
        path, stale = [], 0
        while len(path) < 4000:
            # Stop before turning: near a border, the score favours directions that stay in the volume.
            if score(mask, x + STEP_VOX * d, oriented(points, d))[1] < MIN_IN_VOLUME:
                break
            u, v = basis(d)
            best = None
            for f in fan:
                cand = f[2] * d + f[0] * u + f[1] * v
                p = oriented(points, cand)
                cu, cv = basis(cand)
                for a, b in lateral:
                    c = x + STEP_VOX * cand + a * cu + b * cv
                    s, inside = score(mask, c, p)
                    if inside >= MIN_IN_VOLUME and (best is None or s > best[0]):
                        best = (s, c, cand)
            if best is None or best[0] < tau:
                break
            s, x, d = best
            path.append(x)
            new = score(free, x, oriented(points, d))[0]
            stale = stale + 1 if new / max(s, 1e-6) < 0.25 else 0
            if stale >= stale_limit:
                path = path[:-stale]
                x = path[-1] if path else x
                break
        return path, x, d

    def extend(x, d):
        """Straight on to the end of the mask or of the volume, at most one template length."""
        path, gap = [], 0
        p = oriented(slab, d)
        for k in range(1, int(length) + 1):
            c = x + k * d
            if (c < 0).any() or (c > shape_xyz - 1).any():
                break
            s, inside = score(mask, c, p)
            gap = gap + 1 if s < 0.6 * tau else 0
            if gap > 2:
                break
            path.append(c)
        return path[:-gap] if gap else path

    peaks = (response == ndimage.maximum_filter(response, size=3)) & (response >= tau)
    seeds = np.argwhere(peaks)
    seeds = seeds[np.argsort(-response[peaks])]
    lines = []
    for z, y, x in seeds:
        d0 = directions[best_dir[z, y, x]]
        x0 = np.array([x, y, z], float)
        p = oriented(points, d0)
        if score(free, x0, p)[0] < 0.6 * score(mask, x0, p)[0]:
            continue
        fwd, xf, df = walk(x0, -d0)
        back, xb, db = walk(x0, d0)
        fwd += extend(xf, df)
        back += extend(xb, db)
        line = np.array(back[::-1] + [x0] + fwd)
        if len(line) < 2:
            continue
        for q in LF.fit_filament_spline(line, apix, 0, apix)[0]:
            lo = np.maximum(np.round(q[::-1] - rmax - 1).astype(int), 0)
            hi = np.minimum(np.round(q[::-1] + rmax + 2).astype(int), np.array(mask.shape))
            zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            sub = free[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            sub[(zz - q[2]) ** 2 + (yy - q[1]) ** 2 + (xx - q[0]) ** 2 <= (rmax + 1) ** 2] = False
        # A filament cut by the border of the volume may be short.
        margin = rmax + length / 2
        at_border = bool((line.min(0) < margin).any() or (line.max(0) > shape_xyz - 1 - margin).any())
        if LF.polyline_length(line) * apix >= max((1 if at_border else 2) * diameter_ang, 140.0):
            lines.append(line)
    return [LF.fit_filament_spline(q, apix, FINAL_SMOOTHING_ANG, apix)[0] for q in lines], float(free.sum() / mask.sum())


def choose_template(mask, diameter_ang, apix):
    """``(rmin, rmax, profile)`` from the radial profile around a first skeleton trace."""
    first = LF.mask_to_centerlines(mask, apix, diameter_ang, 0.5 * diameter_ang)
    if not first:
        raise ValueError("no skeleton in the mask")
    profile = radial_profile(mask, first, diameter_ang / 2 / apix + 5)
    high = np.where(profile >= 0.5 * profile.max())[0]
    rmin = float(high.min())
    return (0.0 if rmin <= 1 else rmin), float(high.max()) + 0.5, profile


def write_model(polylines, path):
    import imodmodel
    from imodmodel.models import ContourType

    rows = [{"object_id": 0, "contour_id": i, "x": x, "y": y, "z": z}
            for i, p in enumerate(polylines) for x, y, z in np.asarray(p, float)]
    imodmodel.write(pd.DataFrame(rows), str(path), type=ContourType.OPEN)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask_dir", required=True, help="Binary masks, <stem>.mrc")
    ap.add_argument("--out_dir", required=True, help="Writes <stem>.mod and summary.json")
    ap.add_argument("--diameter_ang", type=float, required=True)
    ap.add_argument("--runs", default=None, help="Stems, comma-separated (default: all)")
    ap.add_argument("--rmin", type=float, default=None, help="Template inner radius, voxels (with --rmax)")
    ap.add_argument("--rmax", type=float, default=None, help="Template outer radius, voxels (with --rmin)")
    ap.add_argument("--apix", type=float, default=None, help="Voxel size, A (default: MRC header)")
    a = ap.parse_args(argv)
    if (a.rmin is None) != (a.rmax is None):
        ap.error("--rmin and --rmax go together")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [r.strip() for r in a.runs.split(",") if r.strip()] if a.runs else None
    files = [Path(a.mask_dir) / f"{n}.mrc" for n in names] if names else sorted(Path(a.mask_dir).glob("*.mrc"))
    summary = {}
    for f in files:
        t0 = time.time()
        with mrcfile.open(str(f), permissive=True) as m:
            mask = np.asarray(m.data) > 0
            apix = float(a.apix or m.voxel_size.x)
        rmin, rmax, profile = a.rmin, a.rmax, None
        if rmin is None:
            rmin, rmax, profile = choose_template(mask, a.diameter_ang, apix)
        length = max(20.0, a.diameter_ang / apix)
        if profile is None:
            first = LF.mask_to_centerlines(mask, apix, a.diameter_ang, 0.5 * a.diameter_ang)
            profile = radial_profile(mask, first, max(rmax, a.diameter_ang / 2 / apix) + 5)
        tau = 0.6 * float(profile[int(rmin):int(rmax) + 1].mean())
        lines, unclaimed = trace_mask(mask, a.diameter_ang, apix, rmin, rmax, length, tau, device)
        write_model(lines, out / f"{f.stem}.mod")
        total = sum(LF.polyline_length(p) for p in lines) * apix
        summary[f.stem] = {"lines": len(lines), "total_length_ang": round(float(total)), "rmin": rmin, "rmax": rmax,
                           "tau": round(tau, 3), "unclaimed_mask_fraction": round(unclaimed, 3),
                           "seconds": round(time.time() - t0)}
        print(f.stem, summary[f.stem], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
