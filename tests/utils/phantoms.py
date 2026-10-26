"""Synthetic tomograms for tests (written at Gate 1, frozen afterwards).

Main entry points
-----------------
* :func:`make_phantom` -> :class:`Phantom` ``(volume, coords, polylines, masks)``.
  It is a ``NamedTuple``: unpack all four fields, or use attribute access
  (``ph.volume``, ``ph.coords``, ``ph.polylines``, ``ph.masks``).
* :func:`write_mrc`, :func:`write_star` (RELION 3.1 style).
* :func:`make_dataset`: tomograms + one STAR file, ready for ``tomocpt prepare_data``.

Conventions
-----------
* Volumes are ``(Z, Y, X)`` float32. Objects have **dark** contrast (negative density on a
  zero background), as in cryo-ET.
* Coordinates and polylines are ``(x, y, z)`` in voxels (RELION order), i.e. voxel
  ``vol[z, y, x]``. Coordinates are floats (sub-voxel) unless ``integer_coords=True``.
* All randomness comes from ``np.random.default_rng(seed)``. Geometry is drawn before the
  noise, so calling again with the same ``seed`` and ``snr=None`` (and/or ``wedge_deg=None``)
  gives the same objects without noise (and/or without the wedge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

try:  # works both as ``utils.phantoms`` and when imported by path
    from .wedge import apply_missing_wedge
except ImportError:  # pragma: no cover
    from wedge import apply_missing_wedge  # type: ignore

__all__ = ["Phantom", "SyntheticDataset", "make_phantom", "write_mrc", "write_star", "make_dataset"]

#: Default species: class 1, 120 A solid spheres, 6 copies.
DEFAULT_SPECIES = ((1, 120.0, 6, False),)


class Phantom(NamedTuple):
    #: ``(Z, Y, X)`` float32 volume (objects dark), after the wedge and the noise.
    volume: np.ndarray
    #: ``{class_number: (N, 3) array of (x, y, z) voxel centres}`` for every species class.
    coords: Dict[int, np.ndarray]
    #: ``{class_number: [ (M_i, 3) arrays of (x, y, z) ]}`` centrelines for every tube class,
    #: densely sampled (0.5 voxel spacing), clipped to the volume.
    polylines: Dict[int, List[np.ndarray]]
    #: ``{class_number: (Z, Y, X) bool}`` exact tube-wall masks (union of the class's tubes).
    masks: Dict[int, np.ndarray]


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _occupancy_shell(dist: np.ndarray, r_out: float, r_in: Optional[float]) -> np.ndarray:
    """Soft (anti-aliased, 1-voxel ramp) occupancy of ``r_in <= dist <= r_out``."""
    occ = np.clip(r_out + 0.5 - dist, 0.0, 1.0)
    if r_in is not None and r_in > 0:
        occ = occ * np.clip(dist - r_in + 0.5, 0.0, 1.0)
    return occ


def _box(center_zyx, radius, shape):
    lo = [max(0, int(math.floor(c - radius - 1))) for c in center_zyx]
    hi = [min(s, int(math.ceil(c + radius + 2))) for c, s in zip(center_zyx, shape)]
    return lo, hi


def _trace_curve(rng, shape, r_out, min_curv_radius_vox, ds=0.5):
    """A smooth random curve (zyx points, spacing ``ds``) crossing the volume.

    The curve turns with a constant curvature ``kappa <= 1 / min_curv_radius_vox`` about a
    slowly drifting normal, so its radius of curvature is never below
    ``min_curv_radius_vox``. Mostly in-plane (the Z extent of a tomogram is small).
    Returns (points_all, points_inside): the first extends ``r_out + 2`` past the border
    (for rendering), the second is clipped to ``[0, dim - 1]``.
    """
    shape = np.asarray(shape, dtype=float)
    start = np.array([rng.uniform(0.35, 0.65) * (s - 1) for s in shape])
    phi = rng.uniform(0, 2 * np.pi)
    tilt = np.radians(rng.uniform(-8.0, 8.0))
    t0 = _unit(np.array([np.sin(tilt), np.cos(tilt) * np.sin(phi), np.cos(tilt) * np.cos(phi)]))
    kappa = rng.uniform(0.3, 1.0) / float(min_curv_radius_vox) if min_curv_radius_vox and np.isfinite(min_curv_radius_vox) else 0.0
    # initial normal: in-plane, perpendicular to t
    n0 = _unit(np.cross(t0, np.array([1.0, 0.0, 0.0])))
    if rng.random() < 0.5:
        n0 = -n0
    margin = r_out + 2.0
    lo, hi = -margin, shape - 1 + margin
    max_steps = int(4 * shape.max() / ds) + 10

    def walk(direction, normal):
        pts = []
        p, t, n = start.copy(), direction.copy(), normal.copy()
        for _ in range(max_steps):
            p = p + ds * t
            if np.any(p < lo) or np.any(p > hi):
                break
            pts.append(p.copy())
            # drift the normal a little (kept perpendicular to t), then turn t towards it
            n = n + rng.normal(0.0, 0.02, 3)
            n = _unit(n - np.dot(n, t) * t)
            t = _unit(t + kappa * ds * n)
            n = _unit(n - np.dot(n, t) * t)
        return pts

    fwd = walk(t0, n0)
    bwd = walk(-t0, n0)
    pts = np.array(bwd[::-1] + [start] + fwd)
    inside = np.all((pts >= 0) & (pts <= shape - 1), axis=1)
    # keep the longest contiguous inside run
    best, cur_start, best_rng = 0, None, (0, 0)
    for i, ok in enumerate(list(inside) + [False]):
        if ok and cur_start is None:
            cur_start = i
        elif not ok and cur_start is not None:
            if i - cur_start > best:
                best, best_rng = i - cur_start, (cur_start, i)
            cur_start = None
    return pts, pts[best_rng[0]:best_rng[1]]


def _distance_to_polyline(points_zyx: np.ndarray, shape, reach: float, chunk: int = 16) -> np.ndarray:
    """Exact Euclidean distance (voxels) from every voxel to the polyline, ``inf`` beyond ``reach``."""
    dist = np.full(tuple(shape), np.inf, dtype=np.float32)
    if len(points_zyx) < 2:
        return dist
    for s in range(0, len(points_zyx) - 1, chunk):
        seg = points_zyx[s:s + chunk + 1]
        a, b = seg[:-1], seg[1:]
        lo = np.maximum(0, np.floor(seg.min(0) - reach - 1)).astype(int)
        hi = np.minimum(np.asarray(shape), np.ceil(seg.max(0) + reach + 2)).astype(int)
        if np.any(hi <= lo):
            continue
        zz, yy, xx = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij")
        p = np.stack([zz, yy, xx], -1).reshape(-1, 1, 3).astype(np.float64)
        ab = (b - a)[None]
        denom = np.maximum((ab ** 2).sum(-1), 1e-12)
        t = np.clip(((p - a[None]) * ab).sum(-1) / denom, 0.0, 1.0)
        d = np.sqrt(((p - (a[None] + t[..., None] * ab)) ** 2).sum(-1)).min(1)
        sub = dist[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.minimum(sub, d.reshape(sub.shape).astype(np.float32), out=sub)
    return dist


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------

def make_phantom(
    shape: Tuple[int, int, int] = (64, 128, 128),
    apix: float = 10.0,
    species: Sequence[Tuple[int, float, int, bool]] = DEFAULT_SPECIES,
    tubes: Sequence[Tuple[int, float, int]] = (),
    wedge_deg: Optional[float] = 60.0,
    snr: Optional[float] = 0.5,
    seed: int = 0,
    allow_border: bool = False,
    *,
    tube_min_curvature_radius_ang: float = 1500.0,
    hollow_wall_fraction: float = 0.2,
    integer_coords: bool = False,
    max_tries: int = 5000,
) -> Phantom:
    """Make a synthetic tomogram.

    Parameters
    ----------
    shape : (Z, Y, X)
    apix : float
        Voxel size in Angstrom.
    species : tuple of ``(class_number, diameter_ang, count, hollow)``
        Spheres. Solid, or hollow shells with a wall of ``hollow_wall_fraction * radius``
        (at least 1.5 voxels). Spheres never overlap each other or a tube (1-voxel gap).
        With ``allow_border=False`` every sphere lies fully inside the volume with a margin
        of 2 voxels (so the legacy label writer does not skip it); with
        ``allow_border=True`` centres are uniform over the whole volume and spheres may be
        cut by the border.
    tubes : tuple of ``(class_number, diameter_ang, count)``
        Cylinders along smooth random curves that cross the volume. Hollow with a wall of
        ``0.2 * diameter`` (50 A for a 250 A microtubule) when ``diameter >= 100`` A, solid
        otherwise. The radius of curvature of every centreline is at least
        ``tube_min_curvature_radius_ang``. Tubes may cross each other.
    wedge_deg : float or None
        Maximum tilt angle for :func:`utils.wedge.apply_missing_wedge` (tilt axis Y).
        ``None`` or ``>= 90`` applies no wedge.
    snr : float or None
        Signal-to-noise ratio ``var(signal) / var(noise)`` of the added white Gaussian noise,
        with ``signal`` the wedged density. ``None`` or ``inf`` adds no noise.
    seed : int
    allow_border : bool

    Returns
    -------
    Phantom
        ``(volume, coords, polylines, masks)``; see :class:`Phantom`.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)
    occupancy = np.zeros(shape, dtype=np.float32)

    # ---- tubes first: spheres are placed around them ---------------------------------
    polylines: Dict[int, List[np.ndarray]] = {}
    masks: Dict[int, np.ndarray] = {}
    tube_points: List[Tuple[np.ndarray, float]] = []  # (zyx points, outer radius)
    min_curv_vox = tube_min_curvature_radius_ang / apix
    for class_number, diameter_ang, count in tubes:
        class_number = int(class_number)
        r_out = diameter_ang / (2.0 * apix)
        r_in = r_out - hollow_wall_fraction * diameter_ang / apix if diameter_ang >= 100.0 else None
        polylines.setdefault(class_number, [])
        class_mask = masks.setdefault(class_number, np.zeros(shape, dtype=bool))
        for _ in range(int(count)):
            for _attempt in range(50):
                pts_all, pts_in = _trace_curve(rng, shape, r_out, min_curv_vox)
                if len(pts_in) * 0.5 >= min(shape[1:]) * 0.5:  # long enough to be useful
                    break
            dist = _distance_to_polyline(pts_all, shape, reach=r_out + 1.5)
            occupancy = np.maximum(occupancy, _occupancy_shell(dist, r_out, r_in).astype(np.float32))
            wall = dist <= r_out
            if r_in is not None:
                wall &= dist >= r_in
            class_mask |= wall
            polylines[class_number].append(pts_in[:, ::-1].copy())  # zyx -> xyz
            tube_points.append((pts_all, r_out))

    # ---- spheres -----------------------------------------------------------------------
    placed: List[Tuple[np.ndarray, float]] = []  # (zyx centre, radius)
    coords_zyx: Dict[int, List[np.ndarray]] = {int(c): [] for c, *_ in species}
    order = sorted(range(len(species)), key=lambda i: -species[i][1])  # big first packs better
    for i in order:
        class_number, diameter_ang, count, hollow = species[i]
        class_number = int(class_number)
        r = diameter_ang / (2.0 * apix)
        wall = max(1.5, hollow_wall_fraction * r)
        for _ in range(int(count)):
            for _attempt in range(max_tries):
                if allow_border:
                    c = np.array([rng.uniform(0, s - 1) for s in shape])
                else:
                    m = math.ceil(r) + 2
                    if any(s - 1 - m < m for s in shape):
                        raise ValueError(f"volume {shape} too small for a {diameter_ang} A sphere at {apix} A/px")
                    c = np.array([rng.uniform(m, s - 1 - m) for s in shape])
                if integer_coords:
                    c = np.round(c)
                if any(np.linalg.norm(c - pc) < r + pr + 1.0 for pc, pr in placed):
                    continue
                if any(np.min(np.linalg.norm(tp - c, axis=1)) < r + tr + 1.0 for tp, tr in tube_points):
                    continue
                break
            else:
                raise RuntimeError(f"could not place species {class_number} ({diameter_ang} A) without overlap")
            placed.append((c, r))
            coords_zyx[class_number].append(c)
            lo, hi = _box(c, r, shape)
            zz, yy, xx = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij")
            d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
            occ = _occupancy_shell(d, r, r - wall if hollow else None).astype(np.float32)
            sub = occupancy[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            np.maximum(sub, occ, out=sub)

    coords = {
        c: (np.array(v)[:, ::-1].astype(np.float64) if v else np.zeros((0, 3), dtype=np.float64))
        for c, v in coords_zyx.items()
    }

    # ---- contrast, wedge, noise ----------------------------------------------------------
    volume = -occupancy
    if wedge_deg is not None and wedge_deg < 90:
        volume = apply_missing_wedge(volume, max_angle_deg=float(wedge_deg), tilt_axis="y")
    if snr is not None and np.isfinite(snr):
        if snr <= 0:
            raise ValueError("snr must be > 0 (or None for no noise)")
        var = float(volume.var())
        std = math.sqrt(var / snr) if var > 0 else 1.0
        volume = volume + rng.normal(0.0, std, size=shape).astype(np.float32)
    return Phantom(volume.astype(np.float32), coords, polylines, masks)


def write_mrc(path: Union[str, Path], vol: np.ndarray, apix: float) -> Path:
    """Write ``vol`` (Z, Y, X) as float32 MRC with voxel size ``apix`` (A)."""
    import mrcfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(vol, dtype=np.float32))
        mrc.voxel_size = float(apix)
    return path


def write_star(path: Union[str, Path], coords_by_class, tomo_name, apix: float) -> Path:
    """Write a RELION 3.1 particle STAR file (``data_optics`` + ``data_particles``).

    Parameters
    ----------
    coords_by_class : ``{class_number: (N, 3) xyz}`` for one tomogram, or a list of such
        dicts (one per tomogram) when ``tomo_name`` is a list.
    tomo_name : str or list of str
        Written to ``rlnMicrographName``.
    apix : float
        ``rlnImagePixelSize``; coordinates are in pixels of this size (tomogram voxels).

    Columns: ``rlnCoordinateX/Y/Z``, ``rlnOriginX/Y/ZAngst`` (0), ``rlnMicrographName``,
    ``rlnClassNumber``, ``rlnOpticsGroup``.
    """
    import pandas as pd
    import starfile

    if isinstance(tomo_name, (str, Path)):
        names, per_tomo = [str(tomo_name)], [coords_by_class]
    else:
        names, per_tomo = [str(n) for n in tomo_name], list(coords_by_class)
        if len(names) != len(per_tomo):
            raise ValueError("tomo_name and coords_by_class must have the same length")
    rows = []
    for name, cbc in zip(names, per_tomo):
        for class_number in sorted(cbc):
            for x, y, z in np.asarray(cbc[class_number], dtype=float).reshape(-1, 3):
                rows.append(dict(rlnCoordinateX=x, rlnCoordinateY=y, rlnCoordinateZ=z,
                                 rlnOriginXAngst=0.0, rlnOriginYAngst=0.0, rlnOriginZAngst=0.0,
                                 rlnMicrographName=name, rlnClassNumber=int(class_number),
                                 rlnOpticsGroup=1))
    columns = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnOriginXAngst",
               "rlnOriginYAngst", "rlnOriginZAngst", "rlnMicrographName", "rlnClassNumber",
               "rlnOpticsGroup"]
    particles = pd.DataFrame(rows, columns=columns)
    optics = pd.DataFrame([dict(rlnOpticsGroup=1, rlnOpticsGroupName="opticsGroup1",
                                rlnImagePixelSize=float(apix), rlnVoltage=300.0,
                                rlnSphericalAberration=2.7, rlnAmplitudeContrast=0.1)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)
    return path


@dataclass
class SyntheticDataset:
    """What :func:`make_dataset` wrote."""
    root: Path
    tomo_dir: Path
    star_path: Path
    tomo_paths: List[Path]
    tomo_names: List[str]
    apix: float
    shape: Tuple[int, int, int]
    species: Tuple
    tubes: Tuple
    coords: List[Dict[int, np.ndarray]] = field(default_factory=list)       # per tomogram
    polylines: List[Dict[int, List[np.ndarray]]] = field(default_factory=list)
    masks: List[Dict[int, np.ndarray]] = field(default_factory=list)

    @property
    def class_diameters(self) -> Dict[int, float]:
        return {int(c): float(d) for c, d, *_ in self.species}

    def legacy_prepdata_fields(self, training_data_dir: Union[str, Path, None] = None) -> dict:
        """Field values for a legacy (pipeline v1) ``PrepdataConfig``: one dataset per species
        (comma-joined strings, as the legacy CLI expects), coordinate type STAR."""
        classes = [int(c) for c, *_ in self.species]
        out = dict(
            raw_data_dir=",".join(str(self.tomo_dir) for _ in classes),
            coordinate_files=",".join(str(self.star_path) for _ in classes),
            particle_length_ang=",".join(repr(float(self.class_diameters[c])) for c in classes),
            class_id=",".join(str(c) for c in classes),
        )
        if training_data_dir is not None:
            out["training_data_dir"] = Path(training_data_dir)
        return out


def make_dataset(
    tmp_path: Union[str, Path],
    n_tomos: int = 2,
    shape: Tuple[int, int, int] = (64, 128, 128),
    apix: float = 10.0,
    species: Sequence[Tuple[int, float, int, bool]] = DEFAULT_SPECIES,
    tubes: Sequence[Tuple[int, float, int]] = (),
    wedge_deg: Optional[float] = 60.0,
    snr: Optional[float] = 0.5,
    seed: int = 0,
    allow_border: bool = False,
    *,
    tomo_prefix: str = "TS_",
    star_name: str = "particles.star",
    **phantom_kwargs,
) -> SyntheticDataset:
    """Write ``n_tomos`` phantom tomograms (``<tmp_path>/tomograms/TS_01.mrc``, ...) and one
    RELION 3.1 STAR file (``<tmp_path>/particles.star``) with every species of every tomogram.

    Tomogram ``i`` (0-based) uses ``seed + i``. Default ``n_tomos=2`` because one tomogram
    cannot be split into train/val. Tube classes are not written to the STAR file (their
    centrelines and masks are returned).
    """
    root = Path(tmp_path)
    tomo_dir = root / "tomograms"
    tomo_dir.mkdir(parents=True, exist_ok=True)
    names, paths, coords, polys, masks = [], [], [], [], []
    for i in range(int(n_tomos)):
        name = f"{tomo_prefix}{i + 1:02d}"
        ph = make_phantom(shape=shape, apix=apix, species=species, tubes=tubes, wedge_deg=wedge_deg,
                          snr=snr, seed=seed + i, allow_border=allow_border, **phantom_kwargs)
        paths.append(write_mrc(tomo_dir / f"{name}.mrc", ph.volume, apix))
        names.append(name)
        coords.append(ph.coords)
        polys.append(ph.polylines)
        masks.append(ph.masks)
    star_path = write_star(root / star_name, coords, names, apix)
    return SyntheticDataset(root=root, tomo_dir=tomo_dir, star_path=star_path, tomo_paths=paths,
                            tomo_names=names, apix=float(apix), shape=tuple(shape),
                            species=tuple(tuple(s) for s in species), tubes=tuple(tuple(t) for t in tubes),
                            coords=coords, polylines=polys, masks=masks)
