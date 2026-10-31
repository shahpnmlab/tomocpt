"""Helpers for the Gate 6 tests (no tests here).

New public names (spec ``06_AGENT_FILAMENTS.md``, "Public names") are imported inside the
functions, so collection works on code without ``tomocpt.labels.filaments``.

Conventions (spec T6.1, CLAUDE.md 5.2):

* polylines are ``[K, 3]`` arrays of ``x, y, z`` in native voxels;
* volumes and segments are ``z, y, x``; a segment row is
  ``z0, y0, x0, z1, y1, x1, class_index, d_vox`` (float32 ``[M, 8]``).

Geometry truths are computed here independently of the code under test: dense analytic
curves, exact distance from voxel centres to a polyline (segments with clamped ends, so tube
ends are round caps), and the CLAUDE.md 1.3 profile.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from utils.tiny import applied_main_config

TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = TESTS_DIR / "golden"
CHUNK = 64
EPS = 1e-4


# --------------------------------------------------------------------------------------
# profile and geometry
# --------------------------------------------------------------------------------------

def profile(r, radius: float) -> np.ndarray:
    """CLAUDE.md 1.3: ``0.5 exp(-r^2/2 ss^2) + 0.5 exp(-r^2/2 sl^2)``, ``ss = 0.33 R/2``,
    ``sl = 0.66 R/2``; 1 at ``r = 0``; 0 for ``r > R``."""
    r = np.asarray(r, dtype=np.float64)
    ss, sl = 0.33 * radius / 2.0, 0.66 * radius / 2.0
    v = 0.5 * np.exp(-r ** 2 / (2 * ss ** 2)) + 0.5 * np.exp(-r ** 2 / (2 * sl ** 2))
    return np.where(r > radius, 0.0, v)


def arc_xyz(center_xy, radius_vox, theta0, theta1, z, n=4001) -> np.ndarray:
    """Dense circular arc in the plane ``z = const`` (``x = cx + R cos t``, ``y = cy + R sin t``)."""
    t = np.linspace(theta0, theta1, n)
    return np.stack([center_xy[0] + radius_vox * np.cos(t), center_xy[1] + radius_vox * np.sin(t),
                     np.full_like(t, float(z))], axis=1)


def line_xyz(p0, p1, n=2001) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) * np.asarray(p0, float)[None] + t * np.asarray(p1, float)[None]


def helix_xyz(radius_vox, pitch_vox, t0, t1, origin=(0.0, 0.0, 0.0), n=20001) -> np.ndarray:
    """Helix with its axis along X: ``x = ox + pitch t / 2pi``, ``y = oy + r cos t``, ``z = oz + r sin t``."""
    t = np.linspace(t0, t1, n)
    ox, oy, oz = origin
    return np.stack([ox + pitch_vox * t / (2 * np.pi), oy + radius_vox * np.cos(t),
                     oz + radius_vox * np.sin(t)], axis=1)


def polyline_length(p) -> float:
    p = np.asarray(p, float).reshape(-1, 3)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def cumulative_length(p) -> np.ndarray:
    p = np.asarray(p, float).reshape(-1, 3)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def resample_by_length(p, step: float) -> np.ndarray:
    """Points every ``step`` (approximately) along a dense polyline, both ends included."""
    p = np.asarray(p, float)
    s = cumulative_length(p)
    n = max(2, int(round(s[-1] / step)) + 1)
    targets = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(targets, s, p[:, a]) for a in range(3)], axis=1)


def densify(p, step: float = 0.25) -> np.ndarray:
    """Linear interpolation of a polyline at most ``step`` apart (keeps every vertex)."""
    p = np.asarray(p, float).reshape(-1, 3)
    if len(p) < 2:
        return p.copy()
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(a[None] + t * (b - a)[None])
    return np.concatenate(out)


def point_to_polyline_distance(points_xyz, poly_xyz) -> np.ndarray:
    """Exact distance from each point to a polyline (union of its segments)."""
    pts = np.asarray(points_xyz, float).reshape(-1, 3)
    poly = np.asarray(poly_xyz, float).reshape(-1, 3)
    if len(poly) == 1:
        return np.linalg.norm(pts - poly[0], axis=1)
    best = np.full(len(pts), np.inf)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-300)
    for i0 in range(0, len(pts), 4096):
        p = pts[i0:i0 + 4096]
        t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0.0, 1.0)
        q = a[None] + t[..., None] * ab[None]
        best[i0:i0 + 4096] = np.sqrt(((p[:, None, :] - q) ** 2).sum(-1)).min(1)
    return best


def distance_field(shape_zyx, polylines_xyz: Sequence[np.ndarray], reach: float) -> np.ndarray:
    """Exact distance from every voxel centre ``(z, y, x)`` to the union of the polylines
    (segments with clamped ends), computed inside a box of ``reach`` around each segment;
    ``inf`` elsewhere."""
    shape = tuple(int(s) for s in shape_zyx)
    dist = np.full(shape, np.inf)
    for poly in polylines_xyz:
        poly = np.asarray(poly, float).reshape(-1, 3)
        segs = [(poly[0], poly[0])] if len(poly) == 1 else list(zip(poly[:-1], poly[1:]))
        for a, b in segs:
            a_zyx, b_zyx = a[::-1], b[::-1]
            lo = [max(0, int(math.floor(min(a_zyx[k], b_zyx[k]) - reach))) for k in range(3)]
            hi = [min(shape[k], int(math.ceil(max(a_zyx[k], b_zyx[k]) + reach)) + 1) for k in range(3)]
            if any(h <= l for l, h in zip(lo, hi)):
                continue
            zz, yy, xx = np.meshgrid(*[np.arange(l, h, dtype=np.float64) for l, h in zip(lo, hi)], indexing="ij")
            p = np.stack([zz, yy, xx], -1)
            ab = b_zyx - a_zyx
            L2 = float((ab ** 2).sum())
            if L2 > 0:
                t = np.clip(((p - a_zyx) * ab).sum(-1) / L2, 0.0, 1.0)
            else:
                t = np.zeros(p.shape[:-1])
            d = np.sqrt(((p - (a_zyx + t[..., None] * ab)) ** 2).sum(-1))
            box = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
            np.minimum(dist[box], d, out=dist[box])
    return dist


def segments_from_polyline(poly_xyz, class_index: int, d_vox: float) -> np.ndarray:
    """``[M, 8]`` float32 rows ``z0, y0, x0, z1, y1, x1, class_index, d_vox``."""
    p = np.asarray(poly_xyz, float).reshape(-1, 3)
    if len(p) < 2:
        return np.zeros((0, 8), np.float32)
    zyx = p[:, ::-1]
    seg = np.concatenate([zyx[:-1], zyx[1:], np.full((len(p) - 1, 1), class_index),
                          np.full((len(p) - 1, 1), d_vox)], axis=1)
    return seg.astype(np.float32)


def tube_target(shape_zyx, polylines_xyz, R: float) -> np.ndarray:
    """Reference soft tube target: ``profile(distance to the polylines, R)``."""
    d = distance_field(shape_zyx, polylines_xyz, reach=R + 2.0)
    return profile(np.where(np.isfinite(d), d, 1e9), R).astype(np.float32)


def filled_tube_mask(shape_zyx, polylines_xyz, R: float) -> np.ndarray:
    """Solid tube (round caps): voxels within ``R`` of the polylines."""
    d = distance_field(shape_zyx, polylines_xyz, reach=R + 2.0)
    return d <= R


# --------------------------------------------------------------------------------------
# polyline comparison
# --------------------------------------------------------------------------------------

def mean_dist_to_curve(points_xyz, curve_xyz) -> float:
    return float(point_to_polyline_distance(points_xyz, curve_xyz).mean())


def max_dist_to_curve(points_xyz, curve_xyz) -> float:
    return float(point_to_polyline_distance(points_xyz, curve_xyz).max())


def coverage(curve_xyz, recovered_xyz, tol: float, trim: float = 0.0) -> float:
    """Fraction of the (densified) true curve within ``tol`` of the recovered polyline,
    ignoring ``trim`` voxels of arc length at each end."""
    dense = densify(curve_xyz, 0.5)
    s = cumulative_length(dense)
    keep = (s >= trim) & (s <= s[-1] - trim)
    rec = np.asarray(recovered_xyz, float).reshape(-1, 3)
    d = point_to_polyline_distance(dense[keep], rec) if len(rec) else np.full(int(keep.sum()), np.inf)
    return float((d <= tol).mean())


def is_ordered_along(points_xyz, curve_xyz) -> bool:
    """The points' projections onto the curve (arc position of the nearest dense point) are
    strictly monotone (either direction)."""
    dense = densify(curve_xyz, 0.05)
    s = cumulative_length(dense)
    pts = np.asarray(points_xyz, float).reshape(-1, 3)
    idx = np.array([int(np.argmin(((dense - p) ** 2).sum(1))) for p in pts])
    ds = np.diff(s[idx])
    return bool(np.all(ds > 0) or np.all(ds < 0))


def match_polylines(recovered, truths) -> List[int]:
    """For each recovered polyline, the index of the truth with the smallest mean distance."""
    return [int(np.argmin([mean_dist_to_curve(r, t) for t in truths])) for r in recovered]


# --------------------------------------------------------------------------------------
# RELION Euler convention (src/euler.cpp, Euler_angles2matrix)
# --------------------------------------------------------------------------------------

def relion_euler_matrix(rot_deg, tilt_deg, psi_deg) -> np.ndarray:
    """``Euler_angles2matrix(alpha=rot, beta=tilt, gamma=psi)`` of RELION ``src/euler.cpp``."""
    alpha, beta, gamma = np.radians([rot_deg, tilt_deg, psi_deg])
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
    cc, cs = cb * ca, cb * sa
    sc, ss = sb * ca, sb * sa
    A = np.empty((3, 3))
    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = cg * cs + sg * ca
    A[0, 2] = -cg * sb
    A[1, 0] = -sg * cc - cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = sc
    A[2, 1] = ss
    A[2, 2] = cb
    return A


def reference_tilt_psi(d) -> Tuple[float, float]:
    """The obviously-correct angles for a unit direction ``d = (x, y, z)``: the third column of
    ``A(0, tilt, psi)`` is ``(-cos psi sin tilt, sin psi sin tilt, cos tilt)``. Used only to
    check the test's own reference matrix."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, d[2]))))
    psi = math.degrees(math.atan2(d[1], -d[0]))
    return tilt, psi


# --------------------------------------------------------------------------------------
# filament coordinate files (inputs for prepare_data and read_filament_coords)
# --------------------------------------------------------------------------------------

def write_filament_star(path, tomo_name: str, polylines_xyz: Sequence[np.ndarray], apix: float,
                        class_number: int, track_length: bool = True, shuffle_seed: Optional[int] = 0) -> Path:
    """RELION 3.1 style STAR (as ``predict`` writes particle files): ``rlnCoordinateX/Y/Z`` in
    voxels, optics ``rlnImagePixelSize = apix``, ``rlnMicrographName``, ``rlnClassNumber``,
    ``rlnHelicalTubeID`` (1-based) and, optionally, ``rlnHelicalTrackLengthAngst``. Rows are
    shuffled with ``shuffle_seed`` (None keeps the order)."""
    import starfile

    rows = []
    for tid, poly in enumerate(polylines_xyz, 1):
        poly = np.asarray(poly, float)
        s = cumulative_length(poly) * apix
        for (x, y, z), sl in zip(poly, s):
            r = {"rlnMicrographName": tomo_name, "rlnCoordinateX": x, "rlnCoordinateY": y, "rlnCoordinateZ": z,
                 "rlnClassNumber": int(class_number), "rlnHelicalTubeID": tid, "rlnOpticsGroup": 1}
            if track_length:
                r["rlnHelicalTrackLengthAngst"] = sl
            rows.append(r)
    df = pd.DataFrame(rows)
    if shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["OpticsGroup1"],
                           "rlnSphericalAberration": [2.7], "rlnVoltage": [300],
                           "rlnImagePixelSize": [float(apix)], "rlnImageDimensionality": [3]})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"optics": optics, "particles": df}, path, overwrite=True, float_format="%.6f")
    return path


def write_filament_imod(path, polylines_xyz: Sequence[np.ndarray]) -> Path:
    """IMOD model, one object, one **open** contour per filament, points ``x, y, z`` in voxels."""
    import imodmodel
    from imodmodel.models import ContourType

    rows = []
    for cid, poly in enumerate(polylines_xyz):
        for x, y, z in np.asarray(poly, float):
            rows.append({"object_id": 0, "contour_id": cid, "x": x, "y": y, "z": z})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imodmodel.write(pd.DataFrame(rows), str(path), type=ContourType.OPEN)
    return path


def read_filament_csv(path) -> List[np.ndarray]:
    """``<dataset_dir>/filaments/<tomo>.csv`` -> polylines (rows of each ``filament_id`` in file order)."""
    df = pd.read_csv(path)
    assert {"filament_id", "x", "y", "z", "class_number", "diameter_ang"} <= set(df.columns), list(df.columns)
    return [g[["x", "y", "z"]].to_numpy(float) for _, g in df.groupby("filament_id", sort=True)]


def read_star_df(path) -> pd.DataFrame:
    import starfile

    d = starfile.read(str(path))
    if isinstance(d, dict):
        return d["particles"]
    return d


# --------------------------------------------------------------------------------------
# configs and checkpoints with a filament class
# --------------------------------------------------------------------------------------

def v2_config_with_filaments(tmp_path, particles, filaments, cond: bool = False, class_head: str = "sigmoid"):
    """Tiny CPU v2 ``MainConfig``: ``particles`` / ``filaments`` = ``((number, name, diameter_ang), ...)``.
    Filament classes are declared with ``prepData.filament_coords`` (a directory per class)."""
    from utils.tiny import tiny_main_config

    tmp_path = Path(tmp_path)
    cfg = tiny_main_config(tmp_path)
    classes = tuple(particles) + tuple(filaments)
    cfg.train.data_pipeline_version = 2
    cfg.prepData.multiclass = True
    cfg.prepData.class_diameters = ",".join(f"{n}:{d}" for n, _, d in classes)
    cfg.prepData.class_names = ",".join(f"{n}:{name}" for n, name, _ in classes)
    if filaments:
        entries = []
        for n, name, _ in filaments:
            d = tmp_path / "filament_coords" / name
            d.mkdir(parents=True, exist_ok=True)
            entries.append(f"{n}:{d}")
        cfg.prepData.filament_coords = ",".join(entries)
    cfg.train.network.scale_conditioning = cond
    cfg.train.particles.class_head = class_head
    return cfg


def build_ckpt(root: Path, particles, filaments=(), cond: bool = False) -> Path:
    """Tiny pipeline-v2 checkpoint (one-batch CPU fit, as ``g3_helpers.build_v2_checkpoint``)
    whose class table may hold filament classes."""
    from gate3 import g3_helpers as G3

    from tomocpt.networks.pickingModel import BasePickingModel

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = v2_config_with_filaments(root, particles, filaments, cond=cond)
    d_vox = [d / 10.0 for _, _, d in tuple(particles) + tuple(filaments)]
    items = [G3.v2_item(100 + i, d_vox, apix=10.0) for i in range(2)]
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        trainer = G3.cpu_trainer(root / "tb")
        trainer.fit(model, datamodule=G3.datamodule(items))
        path = root / "v2.ckpt"
        trainer.save_checkpoint(path)
    return path


# --------------------------------------------------------------------------------------
# oracle for predict: particles dark, filaments bright
# --------------------------------------------------------------------------------------

def _kinds_of(model) -> List[str]:
    from tomocpt.labels.multiclass import class_table_from_config

    return [str(getattr(r.kind, "value", r.kind)) for r in class_table_from_config(model.hparams.config.prepData)]


def sign_oracle_factory():
    """Oracle network for ``g4_helpers.run_predict``: particle channels
    ``clip(-x_norm - 0.5)``, filament channels ``clip(x_norm - 0.5)`` (see :func:`signed_tomogram`)."""
    from gate4 import g4_helpers as G4

    class SignOracle(G4._OracleBase):
        def __init__(self, kinds):
            super().__init__(len(kinds), conditioned=False)
            self.filament = torch.tensor([k == "filament" for k in kinds])

        def _probs(self, x, apix):
            part = (-x - 0.5).expand(-1, self.C, -1, -1, -1)
            fil = (x - 0.5).expand(-1, self.C, -1, -1, -1)
            sel = self.filament.to(x.device).view(1, -1, 1, 1, 1)
            return torch.where(sel, fil, part)

    return lambda model: SignOracle(_kinds_of(model))


def signed_tomogram(shape_zyx, apix: float, blobs, tubes, amplitude: float = 3.0, slab_frac: float = 0.1):
    """Noise-free volume for :func:`sign_oracle_factory`.

    * ``blobs``: ``(class_number, diameter_ang, (z, y, x))`` -> ``-amplitude * profile``;
    * ``tubes``: ``(class_number, diameter_ang, polyline_xyz)`` -> ``+amplitude * profile``
      of the distance to the polyline;
    * a bright slab (+1) at low X and a dark slab (-1) at high X, ``slab_frac`` of the voxels
      each with a smooth edge: robust normalisation gives median 0, ``p95 - p5 = 2``, so
      ``x_norm = x / 2``; the slabs map to 0 in both oracle channels and a structure peak to 1.
    """
    from scipy.special import erf

    shape = tuple(int(s) for s in shape_zyx)
    vol = np.zeros(shape, np.float64)
    for _, diam, (cz, cy, cx) in blobs:
        R = diam / 2.0 / apix
        zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij")
        r = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        vol = np.minimum(vol, -amplitude * profile(r, R))
    bright = np.zeros(shape, np.float64)
    for _, diam, poly in tubes:
        R = diam / 2.0 / apix
        bright = np.maximum(bright, amplitude * tube_target(shape, [poly], R))
    vol = vol + bright
    x = np.arange(shape[2], dtype=np.float64)
    edge = slab_frac * shape[2]
    vol += 0.5 * (1.0 - erf((x - edge) / 3.0))[None, None, :]
    vol -= 0.5 * (1.0 + erf((x - (shape[2] - 1 - edge)) / 3.0))[None, None, :]
    return vol.astype(np.float32)


# --------------------------------------------------------------------------------------
# prepare_data with filaments
# --------------------------------------------------------------------------------------

def prep_config(ds, tdd, particles, filaments, **extra):
    """Multi-class ``PrepdataConfig`` for a ``utils.phantoms.make_dataset`` result."""
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig

    classes = tuple(particles) + tuple(filaments)
    prep = PrepdataConfig(raw_data_dir=str(ds.tomo_dir), training_data_dir=Path(tdd), particle_length_ang="250",
                          coordinate_files=str(ds.star_path), class_id="all", multiclass=True,
                          class_diameters=",".join(f"{n}:{d}" for n, _, d in classes),
                          class_names=",".join(f"{n}:{name}" for n, name, _ in classes))
    for k, v in extra.items():
        setattr(prep, k, v)
    return prep


def dataset_dirs(tdd) -> List[Path]:
    return sorted(p for p in Path(tdd).iterdir() if p.is_dir())


def filament_csv(tdd, tomo_name: str) -> Path:
    hits = sorted(Path(tdd).glob(f"*/filaments/{tomo_name}.csv"))
    assert len(hits) == 1, f"expected one <dataset_dir>/filaments/{tomo_name}.csv, got {hits}"
    return hits[0]


def write_mrc(path, vol, apix):
    from utils.phantoms import write_mrc as _w

    return _w(path, vol, apix)


def read_mrc(path) -> Tuple[np.ndarray, float]:
    import mrcfile

    with mrcfile.open(str(path), permissive=True) as m:
        return m.data.copy(), float(m.voxel_size.x)
