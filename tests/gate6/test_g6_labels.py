"""G6-C3 (segments stored per chunk), G6-C12 (labels from filament coordinates), G6-C13
(``write_filament_masks``), and the T6.2 class table.

Inputs: two ``make_dataset`` tomograms (64 x 128 x 128, 10 A/px, class 1 "ribo" 120 A as STAR
particles) plus two analytic filaments per tomogram of class 5 "microtubule" (250 A, R = 12.5
voxels): a gentle arc (radius 130 voxels) and a straight line. 120 and 250 A at 10 A/px fit the
default band, so pipeline v2 works at the native voxel size (no resampling, no padding).

Filament inputs (spec "Labels from coordinates"): per tomogram, matched by name,
``<dir>/<tomo>.star`` (RELION 3.1 style, as ``predict`` writes: ``rlnCoordinateX/Y/Z`` in voxels,
optics ``rlnImagePixelSize``, ``rlnHelicalTubeID``, rows shuffled) with or without
``rlnHelicalTrackLengthAngst``, or ``<dir>/<tomo>.mod`` (IMOD, one open contour per filament);
masks: ``<dir>/<tomo>.mrc`` (solid tube, uint8).
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gate6 import g6_helpers as H
from utils.tiny import applied_main_config

APIX = 10.0
SHAPE = (64, 128, 128)
PART = ((1, "ribo", 120.0),)
FIL = ((5, "microtubule", 250.0),)
R_FIL = 250.0 / 2.0 / APIX
TOMOS = ("TS_01", "TS_02")
STAR_STEP = 4.0  # voxels between coordinate rows (40 A, the default predict spacing)


def truths(tomo: str):
    """Dense true centrelines (x, y, z voxels) of one tomogram: an arc and a straight line."""
    k = TOMOS.index(tomo)
    arc = H.arc_xyz((64.0 + 2 * k, -60.0), 130.0, math.radians(70.2), math.radians(109.8), z=32.0 - 2 * k, n=601)
    line = H.line_xyz((20.0, 100.0 - 3 * k, 40.0), (108.0, 110.0 - 3 * k, 24.0 + k), n=201)
    return [arc, line]


def samples(tomo: str):
    return [H.resample_by_length(t, STAR_STEP) for t in truths(tomo)]


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    from utils.phantoms import make_dataset

    root = tmp_path_factory.mktemp("g6_labels")
    d = make_dataset(root / "data", n_tomos=2, shape=SHAPE, apix=APIX, species=((1, 120.0, 4, False),),
                     snr=1.0, seed=11)
    assert tuple(d.tomo_names) == TOMOS
    return d


def _coords_dir(root: Path, kind: str) -> Path:
    out = root / f"fil_{kind}"
    for tomo in TOMOS:
        if kind == "star_tl":
            H.write_filament_star(out / f"{tomo}.star", tomo, samples(tomo), APIX, 5, track_length=True)
        elif kind == "star_nn":
            H.write_filament_star(out / f"{tomo}.star", tomo, samples(tomo), APIX, 5, track_length=False)
        elif kind == "imod":
            H.write_filament_imod(out / f"{tomo}.mod", samples(tomo))
        elif kind == "masks":
            mask = H.filled_tube_mask(SHAPE, truths(tomo), R_FIL).astype(np.uint8)
            H.write_mrc(out / f"{tomo}.mrc", mask, APIX)
        else:
            raise ValueError(kind)
    return out


@pytest.fixture(scope="module")
def prepared(ds):
    """``get(tag, field, kind, **extra) -> training_data_dir`` (prepare_data run once per tag)."""
    from tomocpt.labels.run import prepare_data

    cache = {}

    def get(tag, field, kind, **extra):
        if tag not in cache:
            src = _coords_dir(ds.root, kind)
            tdd = ds.root / f"td_{tag}"
            prep = H.prep_config(ds, tdd, PART, FIL, **{field: f"5:{src}"}, **extra)
            with applied_main_config(None):
                prepare_data(prep)
            cache[tag] = (tdd, prep)
        return cache[tag]

    return get


def _csv_polylines(tdd, tomo):
    path = H.filament_csv(tdd, tomo)
    df = pd.read_csv(path)
    assert set(df["class_number"].astype(int)) == {5}
    assert np.allclose(df["diameter_ang"].astype(float), 250.0)
    return H.read_filament_csv(path)


def _check_centrelines(polys, tomo, tol_max=0.5):
    tr = truths(tomo)
    assert len(polys) == 2, f"{tomo}: {len(polys)} filaments in the CSV, expected 2"
    m = H.match_polylines(polys, tr)
    assert sorted(m) == [0, 1], m
    for p, k in zip(polys, m):
        dmax = H.max_dist_to_curve(p, tr[k])
        assert dmax <= tol_max, f"{tomo} filament {k}: max distance {dmax:.3f} voxel to the true centreline"
        cov = H.coverage(tr[k], p, tol=0.75, trim=1.0)
        assert cov >= 0.98, f"{tomo} filament {k}: coverage {cov:.3f}"
        assert H.is_ordered_along(p, tr[k]), f"{tomo} filament {k}: CSV rows are not ordered along it"
    return m


# ------------------------------------------------------------------------------------------
# T6.2 class table
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["filament_masks", "filament_coords"])
def test_G6_C12_class_table_from_config_has_filament_rows(ds, tmp_path, field):
    """Filament classes (from ``filament_masks`` or ``filament_coords``) are ``kind='filament'``
    rows after the particle rows; a particle class number above them stays first."""
    from tomocpt.labels.multiclass import class_table_from_config

    prep = H.prep_config(ds, tmp_path / "td", ((7, "ribo", 120.0),), ((2, "actin", 70.0), (5, "microtubule", 250.0)),
                         **{field: f"2:{tmp_path / 'a'},5:{tmp_path / 'm'}"})
    table = class_table_from_config(prep)
    got = [(r.number, r.name, str(getattr(r.kind, "value", r.kind))) for r in table]
    assert got == [(7, "ribo", "particle"), (2, "actin", "filament"), (5, "microtubule", "filament")], got


# ------------------------------------------------------------------------------------------
# G6-C12 labels from coordinates
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["star_tl", "star_nn", "imod"])
def test_G6_C12_read_filament_coords(ds, kind):
    """``read_filament_coords`` gives one polyline per filament, the written points in order."""
    from tomocpt.labels.filaments import read_filament_coords

    src = _coords_dir(ds.root / f"read_{kind}", kind)
    for tomo in TOMOS:
        f = next(src.glob(f"{tomo}.*"))
        polys = [np.asarray(p, float) for p in read_filament_coords(f, APIX)]
        want = samples(tomo)
        assert len(polys) == 2, f"{kind} {tomo}: {len(polys)} polylines"
        m = H.match_polylines(polys, want)
        assert sorted(m) == [0, 1]
        for p, k in zip(polys, m):
            w = want[k]
            assert p.shape == w.shape, (kind, p.shape, w.shape)
            if np.linalg.norm(p[0] - w[0]) > np.linalg.norm(p[0] - w[-1]):
                w = w[::-1]
            np.testing.assert_allclose(p, w, atol=1e-3, err_msg=f"{kind} {tomo} filament {k}")


@pytest.mark.parametrize("kind", ["star_tl", "star_nn", "imod"])
def test_G6_C12_prepare_data_from_coordinates(prepared, kind):
    """CSV centrelines within 0.5 voxel of the truth (spline fit, default 10 A smoothing,
    spacing 0.5 x voxel = 5 A), and tube targets equal to those of the true centreline (atol
    0.05). Margin: a reference splprep fit of these inputs deviates <= 0.02 voxel and its
    tube target differs by 0.005."""
    from tomocpt.labels.filaments import render_tubes

    tdd, _ = prepared(kind, "filament_coords", kind)
    for tomo in TOMOS:
        polys = _csv_polylines(tdd, tomo)
        _check_centrelines(polys, tomo)
        for p in polys:
            step = np.linalg.norm(np.diff(p, axis=0), axis=1)
            assert abs(np.median(step) - 0.5) <= 0.01, f"label spacing {np.median(step):.4f} voxel (0.5 x voxel)"
        got = render_tubes(np.concatenate([H.segments_from_polyline(p, 0, 2 * R_FIL) for p in polys]), SHAPE, 1)
        ref = render_tubes(np.concatenate([H.segments_from_polyline(t, 0, 2 * R_FIL) for t in truths(tomo)]), SHAPE, 1)
        err = np.abs(got - ref).max()
        assert err <= 0.05, f"{kind} {tomo}: tube target differs by {err:.4f} from the true centreline's"


def test_G6_C12_filament_label_spacing_ang(prepared):
    tdd, _ = prepared("spacing20", "filament_coords", "star_tl", filament_label_spacing_ang=20.0)
    for tomo in TOMOS:
        for p in _csv_polylines(tdd, tomo):
            step = np.linalg.norm(np.diff(p, axis=0), axis=1)
            assert np.all(np.abs(step[:-1] - 2.0) <= 0.02), f"spacing {step.min():.3f}..{step.max():.3f} voxel"


def test_G6_C12_masks_and_coordinates_for_one_class_raise(ds, tmp_path):
    from tomocpt.labels.run import prepare_data

    masks, coords = _coords_dir(tmp_path, "masks"), _coords_dir(tmp_path, "star_tl")
    prep = H.prep_config(ds, tmp_path / "td", PART, FIL, filament_masks=f"5:{masks}", filament_coords=f"5:{coords}")
    with pytest.raises(ValueError):
        prepare_data(prep)


# ------------------------------------------------------------------------------------------
# G6-C13 write_filament_masks
# ------------------------------------------------------------------------------------------

def test_G6_C13_write_filament_masks(prepared, ds):
    tdd, _ = prepared("write_masks", "filament_coords", "star_tl", write_filament_masks=True)
    for tomo in TOMOS:
        hits = sorted(Path(tdd).glob(f"*/filament_masks/microtubule/{tomo}.mrc"))
        assert len(hits) == 1, f"expected <dataset_dir>/filament_masks/microtubule/{tomo}.mrc, got {hits}"
        mask, apix = H.read_mrc(hits[0])
        assert mask.shape == SHAPE and apix == pytest.approx(APIX)
        assert np.issubdtype(mask.dtype, np.integer) and set(np.unique(mask).tolist()) <= {0, 1}
        polys = _csv_polylines(tdd, tomo)
        d = H.distance_field(SHAPE, polys, reach=R_FIL + 3)
        assert np.all(mask[d <= R_FIL - 1] == 1), "voxels within R - 1 of the spline are missing"
        assert np.all(mask[d > R_FIL + 1] == 0), "voxels beyond R + 1 of the spline are in the mask"


def test_G6_C13_default_writes_no_masks(prepared):
    tdd, prep = prepared("star_tl", "filament_coords", "star_tl")
    assert getattr(prep, "write_filament_masks") is False
    assert not list(Path(tdd).rglob("filament_masks")), "write_filament_masks=False must not write masks"
    assert H.filament_csv(tdd, TOMOS[0]).is_file()


# ------------------------------------------------------------------------------------------
# G6-C3 segments stored per chunk
# ------------------------------------------------------------------------------------------

def _gate2_common():
    p = H.GOLDEN_DIR / "gate2_common.py"
    spec = importlib.util.spec_from_file_location("_gate2_common_for_g6_labels", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grid_starts(size, patch=64, stride=32):
    starts = list(range(0, max(size - patch, 0) + 1, stride))
    if starts[-1] != max(size - patch, 0):
        starts.append(max(size - patch, 0))
    return starts


def test_G6_C3_segments_per_chunk_render_the_tomogram_crop(prepared):
    """Masks -> ``prepare_data`` (``mask_to_centerlines``) -> pipeline-v2 chunking. For every
    chunk, ``render_targets(points) + render_tubes(filaments/<chunk>.npy)`` equals the crop of
    the tomogram-level render (particles from ``coords/<tomo>.csv``, tubes from
    ``filaments/<tomo>.csv``) at one grid position (atol 1e-6). Segments that only pass within R
    of the chunk are stored too, otherwise the crop would differ near the chunk border."""
    from tomocpt.dataManager.chunking import do_chunking
    from tomocpt.defaultConfigs.scale_config import ScaleConfig
    from tomocpt.labels.filaments import render_tubes
    from tomocpt.labels.helpers import points_to_working, render_targets
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.utils import read_particles_csvs

    tdd, prep = prepared("masks", "filament_masks", "masks")
    table = class_table_from_config(prep)
    assert [str(getattr(r.kind, "value", r.kind)) for r in table] == ["particle", "filament"]
    C, fi = len(table), 1
    df = read_particles_csvs(str(tdd)).sort_values("tomogram_path").reset_index(drop=True)
    chunks = Path(tdd).parent / "chunks_masks"
    with _gate2_common().inprocess_chunking():
        do_chunking(df, chunkedDataDir=str(chunks), desired_particle_pixel_size=20, n_cpus=1,
                    require_labels=True, use_gpus=False, pipeline_version=2, scale_cfg=ScaleConfig(),
                    class_table=table, seed=42)
    starts = [_grid_starts(s) for s in SHAPE]
    n_seg_files, n_outside = 0, 0
    for tomo in TOMOS:
        row = df[df["tomogram_path"].astype(str).str.endswith(f"{tomo}.mrc")].iloc[0]
        pts = points_to_working(pd.read_csv(row["coords_path"]), APIX, APIX, table)
        big = render_targets(np.asarray(pts, np.float32), SHAPE, C, border_policy="clip")
        polys = H.read_filament_csv(H.filament_csv(tdd, tomo))
        segs = np.concatenate([H.segments_from_polyline(p, fi, 2 * R_FIL) for p in polys])
        big = np.maximum(big, render_tubes(segs, SHAPE, C))
        dirs = [d for split in ("train", "val") for d in sorted((chunks / split).glob(f"*{tomo}*")) if d.is_dir()]
        assert dirs, f"no chunk directory for {tomo}"
        for d in dirs:
            files = sorted((d / "points").glob("*.npy"))
            assert files
            used = set()
            for f in files:
                p = np.load(f)
                fp = d / "filaments" / f.name
                s = np.load(fp) if fp.is_file() else np.zeros((0, 8), np.float32)
                assert s.dtype == np.float32 and s.ndim == 2 and s.shape[1] == 8, (fp, s.dtype, s.shape)
                if len(s):
                    n_seg_files += 1
                    assert np.all(np.round(s[:, 6]) == fi) and np.allclose(s[:, 7], 2 * R_FIL, rtol=1e-6)
                    ends = s[:, :6].reshape(-1, 2, 3)
                    n_outside += int(np.any((ends < 0) | (ends > 63), axis=(1, 2)).sum())
                small = np.maximum(render_targets(p, (64,) * 3, C, border_policy="clip"),
                                   render_tubes(s, (64,) * 3, C))
                best, best_off = np.inf, None
                for oz in starts[0]:
                    for oy in starts[1]:
                        for ox in starts[2]:
                            err = float(np.abs(big[:, oz:oz + 64, oy:oy + 64, ox:ox + 64] - small).max())
                            if err < best:
                                best, best_off = err, (oz, oy, ox)
                assert best <= 1e-6, (f, best)
                if small[fi].any():
                    assert best_off not in used, f
                    used.add(best_off)
    assert n_seg_files > 0, "no chunk stores filament segments"
    assert n_outside > 0, "no stored segment reaches outside its chunk; the border case is not exercised"
