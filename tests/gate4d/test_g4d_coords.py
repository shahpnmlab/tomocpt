"""G4d-C1..C4: exact coordinates reach the v2 coordinate CSVs and chunks; v1 outputs unchanged."""

from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest

from utils.phantoms import make_dataset, write_star

SPECIES = ((1, 120.0, 4, False), (2, 200.0, 2, False))


def _fractional(a):
    a = np.asarray(a, float)
    return np.abs(a - np.round(a)).max() > 0.05


# ------------------------------------------------------------------------------------------
# C1 / C2 readers
# ------------------------------------------------------------------------------------------

def _star_with_origins(tmp_path):
    import starfile

    coords = {1: np.array([[10.3, 20.7, 30.55], [40.9, 50.1, 12.25]]), 2: np.array([[5.6, 6.4, 7.75]])}
    path = write_star(tmp_path / "p.star", coords, "TS_01", 10.0)
    d = starfile.read(path)
    parts = d["particles"]
    parts["rlnOriginXAngst"] = [3.0, -4.0, 0.0]
    parts["rlnOriginYAngst"] = [0.0, 5.0, 2.5]
    parts["rlnOriginZAngst"] = [-1.0, 0.0, 0.0]
    d["particles"] = parts
    starfile.write(d, path, overwrite=True)
    return path, parts


@pytest.mark.parametrize("tomo_apix", [10.0, 5.0])
def test_G4d_C1_star_exact_and_legacy(tmp_path, tomo_apix):
    from tomocpt.labels.star import Star

    path, parts = _star_with_origins(tmp_path)
    star = Star(path)
    sub, _ = star.get_particle_subset("all")
    xyz = parts[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float)
    shifts = parts[["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]].to_numpy(float) / 10.0
    want = (xyz - shifts) * (10.0 / tomo_apix)

    legacy = star.get_shifted_scaled_coordinates(sub, tomogram_pixel_size=tomo_apix)["TS_01"]
    exact = star.get_shifted_scaled_coordinates(sub, tomogram_pixel_size=tomo_apix, exact=True)["TS_01"]
    assert legacy.dtype.kind == "i"
    np.testing.assert_array_equal(legacy, want.astype(int))
    assert exact.dtype == np.float64
    np.testing.assert_allclose(exact, want, atol=1e-9)
    assert _fractional(exact)

    c_leg, k_leg = star.get_shifted_scaled_coordinates_with_classes(sub, tomogram_pixel_size=tomo_apix)
    c_ex, k_ex = star.get_shifted_scaled_coordinates_with_classes(sub, tomogram_pixel_size=tomo_apix, exact=True)
    np.testing.assert_array_equal(c_leg["TS_01"], want.astype(int))
    np.testing.assert_allclose(c_ex["TS_01"], want, atol=1e-9)
    np.testing.assert_array_equal(k_leg["TS_01"], k_ex["TS_01"])


def _write_mod(path, xyz):
    import imodmodel

    xyz = np.asarray(xyz, float)
    df = pd.DataFrame({"contour_id": np.zeros(len(xyz), int), "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imodmodel.write(df, str(path))
    return Path(path)


def test_G4d_C2_mod_exact_and_legacy(tmp_path):
    from tomocpt.labels.mod import Mod

    xyz = np.array([[10.25, 20.75, 30.5], [41.9, 50.1, 12.6]])
    _write_mod(tmp_path / "mods" / "TS_01.mod", xyz)
    mod = Mod(tmp_path / "mods")
    legacy = mod.get_coordinates()["TS_01"]
    exact = mod.get_coordinates(exact=True)["TS_01"]
    np.testing.assert_array_equal(legacy, xyz.astype(int))
    assert exact.dtype == np.float64
    np.testing.assert_allclose(exact, xyz, atol=1e-4)  # IMOD stores float32


# ------------------------------------------------------------------------------------------
# C3 prepare_data
# ------------------------------------------------------------------------------------------

def _legacy_masks(tmp, coords_int, radius_ang, tomo_dir, class_id, radius_per_point=None):
    from tomocpt.labels.helpers import create_particle_masks

    out = tmp / "legacy_masks"
    create_particle_masks(coords_int, radius_ang=radius_ang, tomogram_path=tomo_dir, output_path=out,
                          class_id=class_id, radius_ang_per_point=radius_per_point)
    return out / f"class_{class_id}" / "labels"


def _same_mrc(a, b):
    with mrcfile.open(a) as ma, mrcfile.open(b) as mb:
        return np.array_equal(ma.data, mb.data) and float(ma.voxel_size.x) == float(mb.voxel_size.x)


def _csv_xyz(path, cls=None):
    df = pd.read_csv(path)
    if cls is not None:
        df = df[df["class_number"].astype(int) == cls]
    return df[["x", "y", "z"]].to_numpy(float)


def _sorted(a):
    a = np.asarray(a, float).reshape(-1, 3)
    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


def test_G4d_C3_multiclass_star(tmp_path):
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    ds = make_dataset(tmp_path / "data", n_tomos=2, shape=(64, 128, 128), apix=10.0, species=SPECIES, seed=5)
    tdd = tmp_path / "training_data"
    prep = PrepdataConfig(raw_data_dir=str(ds.tomo_dir), training_data_dir=tdd, particle_length_ang="200",
                          coordinate_files=str(ds.star_path), class_id="all", multiclass=True,
                          class_diameters="1:120,2:200", class_names="1:a,2:b")
    prepare_data(prep)
    df = read_particles_csvs(str(tdd)).reset_index(drop=True)
    assert len(df) == 2
    for _, row in df.iterrows():
        name = Path(str(row["tomogram_path"])).stem
        i = ds.tomo_names.index(name)
        coords_int, radii = {}, {}
        all_exact = []
        for c, d in ((1, 120.0), (2, 200.0)):
            got = _csv_xyz(row["coords_path"], c)
            want = ds.coords[i][c]
            np.testing.assert_allclose(_sorted(got), _sorted(want), atol=1e-6)
            all_exact.append(want)
        assert _fractional(np.concatenate(all_exact))
        # v1 label MRC equals the one drawn from the truncated coordinates (in STAR order)
        star_df = pd.read_csv(row["coords_path"])
        xyz = star_df[["x", "y", "z"]].to_numpy(float)
        coords_int[name] = xyz.astype(int)
        radii[name] = star_df["diameter_ang"].to_numpy(float) / 2.0
        ref_dir = _legacy_masks(tmp_path / name, coords_int, 100.0, ds.tomo_dir, "all", radii)
        label = Path(str(row["label_path"]))
        assert _same_mrc(label, ref_dir / f"{name}.labels.mrc")


def test_G4d_C3_single_class_star(tmp_path):
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    ds = make_dataset(tmp_path / "data", n_tomos=2, shape=(64, 128, 128), apix=10.0, species=SPECIES, seed=6)
    tdd = tmp_path / "training_data"
    prep = PrepdataConfig(raw_data_dir=str(ds.tomo_dir), training_data_dir=tdd, particle_length_ang="120",
                          coordinate_files=str(ds.star_path), class_id="1")
    prepare_data(prep)
    df = read_particles_csvs(str(tdd)).reset_index(drop=True)
    assert len(df) == 2
    for _, row in df.iterrows():
        name = Path(str(row["tomogram_path"])).stem
        i = ds.tomo_names.index(name)
        got = _csv_xyz(row["coords_path"])
        np.testing.assert_allclose(_sorted(got), _sorted(ds.coords[i][1]), atol=1e-6)
        assert _fractional(got)
        assert int(row["num_particles"]) == len(ds.coords[i][1])
        ref_dir = _legacy_masks(tmp_path / name, {name: got.astype(int)}, 60.0, ds.tomo_dir, "1")
        assert _same_mrc(Path(str(row["label_path"])), ref_dir / f"{name}.labels.mrc")


def test_G4d_C3_imod(tmp_path):
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig, PrepareDataType
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    ds = make_dataset(tmp_path / "data", n_tomos=2, shape=(64, 128, 128), apix=10.0, species=SPECIES, seed=7)
    mods = tmp_path / "mods"
    for name, c in zip(ds.tomo_names, ds.coords):
        _write_mod(mods / f"{name}.mod", c[1])
    tdd = tmp_path / "training_data"
    prep = PrepdataConfig(raw_data_dir=str(ds.tomo_dir), training_data_dir=tdd, particle_length_ang="120",
                          coordinate_files=str(mods), class_id="all", coordinate_file_type=PrepareDataType.imod)
    prepare_data(prep)
    df = read_particles_csvs(str(tdd)).reset_index(drop=True)
    assert len(df) == 2
    for _, row in df.iterrows():
        name = Path(str(row["tomogram_path"])).stem
        i = ds.tomo_names.index(name)
        got = _csv_xyz(row["coords_path"])
        np.testing.assert_allclose(_sorted(got), _sorted(ds.coords[i][1]), atol=1e-4)
        assert _fractional(got)
        ref_dir = _legacy_masks(tmp_path / name, {name: got.astype(np.float32).astype(int)}, 60.0, ds.tomo_dir, "all")
        assert _same_mrc(Path(str(row["label_path"])), ref_dir / f"{name}.labels.mrc")


# ------------------------------------------------------------------------------------------
# C4 chunk points
# ------------------------------------------------------------------------------------------

def test_G4d_C4_chunk_points_use_exact_coordinates(tmp_path):
    from gate2 import g2_helpers as G2
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    ds = make_dataset(tmp_path / "data", n_tomos=2, shape=(64, 96, 96), apix=10.0, species=SPECIES, seed=8)
    tdd = tmp_path / "training_data"
    prep = PrepdataConfig(raw_data_dir=str(ds.tomo_dir), training_data_dir=tdd, particle_length_ang="200",
                          coordinate_files=str(ds.star_path), class_id="all", multiclass=True,
                          class_diameters="1:120,2:200", class_names="1:a,2:b")
    prepare_data(prep)
    df = read_particles_csvs(str(tdd))
    table = class_table_from_config(prep)
    chunks = G2.run_v2_chunking(df, tmp_path / "chunks", table)
    pts = [np.load(p) for p in sorted(Path(chunks).rglob("points/*.npy"))]
    pts = np.concatenate([p for p in pts if len(p)])
    assert len(pts)
    # with no resampling (band holds both classes at 10 A/px) working voxels are native voxels
    # shifted by the chunk origin, so the fractional parts of the exact coordinates survive
    frac_pts = np.mod(pts[:, :3], 1.0)
    exact = np.concatenate([np.concatenate(list(c.values())) for c in ds.coords])[:, ::-1]  # zyx
    frac_exact = np.mod(exact, 1.0)
    matched = [np.min(np.abs(frac_exact - f).sum(axis=1)) < 1e-4 for f in frac_pts]
    assert np.mean(matched) > 0.99, "chunk points do not carry the exact coordinates' fractional parts"
    assert np.abs(frac_pts).max() > 0.05
