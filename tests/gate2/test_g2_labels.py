"""G2-C5 class table (parsing, order, config), G2-C6 multi-class prepare_data,
G2-C7 render_targets, G2-C12 border policy."""

import mrcfile
import numpy as np
import pandas as pd
import pytest

from gate2 import g2_helpers as H


# ------------------------------------------------------------------------------------------
# G2-C5 (parsing and table order; the manifest / training parts are in test_g2_chunking.py)
# ------------------------------------------------------------------------------------------

def test_G2_C5_parse_tolerates_spaces():
    from tomocpt.labels.multiclass import parse_class_diameters, parse_class_names

    assert parse_class_diameters(" 1 : 120 , 2:130.5,  10 :500") == {1: 120.0, 2: 130.5, 10: 500.0}
    assert parse_class_diameters("1:120,2:130") == {1: 120.0, 2: 130.0}
    assert parse_class_names(" 1 : apo-ferritin , 2:ribosome") == {1: "apo-ferritin", 2: "ribosome"}


@pytest.mark.parametrize("text, name", [
    ("1:120,1:130", "1"),      # duplicate class
    ("1:120, 2", "2"),         # malformed entry (no diameter)
    ("1:120,2:", "2"),         # missing diameter
    ("1:120,3:abc", "abc"),    # not a number
])
def test_G2_C5_parse_diameters_errors_name_the_entry(text, name):
    from tomocpt.labels.multiclass import parse_class_diameters

    with pytest.raises(ValueError, match=name):
        parse_class_diameters(text)


def test_G2_C5_parse_names_duplicate_raises():
    from tomocpt.labels.multiclass import parse_class_names

    with pytest.raises(ValueError, match="4"):
        parse_class_names("4:a,4:b")


def test_G2_C5_build_class_table_order():
    """Particles sorted by class number, then filaments sorted by class number."""
    from tomocpt.labels.multiclass import ClassSpec, build_class_table

    table = build_class_table({9: 300.0, 2: 120.0, 5: 250.0, 7: 180.0}, {9: "d", 2: "a", 5: "mt", 7: "c"},
                              kinds={5: "filament"})
    assert [r.number for r in table] == [2, 7, 9, 5]
    assert all(isinstance(r, ClassSpec) for r in table)
    assert [str(getattr(r.kind, "value", r.kind)) for r in table] == ["particle", "particle", "particle", "filament"]
    assert [r.name for r in table] == ["a", "c", "d", "mt"]
    assert [r.diameter_ang for r in table] == [120.0, 180.0, 300.0, 250.0]
    assert ClassSpec(1, "x", 100.0).kind == "particle"


def test_G2_C5_class_table_from_config_single_class():
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.multiclass import class_table_from_config

    table = class_table_from_config(PrepdataConfig(particle_length_ang="140", class_id="all"))
    assert len(table) == 1
    assert (table[0].number, table[0].name, table[0].diameter_ang) == (1, "all", 140.0)
    assert str(getattr(table[0].kind, "value", table[0].kind)) == "particle"


def test_G2_C5_class_table_from_config_multiclass():
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.multiclass import class_table_from_config

    prep = PrepdataConfig(particle_length_ang="300", multiclass=True,
                          class_diameters="3: 300, 1:120, 2:130", class_names="1:apo,2:bamy,3:ribo")
    table = class_table_from_config(prep)
    assert [(r.number, r.name, r.diameter_ang) for r in table] == [(1, "apo", 120.0), (2, "bamy", 130.0),
                                                                   (3, "ribo", 300.0)]


def test_G2_C5_class_without_diameter_raises_naming_it():
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.multiclass import class_table_from_config

    prep = PrepdataConfig(particle_length_ang="300", multiclass=True,
                          class_diameters="1:120,2:130", class_names="1:apo,2:bamy,7:ghost")
    with pytest.raises(ValueError, match="7"):
        class_table_from_config(prep)


# ------------------------------------------------------------------------------------------
# G2-C7
# ------------------------------------------------------------------------------------------

def _dist(shape, centre):
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij")
    return np.sqrt((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2)


def test_G2_C7_render_targets_isolated_particles():
    from tomocpt.labels.helpers import render_targets

    shape = (64, 60, 62)
    pts = np.array([
        [20.3, 30.7, 25.2, 0, 12.0],
        [40.0, 20.0, 44.6, 1, 20.0],
        [44.5, 45.5, 15.5, 0, 9.0],   # below the band, half-voxel centre
    ], dtype=np.float32)
    t = render_targets(pts, shape, 3, border_policy="clip")
    assert isinstance(t, np.ndarray) and t.dtype == np.float32
    assert t.shape == (3,) + shape
    assert t.min() >= 0.0 and t.max() <= 1.0 + 1e-6
    assert np.all(t[2] == 0)  # no particle of class 2
    for z, y, x, c, d in pts:
        c, R = int(c), d / 2.0
        r = _dist(shape, (z, y, x))
        near = r <= R + 1
        # maximum within 1 voxel of the centre
        masked = np.where(near, t[c], -1)
        am = np.unravel_index(np.argmax(masked), shape)
        assert np.linalg.norm(np.array(am) - np.array([z, y, x])) <= 1.0
        # support: > 0 inside R - 0.5, == 0 beyond R + 1 (within this particle's neighbourhood)
        assert np.all(t[c][r < R - 0.5] > 0)
        ring = (r > R + 1) & (r < R + 4)
        assert np.all(t[c][ring] == 0)
        # only in its own channel
        for other in range(3):
            if other != c:
                assert np.all(t[other][near] == 0)


def test_G2_C7_integer_centre_peak_is_one_and_profile_is_max_combined():
    from tomocpt.labels.helpers import render_targets

    shape = (48, 48, 48)
    pts = np.array([[24, 24, 20, 0, 16.0], [24, 24, 26, 0, 16.0]], dtype=np.float32)  # same class, overlapping
    t = render_targets(pts, shape, 1, border_policy="clip")
    assert t[0, 24, 24, 20] >= 0.99 and t[0, 24, 24, 26] >= 0.99
    assert t.max() <= 1.0 + 1e-6  # max-combined, not summed


def test_G2_C7_overlapping_particles_of_different_classes_keep_both_peaks():
    from tomocpt.labels.helpers import render_targets

    shape = (64, 64, 64)
    p0, p1 = (32.0, 32.0, 30.0), (32.0, 32.0, 33.0)
    pts = np.array([[*p0, 0, 16.0], [*p1, 1, 16.0]], dtype=np.float32)
    t = render_targets(pts, shape, 2, border_policy="clip")
    for c, p in ((0, p0), (1, p1)):
        am = np.unravel_index(np.argmax(t[c]), shape)
        assert np.linalg.norm(np.array(am) - np.array(p)) <= 1.0
        assert t[c][tuple(int(v) for v in p)] >= 0.99


def test_G2_C7_empty_points():
    from tomocpt.labels.helpers import render_targets

    t = render_targets(np.zeros((0, 5), dtype=np.float32), (64, 64, 64), 2, border_policy="clip")
    assert t.shape == (2, 64, 64, 64) and t.dtype == np.float32 and not t.any()


# ------------------------------------------------------------------------------------------
# G2-C12
# ------------------------------------------------------------------------------------------

def test_G2_C12_render_skip_drops_and_clip_draws_a_border_particle():
    """A particle 3 voxels from the edge (R = 6): absent with "skip" (v1 rule), partly drawn
    with "clip". The interior particle is drawn identically by both."""
    from tomocpt.labels.helpers import render_targets

    shape = (64, 64, 64)
    edge, inner = (32.0, 32.0, 3.0), (20.0, 20.0, 40.0)
    pts = np.array([[*edge, 0, 12.0], [*inner, 0, 12.0]], dtype=np.float32)
    skip = render_targets(pts, shape, 1, border_policy="skip")
    clip = render_targets(pts, shape, 1, border_policy="clip")
    r_edge, r_in = _dist(shape, edge), _dist(shape, inner)
    assert not skip[0][r_edge <= 7].any()
    assert clip[0][32, 32, 3] >= 0.99
    assert np.all(clip[0][r_edge < 5.5] > 0)            # in-bounds part drawn (x from 0)
    assert clip[0][32, 32, 0] > 0
    np.testing.assert_array_equal(skip[0][r_in <= 7], clip[0][r_in <= 7])
    assert skip[0][20, 20, 40] >= 0.99


def _border_dataset(tmp_path):
    """One 64x96x96 tomogram (10 A/px), class 1 (120 A, R = 6 px): one particle 3 voxels from
    the X edge, one interior. Plus a second tomogram (prepare_data takes a directory)."""
    from utils.phantoms import write_mrc, write_star

    tomo_dir = tmp_path / "tomograms"
    vol = H.sphere_volume((64, 96, 96), [(32, 40, 3), (32, 48, 60)], [6, 6])
    write_mrc(tomo_dir / "TS_01.mrc", vol, 10.0)
    write_mrc(tomo_dir / "TS_02.mrc", H.sphere_volume((64, 96, 96), [(30, 30, 30)], [6], seed=1), 10.0)
    coords = [{1: np.array([[3.0, 40.0, 32.0], [60.0, 48.0, 32.0]])}, {1: np.array([[30.0, 30.0, 30.0]])}]
    star = write_star(tmp_path / "particles.star", coords, ["TS_01", "TS_02"], 10.0)
    return tomo_dir, star


def _prepare_border(tmp_path, **extra):
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    tomo_dir, star = _border_dataset(tmp_path)
    tdd = tmp_path / "training_data"
    prep = PrepdataConfig(raw_data_dir=str(tomo_dir), training_data_dir=tdd, particle_length_ang="120",
                          coordinate_files=str(star), class_id="1", **extra)
    prepare_data(prep)
    df = read_particles_csvs(str(tdd))
    row = df[df["tomogram_path"].map(lambda p: p.endswith("TS_01.mrc"))].iloc[0]
    with mrcfile.open(row["label_path"]) as m:
        return m.data.copy()


def test_G2_C12_v1_label_skip_is_the_default(tmp_path):
    """Legacy behaviour (passes on v2-gate-0-base): the border particle is absent from the v1
    label MRC, the interior one is drawn."""
    label = _prepare_border(tmp_path)
    assert not label[25:40, 33:48, 0:10].any()
    assert label[32, 48, 60] == pytest.approx(1.0)


def test_G2_C12_v1_label_skip_explicit_matches_default(tmp_path):
    a = _prepare_border(tmp_path / "a")
    b = _prepare_border(tmp_path / "b", label_border_policy="skip")
    np.testing.assert_array_equal(a, b)


def test_G2_C12_v1_label_clip_draws_the_border_particle(tmp_path):
    """label_border_policy='clip' also applies to v1 label MRCs: the in-bounds part of the
    border particle is drawn; the interior particle is unchanged."""
    skip = _prepare_border(tmp_path / "skip")
    clip = _prepare_border(tmp_path / "clip", label_border_policy="clip")
    assert clip[32, 40, 3] == pytest.approx(1.0)
    assert clip[32, 40, 0] > 0
    np.testing.assert_array_equal(clip[:, :, 20:], skip[:, :, 20:])


# ------------------------------------------------------------------------------------------
# G2-C6
# ------------------------------------------------------------------------------------------

C6_SPECIES = ((1, 120.0, 4, False), (2, 200.0, 3, False), (3, 300.0, 2, False))


def test_G2_C6_multiclass_prepare_data(tmp_path):
    """A STAR with three classes on two tomograms gives one dataset; every tomogram gets a
    coordinate CSV with all classes (native voxels, class diameters); the tracking CSV gets
    coords_path and apix_native; class_annotated is respected in the chunks."""
    from utils.phantoms import make_dataset

    from tomocpt.dataManager.metadata import read_chunk_layout
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    ds = make_dataset(tmp_path / "data", n_tomos=2, shape=(64, 128, 128), apix=10.0, species=C6_SPECIES, seed=3)
    tdd = tmp_path / "training_data"
    prep = PrepdataConfig(raw_data_dir=str(ds.tomo_dir), training_data_dir=tdd, particle_length_ang="300",
                          coordinate_files=str(ds.star_path), class_id="all", multiclass=True,
                          class_diameters="1:120,2:200,3:300", class_names="1:small,2:medium,3:large",
                          class_annotated="2")
    prepare_data(prep)

    tracking = sorted(tdd.glob("*/*.csv"))
    assert len({p.parent for p in tracking}) == 1, f"expected one dataset, got {tracking}"
    df = read_particles_csvs(str(tdd)).reset_index(drop=True)
    assert len(df) == 2 and df["tomogram_path"].nunique() == 2
    assert {"coords_path", "apix_native"} <= set(df.columns)
    assert np.allclose(df["apix_native"].astype(float), 10.0)

    diam = {1: 120.0, 2: 200.0, 3: 300.0}
    for _, row in df.iterrows():
        coords = pd.read_csv(row["coords_path"])
        assert {"x", "y", "z", "class_number", "diameter_ang"} <= set(coords.columns)
        assert set(coords["class_number"].astype(int)) == {1, 2, 3}
        i = ds.tomo_names.index(str(row["tomogram_path"]).rsplit("/", 1)[-1][:-4])
        for c in (1, 2, 3):
            sub = coords[coords["class_number"].astype(int) == c]
            assert np.allclose(sub["diameter_ang"].astype(float), diam[c])
            got = np.sort(sub[["x", "y", "z"]].to_numpy(float), axis=0)
            want = np.sort(ds.coords[i][c], axis=0)
            assert got.shape == want.shape
            assert np.abs(got - want).max() <= 1.0  # legacy STAR reading truncates to whole voxels

    table = class_table_from_config(prep)
    assert [r.number for r in table] == [1, 2, 3]
    chunks = H.run_v2_chunking(df, tmp_path / "chunks", table)
    ms = H.metas(chunks)
    assert ms
    for d, m in ms.items():
        assert list(m.class_available) == [False, True, False], d
    layout = read_chunk_layout(chunks)
    assert layout["val_tomograms"]
    for e in layout["val_tomograms"]:
        assert list(e["class_available"]) == [False, True, False]
