"""G4-C3 (two-level phantom, coordinates), G4-C4 oracle round trip, G4-C6 outputs,
G4-C7 float16 accumulation, G4-C8 voxel size 0.

The oracle replaces the network of every loaded ``InferenceModel`` (``g4_helpers``). Every
channel of the oracle sees the same blobs, so a blob gives the same score in every channel
and cross-class suppression would be a tie: these runs set ``cross_class_nms=False``
(cross-class suppression itself is G4-C5).
"""

import shutil

import numpy as np
import pytest

from gate4 import g4_helpers as H

TOL_C4 = 0.5  # native voxels (spec)


def two_level_phantom():
    """10 A/px, 80x192x192: four 100 A blobs (class 1) and one 500 A blob (class 2)."""
    apix = 10.0
    parts = [(1, 100.0, (40.2, 30.3, 30.6)), (1, 100.0, (39.7, 30.8, 160.4)),
             (1, 100.0, (40.6, 160.1, 30.2)), (1, 100.0, (39.4, 160.6, 160.3)),
             (2, 500.0, (40.2, 96.3, 95.8))]
    vol, coords = H.blob_tomogram((80, 192, 192), apix, parts)
    return vol, coords, apix


def small_big_phantom():
    """5 A/px, 100x240x240: four 120 A blobs (class 3) and one 350 A blob (class 7)."""
    apix = 5.0
    parts = [(3, 120.0, (50.2, 40.3, 40.6)), (3, 120.0, (49.7, 40.8, 199.4)),
             (3, 120.0, (50.6, 199.1, 40.2)), (3, 120.0, (49.4, 199.6, 199.3)),
             (7, 350.0, (50.3, 120.4, 119.7))]
    vol, coords = H.blob_tomogram((100, 240, 240), apix, parts)
    return vol, coords, apix


def class_picks(pred_dir, stem, name):
    """Rows ``[x, y, z, score]`` of ``<stem>_<name>.star`` (empty if the file is absent)."""
    p = pred_dir / f"{stem}_{name}.star"
    if not p.is_file():
        return np.zeros((0, 4))
    parts = list(H.picks_by_class(p).values())
    return np.concatenate(parts) if parts else np.zeros((0, 4))


def _assert_recovered(picks, true_xyz, tol, what):
    d = H.nearest_dist(picks[:, :3], true_xyz)
    assert np.all(d <= tol), f"{what}: distance to nearest pick {d} (tolerance {tol})"


def _assert_same_picks(a, b, tol, what):
    """Same number of picks and a one-to-one correspondence within ``tol`` voxels."""
    assert len(a) >= 1 and a.shape == b.shape, f"{what}: {len(a)} vs {len(b)} picks"
    assert np.all(H.nearest_dist(b[:, :3], a[:, :3]) <= tol), what
    assert np.all(H.nearest_dist(a[:, :3], b[:, :3]) <= tol), what


# ------------------------------------------------------------------------------------------
# shared runs (module scope: one predict call each)
# ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_big_run(tmp_path_factory, v2_ckpts):
    """v2, one level with resampling (5 A/px -> 10.9375 A/px), RELION 3.1 output, maps saved."""
    tmp = tmp_path_factory.mktemp("g4_small_big")
    info = v2_ckpts("small_big")
    vol, coords, apix = small_big_phantom()
    tomo = H.write_mrc(tmp / "tomograms" / "TS_sb.mrc", vol, apix)
    with pytest.MonkeyPatch.context() as mp:
        pd, _ = H.run_predict(tmp / "run", mp, dict(
            weights=info["path"], tomogram_file=tomo, cross_class_nms=False,
            save_prediction_confidence_map=True, predictions_coord_format="relion_31",
            predictions_coord_filename="picks.star"), oracle=H.neg_oracle_factory())
    return dict(tmp=tmp, pred=pd, tomo=tomo, coords=coords, ckpt=info["path"], shape=vol.shape, apix=apix)


@pytest.fixture(scope="module")
def two_level_run(tmp_path_factory, v2_ckpts):
    """v2, two levels (conditioned), float32 accumulation on the CPU, maps saved."""
    tmp = tmp_path_factory.mktemp("g4_two_level")
    info = v2_ckpts("two_level_cond")
    vol, coords, apix = two_level_phantom()
    tomo = H.write_mrc(tmp / "tomograms" / "TS_tl.mrc", vol, apix)
    base = dict(weights=info["path"], tomogram_file=tomo, cross_class_nms=False,
                save_prediction_confidence_map=True, accumulate_device="cpu")
    with pytest.MonkeyPatch.context() as mp:
        pd, _ = H.run_predict(tmp / "f32", mp, dict(base, accumulate_dtype="float32"),
                              oracle=H.neg_oracle_factory(conditioned=True))
    return dict(tmp=tmp, pred=pd, tomo=tomo, coords=coords, base=base, shape=vol.shape)


# ------------------------------------------------------------------------------------------
# G4-C4
# ------------------------------------------------------------------------------------------

def test_G4_C4_v1_oracle_round_trip(tmp_path, monkeypatch, legacy_ckpt):
    """v1 (legacy checkpoint), 150 A at 5 A/px = 30 px, rescaled to 20 px and back (no axis
    needs padding, so this passes with and without the unpad fix)."""
    apix, L = 5.0, 150.0
    parts = [(1, L, (48.3, 60.2, 60.7)), (1, L, (47.6, 60.9, 130.4)),
             (1, L, (49.1, 131.3, 60.2)), (1, L, (48.4, 130.6, 131.8))]
    vol, coords = H.blob_tomogram((96, 192, 192), apix, parts)
    tomo = H.write_mrc(tmp_path / "t" / "TS_v1.mrc", vol, apix)
    pd, _ = H.run_predict(tmp_path, monkeypatch, dict(weights=legacy_ckpt, tomogram_file=tomo, length=L),
                          oracle=H.neg_oracle_factory())
    picks = H.picks_by_class(pd / "tomocpt_coords.star")
    allp = np.concatenate(list(picks.values()))
    _assert_recovered(allp, coords[1], TOL_C4, "v1")


def test_G4_C4_v1_oracle_round_trip_thin_tomogram_with_z_padding(tmp_path, monkeypatch, legacy_ckpt):
    """v1, thin tomogram: 72 native slices at 5 A/px become 48 after the 150 A -> 20 px rescale,
    so Z is padded to 64. With the unpad axis-order fix (human decision, 27 Oct 2026; CLAUDE.md
    1.10) every centre is recovered within 0.5 native voxel. Fails on the legacy code, which
    crops the padding from the wrong axis."""
    apix, L = 5.0, 150.0
    parts = [(1, L, (36.3, 60.2, 60.7)), (1, L, (35.6, 60.9, 130.4)),
             (1, L, (36.8, 131.3, 60.2)), (1, L, (35.4, 130.6, 131.8))]
    vol, coords = H.blob_tomogram((72, 192, 192), apix, parts)
    tomo = H.write_mrc(tmp_path / "t" / "TS_v1thin.mrc", vol, apix)
    pd, _ = H.run_predict(tmp_path, monkeypatch, dict(weights=legacy_ckpt, tomogram_file=tomo, length=L),
                          oracle=H.neg_oracle_factory())
    star = pd / "tomocpt_coords.star"
    assert star.is_file(), "no picks written"
    allp = np.concatenate(list(H.picks_by_class(star).values()))
    _assert_recovered(allp, coords[1], TOL_C4, "v1 thin (Z padded)")


def test_G4_C4_v2_oracle_round_trip_with_resampling(small_big_run):
    r = small_big_run
    for number, name in ((3, "small"), (7, "big")):
        _assert_recovered(class_picks(r["pred"], "picks", name), r["coords"][number], TOL_C4, f"v2 {name}")


# ------------------------------------------------------------------------------------------
# G4-C3
# ------------------------------------------------------------------------------------------

def test_G4_C3_two_level_phantom_every_class_picked_in_one_call(two_level_run):
    r = two_level_run
    for number, name in ((1, "s100"), (2, "l500")):
        assert (r["pred"] / f"TS_tl_{name}.mrc").is_file()
        _assert_recovered(class_picks(r["pred"], "tomocpt_coords", name), r["coords"][number], TOL_C4,
                          f"two-level {name}")


# ------------------------------------------------------------------------------------------
# G4-C6
# ------------------------------------------------------------------------------------------

def test_G4_C6_relion_one_star_per_class_with_original_numbers(small_big_run):
    from tomocpt.labels.star import Star

    pd = small_big_run["pred"]
    assert not (pd / "picks.star").exists(), "v2 writes no merged STAR file"
    stars = sorted(p.name for p in pd.glob("*.star"))
    assert stars == ["picks_big.star", "picks_small.star"], stars
    for number, name in ((3, "small"), (7, "big")):
        s = Star(pd / f"picks_{name}.star")
        df = s.particle_data
        assert s.version == "rln31"
        assert len(df) >= 1
        assert set(df["rlnClassNumber"].astype(int)) == {number}
        assert "rlnAutopickFigureOfMerit" in df.columns
        assert df["rlnAutopickFigureOfMerit"].astype(float).between(0, 1).all()
        assert set(df[s.tomo_label].astype(str)) == {"TS_sb"}


def test_G4_C6_maps_per_class_and_merged(small_big_run):
    r = small_big_run
    for name in ("small", "big"):
        m, a = H.read_map(r["pred"] / f"TS_sb_{name}.mrc")
        assert m.shape == r["shape"] and a == pytest.approx(r["apix"])
    heat, a = H.read_map(r["pred"] / "TS_sb_heatmap.mrc")
    assert heat.dtype == np.float32 and heat.shape == r["shape"] and a == pytest.approx(r["apix"])
    cm, a = H.read_map(r["pred"] / "TS_sb_classmap.mrc")
    assert np.issubdtype(cm.dtype, np.integer) and cm.shape == r["shape"] and a == pytest.approx(r["apix"])
    assert set(np.unique(cm).tolist()) <= {0, 3, 7}
    small, _ = H.read_map(r["pred"] / "TS_sb_small.mrc")
    big, _ = H.read_map(r["pred"] / "TS_sb_big.mrc")
    mx = np.maximum(small.astype(np.float32), big.astype(np.float32))
    np.testing.assert_allclose(heat, mx, atol=1e-3)
    # threshold 0.3 for both classes (no calibration): below it 0, above it a class number
    assert np.all(cm[mx < 0.29] == 0), "voxels below every class threshold must be 0"
    assert np.all(np.isin(cm[mx > 0.31], (3, 7))), "voxels above a threshold must carry a class"
    # a blob centre passes 0.3 in both channels (same oracle value): it must get a class
    x, y, z = np.round(r["coords"][7][0]).astype(int)
    assert int(cm[z, y, x]) in (3, 7)


def _skip_run(r, monkeypatch, tag, **infer):
    pd = r["tmp"] / f"skip_{tag}" / "predictions"
    pd.mkdir(parents=True)
    for p in r["pred"].glob("*.mrc"):
        shutil.copy(p, pd / p.name)
    out, _ = H.run_predict(r["tmp"] / f"skip_{tag}", monkeypatch, dict(
        weights=r["ckpt"], tomogram_file=r["tomo"], cross_class_nms=False, skip_inference=True,
        predictions_dir=pd, **infer))
    return out


def test_G4_C6_skip_inference_rereads_the_class_maps(small_big_run, monkeypatch):
    r = small_big_run
    pd = _skip_run(r, monkeypatch, "relion", predictions_coord_format="relion_31",
                   predictions_coord_filename="picks.star")
    assert not (pd / "picks.star").exists()
    for name in ("small", "big"):
        a, b = class_picks(r["pred"], "picks", name), class_picks(pd, "picks", name)
        _assert_same_picks(a, b, 0.1, name)


def test_G4_C6_relion50_warp_imod_names(small_big_run, monkeypatch):
    from tomocpt.labels.star import Star

    r = small_big_run
    pd = _skip_run(r, monkeypatch, "r50", predictions_coord_format="relion_50",
                   predictions_coord_filename="picks.star")
    assert sorted(p.name for p in pd.glob("*.star")) == ["picks_big.star", "picks_small.star"]
    for number, name in ((3, "small"), (7, "big")):
        s = Star(pd / f"picks_{name}.star")
        assert s.version == "rln50"
        assert set(s.particle_data["rlnClassNumber"].astype(int)) == {number}
        assert "rlnAutopickFigureOfMerit" in s.particle_data.columns

    pd = _skip_run(r, monkeypatch, "warp", predictions_coord_format="warp",
                   predictions_coord_filename="picks.star")
    assert sorted(p.name for p in pd.glob("*.star")) == ["picks_big.star", "picks_small.star"]
    for name in ("small", "big"):
        df = H.read_star_particles(pd / f"picks_{name}.star")
        assert len(df) >= 1 and "rlnCoordinateX" in df.columns



def test_G4_C6_imod_one_model_per_class(small_big_run, monkeypatch):
    """IMOD: ``<tomogram stem>_<class name>.mod`` per class, readable with ``imodmodel.read``
    and holding that class's picks (x, y, z) (``imodmodel==0.1.0`` provides ``imodmodel.write``)."""
    import imodmodel

    r = small_big_run
    pd = _skip_run(r, monkeypatch, "imod", predictions_coord_format="imod")
    assert sorted(p.name for p in pd.glob("*.mod")) == ["TS_sb_big.mod", "TS_sb_small.mod"]
    for number, name in ((3, "small"), (7, "big")):
        df = imodmodel.read(str(pd / f"TS_sb_{name}.mod"))
        _assert_recovered(df[["x", "y", "z"]].to_numpy(float), r["coords"][number], TOL_C4, f"imod {name}")


# ------------------------------------------------------------------------------------------
# G4-C7
# ------------------------------------------------------------------------------------------

def test_G4_C7_float16_cpu_accumulation(two_level_run, monkeypatch):
    r = two_level_run
    pd16, _ = H.run_predict(r["tmp"] / "f16", monkeypatch, dict(r["base"], accumulate_dtype="float16"),
                            oracle=H.neg_oracle_factory(conditioned=True))
    for name in ("s100", "l500", "heatmap"):
        a, _ = H.read_map(r["pred"] / f"TS_tl_{name}.mrc")
        b, _ = H.read_map(pd16 / f"TS_tl_{name}.mrc")
        assert np.abs(a.astype(np.float32) - b.astype(np.float32)).max() <= 1e-3, name
    for name in ("s100", "l500"):
        a, b = class_picks(r["pred"], "tomocpt_coords", name), class_picks(pd16, "tomocpt_coords", name)
        _assert_same_picks(a, b, 0.1, name)


# ------------------------------------------------------------------------------------------
# G4-C8
# ------------------------------------------------------------------------------------------

def test_G4_C8_header_voxel_size_zero_raises_naming_apix_override(tmp_path, monkeypatch, v2_ckpts):
    info = v2_ckpts("two_level_cond")
    tomo = H.write_mrc(tmp_path / "t" / "TS_zero.mrc", H.noise_tomogram((64, 96, 96)), 0.0)
    infer = dict(weights=info["path"], tomogram_file=tomo, save_prediction_confidence_map=True,
                 save_predicted_coords=False)
    with pytest.raises(ValueError, match="apix_override"):
        H.run_predict(tmp_path / "a", monkeypatch, infer, oracle=H.tag_oracle_factory())
    pd, _ = H.run_predict(tmp_path / "b", monkeypatch, dict(infer, apix_override=10.0),
                          oracle=H.tag_oracle_factory())
    m, a = H.read_map(pd / "TS_zero_s100.mrc")
    assert m.shape == (64, 96, 96) and a == pytest.approx(10.0)
