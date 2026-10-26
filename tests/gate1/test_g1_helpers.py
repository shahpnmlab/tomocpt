"""Self-checks of the shared helpers in ``tests/utils`` (frozen after Gate 1).

Not a Gate 1 criterion by themselves (T1.4): they make sure the helpers later gates rely on
do what their docstrings say.
"""

import mrcfile
import numpy as np
import pandas as pd
import pytest
import torch

from utils.phantoms import make_dataset, make_phantom, write_mrc, write_star
from utils.tiny import random_chunk_datamodule, tiny_main_config, tiny_swin_kwargs
from utils.wedge import apply_missing_wedge, missing_wedge_mask, wedge_power_fraction


def test_G1_helpers_wedge():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((32, 48, 64)).astype(np.float32)
    frac = wedge_power_fraction(noise, 60.0)
    assert 0.2 < frac < 0.45  # missing region is ~1/3 of k-space for +/-60
    wedged = apply_missing_wedge(noise, 60.0)
    assert wedged.dtype == np.float32 and wedged.shape == noise.shape
    assert wedge_power_fraction(wedged, 60.0) < 1e-10
    # the other orientation is not empty
    assert wedge_power_fraction(wedged, 60.0, tilt_axis="x") > 0.1
    m = missing_wedge_mask((16, 16, 16), 60.0)
    assert m.dtype == bool and m[0, 0, 0] and m[0, 5, 3] and not m[5, 0, 0] and not m[5, 7, 0]
    # constant along ky for tilt axis y
    assert (m == m[:, :1, :]).all()
    assert missing_wedge_mask((8, 8, 8), 90.0).all()


def test_G1_helpers_phantom_species_and_tubes():
    ph = make_phantom(shape=(64, 128, 128), apix=10.0,
                      species=((1, 120.0, 5, False), (2, 200.0, 3, True)),
                      tubes=((3, 250.0, 1), (4, 60.0, 1)), wedge_deg=60.0, snr=None, seed=3)
    vol, coords, polylines, masks = ph
    assert vol.shape == (64, 128, 128) and vol.dtype == np.float32
    assert set(coords) == {1, 2} and coords[1].shape == (5, 3) and coords[2].shape == (3, 3)
    assert set(polylines) == {3, 4} and set(masks) == {3, 4}
    assert masks[3].dtype == bool and masks[3].any() and masks[4].any()
    # dark contrast at solid-sphere centres (wedge applied, no noise)
    clean = make_phantom(shape=(64, 128, 128), apix=10.0,
                         species=((1, 120.0, 5, False), (2, 200.0, 3, True)),
                         tubes=((3, 250.0, 1), (4, 60.0, 1)), wedge_deg=None, snr=None, seed=3)
    for x, y, z in np.round(clean.coords[1]).astype(int):
        assert clean.volume[z, y, x] < -0.9
    # hollow sphere: centre empty
    for x, y, z in np.round(clean.coords[2]).astype(int):
        assert clean.volume[z, y, x] > -0.1
    # same geometry with and without wedge/noise
    for c in coords:
        np.testing.assert_array_equal(coords[c], clean.coords[c])
    # spheres inside, without overlap
    allc = [(c, 6.0) for c in coords[1]] + [(c, 10.0) for c in coords[2]]
    for i in range(len(allc)):
        (x, y, z), r = allc[i]
        assert r + 2 <= x <= 127 - r - 2 and r + 2 <= y <= 127 - r - 2 and r + 2 <= z <= 63 - r - 2
        for j in range(i):
            assert np.linalg.norm(allc[i][0] - allc[j][0]) >= allc[i][1] + allc[j][1]
    # polylines are inside the volume, xyz order, ~0.5 voxel spacing
    for pl_ in polylines[3]:
        assert pl_.shape[1] == 3 and len(pl_) > 50
        assert (pl_ >= 0).all() and (pl_[:, 0] <= 127).all() and (pl_[:, 2] <= 63).all()
        steps = np.linalg.norm(np.diff(pl_, axis=0), axis=1)
        assert np.allclose(steps, 0.5, atol=1e-6)
    # hollow 250 A tube: the centreline itself is not in the wall mask, the wall is
    pts = np.round(polylines[3][0][::20]).astype(int)
    assert not masks[3][pts[:, 2], pts[:, 1], pts[:, 0]].any()


def test_G1_helpers_phantom_noise_and_border():
    a = make_phantom(seed=1, snr=0.5)
    b = make_phantom(seed=1, snr=0.5)
    np.testing.assert_array_equal(a.volume, b.volume)
    clean = make_phantom(seed=1, snr=None)
    noise = a.volume - clean.volume
    assert noise.var() == pytest.approx(clean.volume.var() / 0.5, rel=0.05)
    border = make_phantom(shape=(32, 64, 64), species=((1, 100.0, 40, False),), allow_border=True,
                          snr=None, seed=0)
    c = border.coords[1]
    assert c.shape == (40, 3)
    assert ((c < 5) | (c > np.array([64, 64, 32]) - 6)).any()


def test_G1_helpers_star_roundtrip_with_legacy_reader(tmp_path):
    from tomocpt.labels.star import Star

    coords = {1: np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 12.0]]), 2: np.array([[5.0, 6.0, 7.0]])}
    path = write_star(tmp_path / "p.star", coords, "TS_01", 10.0)
    star = Star(path)
    assert star.version == "rln31"
    sub, cid = star.get_particle_subset("1")
    assert cid == "1" and len(sub) == 2
    out = star.get_shifted_scaled_coordinates(sub, tomogram_pixel_size=10.0)
    np.testing.assert_array_equal(out["TS_01"], coords[1].astype(int))

    p = write_mrc(tmp_path / "v.mrc", np.zeros((4, 5, 6)), 7.5)
    with mrcfile.open(p) as m:
        assert m.data.shape == (4, 5, 6) and float(m.voxel_size.x) == pytest.approx(7.5)


def test_G1_helpers_make_dataset_runs_legacy_prepare_data(tmp_path):
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.run import prepare_data

    ds = make_dataset(tmp_path / "raw", species=((1, 120.0, 4, False), (2, 200.0, 2, False)), snr=1.0)
    assert len(ds.tomo_paths) == 2 and all(p.is_file() for p in ds.tomo_paths)
    cfg = PrepdataConfig(**ds.legacy_prepdata_fields(tmp_path / "train_data"))
    prepare_data(cfg)
    csvs = sorted((tmp_path / "train_data").rglob("*tracking*.csv"))
    assert len(csvs) == 2
    df = pd.concat([pd.read_csv(c) for c in csvs])
    assert sorted(df["num_particles"].tolist()) == [2, 2, 4, 4]
    for label_path in df["label_path"]:
        with mrcfile.open(label_path) as m:
            assert m.data.shape == ds.shape and m.data.max() == pytest.approx(1.0)


def test_G1_helpers_tiny(tmp_path):
    cfg = tiny_main_config(tmp_path)
    assert cfg.train.network.SwinUNETR.feature_size == 12 and cfg.train.batch_size == 2
    assert cfg.train.use_gpus is False and cfg.train.n_cpus_for_dataloading == 0
    assert tiny_swin_kwargs()["feature_size"] == 12

    b = next(iter(random_chunk_datamodule().train_dataloader()))
    assert set(b) == {"input_data", "target_data"}
    assert b["input_data"].shape == (2, 1, 64, 64, 64) and b["target_data"].shape == (2, 1, 64, 64, 64)

    b = next(iter(random_chunk_datamodule(n=3, n_classes=2, v2=True, batch_size=3).train_dataloader()))
    assert b["input_data"].shape == (3, 1, 64, 64, 64) and b["input_data"].dtype == torch.float32
    assert b["target_data"].shape == (3, 2, 64, 64, 64) and b["target_data"].dtype == torch.float32
    assert 0 <= float(b["target_data"].min()) and float(b["target_data"].max()) <= 1
    assert b["class_available"].shape == (3, 2) and b["class_available"].dtype == torch.bool
    assert b["class_d_vox"].shape == (3, 2) and b["class_d_vox"].dtype == torch.float32
    assert b["apix_work"].shape == (3, 1) and b["apix_work"].dtype == torch.float32
    assert b["dataset_id"].shape == (3,) and b["dataset_id"].dtype == torch.int64
