"""Dataset and augmentation v2: G2-C14 wedge preservation, G2-C15 target stays a LabelMap,
G2-C16 zoom, G2-C18 repeat-factor sampling (the v1 parts are in test_g2_v1_equivalence.py)."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from gate2 import g2_helpers as H
from utils.tiny import applied_main_config
from utils.wedge import apply_missing_wedge, wedge_power_fraction


def _subject(vol, target=None):
    if target is None:
        target = np.zeros((1,) + vol.shape, dtype=np.float32)
    return tio.Subject(input_data=tio.ScalarImage(tensor=torch.from_numpy(np.ascontiguousarray(vol))[None].float()),
                       target_data=tio.LabelMap(tensor=torch.from_numpy(np.ascontiguousarray(target)).float()))


# ------------------------------------------------------------------------------------------
# G2-C14
# ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wedged_noise():
    rng = np.random.default_rng(0)
    vol = apply_missing_wedge(rng.standard_normal((64, 64, 64)).astype(np.float32), 60.0, "y")
    vol = (vol / vol.std()).astype(np.float32)
    assert wedge_power_fraction(vol) < 1e-6
    return vol


@pytest.mark.parametrize("use_mw_aug", [False, True])
def test_G2_C14_flips_keep_the_missing_wedge(wedged_noise, use_mw_aug):
    """rotation_mode='flips' with the default AugmentationConfig otherwise: 50 draws keep
    wedge_power_fraction < 0.02, also with the Fourier missing-wedge augmentation added.

    Flips mode drops RandomElasticDeformation (human decision, 27 Oct 2026: elastic
    deformation moves wedge power by up to 0.16); the OneOf keeps only RandomBlur.
    """
    from tomocpt.dataManager.dataloading import build_training_transforms
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    cfg = AugmentationConfig(rotation_mode="flips", flip_p=0.5, use_mw_aug=use_mw_aug)
    pipe = build_training_transforms(cfg)
    found = list(H.iter_transforms(pipe))
    assert not any(isinstance(t, tio.RandomAffine) for t in found)
    assert not any(isinstance(t, tio.RandomElasticDeformation) for t in found)
    assert any(isinstance(t, tio.RandomBlur) for t in found)
    flips = [t for t in found if isinstance(t, tio.RandomFlip)]
    assert len(flips) == 1
    assert sorted(int(a) for a in flips[0].axes) == [0, 1, 2]
    assert flips[0].flip_probability == pytest.approx(0.5)

    n_changed = 0
    for _ in range(50):
        out = pipe(_subject(wedged_noise.copy()))["input_data"].data[0].numpy()
        assert wedge_power_fraction(out) < 0.02
        if not np.allclose(np.abs(out), np.abs(wedged_noise), atol=1e-5):
            n_changed += 1
    assert n_changed > 0  # flips (or the wedge augmentation) did something


def test_G2_C14_affine_mode_unchanged():
    """rotation_mode='affine' (the default) keeps RandomAffine and RandomElasticDeformation."""
    from tomocpt.dataManager.dataloading import build_training_transforms
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    for cfg in (AugmentationConfig(), AugmentationConfig(rotation_mode="affine")):
        found = list(H.iter_transforms(build_training_transforms(cfg)))
        assert sum(isinstance(t, tio.RandomAffine) for t in found) == 1
        assert sum(isinstance(t, tio.RandomElasticDeformation) for t in found) == 1
        assert not any(isinstance(t, tio.RandomFlip) for t in found)


def test_G2_C14_flip_only_pipeline_order(wedged_noise):
    """Order is kept: spatial (flip) -> Fourier -> intensity (contrast inversion)."""
    from tomocpt.dataManager.dataloading import build_training_transforms
    from tomocpt.dataManager.fourier_augmentations import RandomFourierDegradation
    from tomocpt.dataManager.intensity_augmentations import RandomContrastInversion
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    pipe = build_training_transforms(AugmentationConfig(rotation_mode="flips", flip_p=0.5, use_mw_aug=True))
    top = list(pipe.transforms)
    kinds = []
    for t in top:
        sub = list(H.iter_transforms(t))
        if any(isinstance(s, tio.RandomFlip) for s in sub):
            kinds.append("flip")
        elif isinstance(t, RandomFourierDegradation):
            kinds.append("fourier")
        elif isinstance(t, RandomContrastInversion):
            kinds.append("inversion")
    assert kinds == ["flip", "fourier", "inversion"]


# ------------------------------------------------------------------------------------------
# G2-C15
# ------------------------------------------------------------------------------------------

def _write_v1_chunk(tmp_path, vol, label):
    vol_path = tmp_path / "volumes" / "volumes_0_0_0_0.nii.gz"
    lab_path = tmp_path / "labels_supervised" / "labels_supervised_0_0_0_0.nii.gz"
    vol_path.parent.mkdir(parents=True)
    lab_path.parent.mkdir(parents=True)
    tio.ScalarImage(tensor=torch.from_numpy(vol)[None]).save(vol_path)
    tio.ScalarImage(tensor=torch.from_numpy(label)[None]).save(lab_path)
    return str(vol_path), str(lab_path)


def _sphere_target(shape=(64, 64, 64), centre=(30.0, 34.0, 27.0), d=20.0):
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij")
    r = np.sqrt((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2)
    return H.profile(r, d / 2.0).astype(np.float32)


def test_G2_C15_v1_target_is_a_labelmap(tmp_path):
    """v1 (passes on v2-gate-0-base): the target reaches the transforms as a tio.LabelMap, and
    with every intensity and Fourier augmentation at p=1 (spatial transforms off) it comes
    out unchanged."""
    from tomocpt.dataManager.dataloading import VolumeDatsetIO, build_training_transforms

    rng = np.random.default_rng(0)
    vol = np.clip(rng.standard_normal((64, 64, 64)), -3, 3).astype(np.float32)
    label = _sphere_target()
    paths = _write_v1_chunk(tmp_path, vol, label)

    seen = []

    def probe(subject):
        seen.append(type(subject["target_data"]))
        return subject

    VolumeDatsetIO([paths], transform=probe)[0]
    assert seen == [tio.LabelMap]

    ds = VolumeDatsetIO([paths], transform=build_training_transforms(H.all_intensity_aug()))
    for _ in range(3):
        item = ds[0]
        np.testing.assert_array_equal(item["target_data"][0].numpy(), label)
        assert not np.allclose(item["input_data"][0].numpy(), vol)


def test_G2_C15_v2_intensity_and_fourier_augmentations_leave_the_target_unchanged(v2main):
    """v2: with every intensity and Fourier augmentation at p=1 (spatial transforms off), each
    training sample's target equals the target without them (so it is a LabelMap that
    TorchIO's intensity transforms skip)."""
    plain = H.make_data(v2main["chunks"], augmentation=H.no_spatial_aug())
    heavy = H.make_data(v2main["chunks"], augmentation=H.all_intensity_aug())
    n = len(plain.dataset_training)
    assert n == len(heavy.dataset_training) > 0
    for i in range(n):
        a, b = plain.dataset_training[i], heavy.dataset_training[i]
        assert torch.equal(a["target_data"], b["target_data"]), i
        assert torch.equal(a["class_available"], b["class_available"])
        assert not torch.allclose(a["input_data"].abs(), b["input_data"].abs())


def _rotate(t, interpolation, degrees):
    """30 degree rotation of a LabelMap target with TorchIO, plus the input positions every
    output voxel samples (linear interpolation of coordinate ramps is exact)."""
    shape = t.shape
    ramps = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
    subj = tio.Subject(t=tio.LabelMap(tensor=torch.from_numpy(t)[None]),
                       **{f"r{i}": tio.ScalarImage(tensor=torch.from_numpy(ramps[i])[None]) for i in range(3)})
    aff = tio.Affine(scales=(1, 1, 1), degrees=degrees, translation=(0, 0, 0), center="image",
                     image_interpolation="linear", label_interpolation=interpolation, default_pad_value=0)
    out = aff(subj)
    return out["t"].data[0].numpy(), [out[f"r{i}"].data[0].numpy().astype(np.float64) for i in range(3)]


def test_G2_C15_linear_label_interpolation_keeps_the_peak():
    """label_interpolation='linear' reaches RandomAffine and RandomElasticDeformation (the
    target stays a LabelMap). A 30 degree rotation with that interpolation keeps the target
    maximum close to the analytic peak and the whole target close to the analytic rotated
    target, much closer than nearest-neighbour.

    Tolerances (G2-C15, human decision 27 Oct 2026): d_vox 20, 30 degrees; peak within 5 % of
    the analytic profile sampled at the output grid, max abs error <= 0.06 against the
    analytic rotated profile (nearest-neighbour: 0.11-0.15), and closer than nearest.
    """
    from tomocpt.dataManager.dataloading import build_training_transforms
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    pipe = build_training_transforms(AugmentationConfig(label_interpolation="linear"))
    found = list(H.iter_transforms(pipe))
    affines = [t for t in found if isinstance(t, tio.RandomAffine)]
    elastics = [t for t in found if isinstance(t, tio.RandomElasticDeformation)]
    assert len(affines) == 1 and len(elastics) == 1
    for t in affines + elastics:
        assert str(t.label_interpolation).lower().endswith("linear"), t
    default = build_training_transforms(AugmentationConfig())
    for t in H.iter_transforms(default):
        if isinstance(t, (tio.RandomAffine, tio.RandomElasticDeformation)):
            assert str(t.label_interpolation).lower().endswith("nearest")  # v1 default unchanged

    interp = affines[0].label_interpolation
    errs_lin, errs_nn = [], []
    for centre in [(20.0, 24.0, 40.0), (31.0, 33.0, 29.0), (40.0, 20.0, 22.0)]:
        t = _sphere_target(centre=centre, d=20.0)
        for deg in [(30, 0, 0), (0, 30, 0), (0, 0, 30)]:
            out, pos = _rotate(t, interp, deg)
            r = np.sqrt(sum((p - c) ** 2 for p, c in zip(pos, centre)))
            ana = H.profile(r, 10.0)
            assert abs(out.max() / ana.max() - 1.0) <= 0.05, (centre, deg, out.max(), ana.max())
            errs_lin.append(np.abs(out - ana).max())
            out_nn, _ = _rotate(t, "nearest", deg)
            errs_nn.append(np.abs(out_nn - ana).max())
    assert max(errs_lin) <= 0.06
    assert np.mean(errs_lin) < 0.5 * np.mean(errs_nn)


# ------------------------------------------------------------------------------------------
# G2-C16
# ------------------------------------------------------------------------------------------

def _radius_along_axes(t, peak):
    """Mean over the 6 axis directions of (first zero voxel distance - 0.5)."""
    dists = []
    for axis in range(3):
        for sign in (1, -1):
            p = list(peak)
            k = 0
            while True:
                k += 1
                p[axis] = peak[axis] + sign * k
                if not 0 <= p[axis] < t.shape[axis]:
                    return None
                if t[tuple(p)] == 0:
                    break
            dists.append(k - 0.5)
    return float(np.mean(dists))


def test_G2_C16_zoom_augmentation(v2main):
    """zoom_aug_max=1.5: for every drawn s (= apix_in / apix_work), class_d_vox = input x s
    and apix_work = input / s (rtol 1e-6), 1 <= s <= 1.5; the target radius grows by s
    (+-1 voxel); the 300 A class (d_vox 32 = band_max before zoom) becomes unavailable for
    s > 1."""
    from tomocpt.defaultConfigs.scale_config import ScaleConfig

    ms = H.metas(v2main["chunks"], splits=("train",))
    apix_in = {round(m.apix_work, 9) for m in ms.values()}
    assert len(apix_in) == 1
    apix_in = apix_in.pop()
    d_in = np.array([r.diameter_ang / apix_in for r in v2main["table"]])
    assert d_in[1] == pytest.approx(32.0)

    data = H.make_data(v2main["chunks"], augmentation=H.no_spatial_aug(),
                       scale_cfg=ScaleConfig(zoom_aug_max=1.5), batch_size=1)
    s_seen, n_radius = [], 0
    for epoch in range(4):
        for b in data.train_dataloader():
            apix = float(b["apix_work"][0, 0])
            s = apix_in / apix
            assert 1.0 - 1e-6 <= s <= 1.5 + 1e-6
            s_seen.append(s)
            np.testing.assert_allclose(b["class_d_vox"][0].double().numpy(), d_in * s, rtol=1e-6)
            if s > 1.0 + 1e-6:
                assert not bool(b["class_available"][0, 1])
            assert bool(b["class_available"][0, 0])
            t = b["target_data"][0, 0].numpy()
            # radius of well-separated class-0 peaks
            for flat in np.argsort(t.ravel())[::-1][:200]:
                peak = np.unravel_index(flat, t.shape)
                if t[peak] < 0.9:
                    break
                lo = [p - 1 for p in peak]
                hi = [p + 2 for p in peak]
                if min(lo) < 0 or any(h > 64 for h in hi):
                    continue
                if t[peak] < t[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].max():
                    continue  # not a local maximum
                rad = _radius_along_axes(t, peak)
                if rad is None:
                    continue
                assert abs(rad - d_in[0] / 2.0 * s) <= 1.0, (rad, d_in[0] / 2.0 * s, s)
                n_radius += 1
    assert max(s_seen) > 1.05, "zoom never enlarged a sample"
    assert n_radius >= 3


# ------------------------------------------------------------------------------------------
# G2-C18
# ------------------------------------------------------------------------------------------

def test_G2_C18_repeat_factor_weights_numbers():
    """class 0 in 50 % of chunks -> r = 1; class 1 in 1 % -> sqrt(0.1 / 0.01) = sqrt(10);
    chunks with both -> sqrt(10); empty chunks -> 1."""
    from tomocpt.dataManager.dataloading import repeat_factor_weights

    chunk_classes = [{0, 1}, {1}] + [{0}] * 99 + [set()] * 99   # 200 chunks
    assert sum(0 in s for s in chunk_classes) == 100 and sum(1 in s for s in chunk_classes) == 2
    r = np.asarray(repeat_factor_weights(chunk_classes, 2, 0.1), dtype=np.float64)
    assert r.shape == (200,)
    np.testing.assert_allclose(r[0], math.sqrt(10), rtol=1e-6)
    np.testing.assert_allclose(r[1], math.sqrt(10), rtol=1e-6)
    np.testing.assert_allclose(r[2:101], 1.0, rtol=1e-6)
    np.testing.assert_allclose(r[101:], 1.0, rtol=1e-6)
    # another threshold: r_c = max(1, sqrt(t / f_c))
    r = np.asarray(repeat_factor_weights(chunk_classes, 2, 0.8), dtype=np.float64)
    np.testing.assert_allclose(r[0], math.sqrt(0.8 / 0.01), rtol=1e-6)
    np.testing.assert_allclose(r[2], math.sqrt(0.8 / 0.5), rtol=1e-6)
    np.testing.assert_allclose(r[150], 1.0, rtol=1e-6)


@pytest.fixture(scope="module")
def rfs_chunks(tmp_path_factory):
    """Lattice dataset where chunk membership is unambiguous: tomograms 64 x 192 x 192 at
    10 A/px (no resampling, no padding), particles centred at z = 32 and y, x = 16 mod 32,
    radii 6-7 px, so every particle lies fully inside or fully outside each 64^3 chunk.
    Class 0 (120 A) sits in 16 of the 25 grid chunks, class 1 (140 A) in one."""
    from tomocpt.labels.multiclass import build_class_table

    root = tmp_path_factory.mktemp("g2_rfs")
    c0 = [(48, 48), (48, 144), (144, 48), (144, 144)]
    c1 = [(16, 16)]
    xyz0 = np.array([[x, y, 32.0] for y, x in c0])
    xyz1 = np.array([[x, y, 32.0] for y, x in c1])
    tomos = []
    for i in range(2):
        cz = [(32.0, y, x) for y, x in c0 + c1]
        vol = H.sphere_volume((64, 192, 192), cz, [6.0] * 4 + [7.0], seed=40 + i)
        tomos.append(dict(name=f"TS_{i + 1:02d}", volume=vol, apix=10.0, shape=(64, 192, 192),
                          coords={1: xyz0, 2: xyz1}))
    diam = {1: 120.0, 2: 140.0}
    df = H.write_v2_inputs(root, tomos, diam)
    with applied_main_config(None):
        chunks = H.run_v2_chunking(df, root / "chunks", build_class_table(diam, {1: "a", 2: "b"}))
    return chunks


def test_G2_C18_repeat_factor_train_loader(rfs_chunks):
    """Train loader with chunk_sampling=repeat_factor: WeightedRandomSampler with the chunk
    factors r and num_samples = round(sum(r)); over 20 000 draws class 1's share of chunks is
    within 10 % (relative) of the expected value. The expected value differs from the uniform
    share by more than 10 %, so uniform sampling fails this test."""
    from torch.utils.data import WeightedRandomSampler

    from tomocpt.defaultConfigs.train_config import ChunkSampling

    # threshold 1.0 (not the default 0.1): with ~16 training chunks it gives r = 4 for class 1,
    # so the expected share (~21 %) is far from the uniform share (~6 %). 20 000 draws give a
    # relative standard error of ~1.4 %, so the 10 % tolerance is ~7 sigma.
    t = 1.0
    data = H.make_data(rfs_chunks, augmentation=H.no_spatial_aug(), chunk_sampling=ChunkSampling.repeat_factor,
                       threshold=t, batch_size=4)
    ds = data.dataset_training
    sets = []
    for i in range(len(ds)):
        item = ds[i]
        avail = item["class_available"].numpy()
        tmax = item["target_data"].amax(dim=(1, 2, 3)).numpy()
        sets.append({c for c in range(2) if avail[c] and tmax[c] > 0.5})
    n = len(sets)
    f = np.array([np.mean([c in s for s in sets]) for c in range(2)])
    assert 0 < f[1] < 0.2 and f[0] > 0.3
    r_c = np.maximum(1.0, np.sqrt(t / f))
    r = np.array([max([r_c[c] for c in s], default=1.0) for s in sets])
    has1 = np.array([1 in s for s in sets])
    expected = float((r * has1).sum() / r.sum())
    uniform = float(has1.mean())
    assert abs(expected - uniform) / expected > 0.2

    torch.manual_seed(0)
    loader = data.train_dataloader()
    sampler = loader.sampler
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == int(round(r.sum()))
    np.testing.assert_allclose(np.asarray(sampler.weights, dtype=np.float64), r, rtol=1e-6)
    draws = []
    while len(draws) < 20000:
        draws.extend(int(i) for i in iter(sampler))
    draws = np.array(draws[:20000])
    share = float(has1[draws].mean())
    assert abs(share - expected) / expected <= 0.10, (share, expected, uniform)

    val = data.val_dataloader()
    assert not isinstance(getattr(val, "sampler", None), WeightedRandomSampler)


def test_G2_C18_manifest_lists_validation_tomograms(v2main):
    from tomocpt.dataManager.metadata import read_chunk_layout

    layout = read_chunk_layout(v2main["chunks"])
    entries = layout["val_tomograms"]
    assert entries
    val_stems = set()
    for d, m in H.metas(v2main["chunks"], splits=("val",)).items():
        val_stems.add(Path(str(m.tomo_name)).stem)
    train_stems = {Path(str(m.tomo_name)).stem for m in H.metas(v2main["chunks"], splits=("train",)).values()}
    assert val_stems and val_stems.isdisjoint(train_stems)
    got = set()
    for e in entries:
        assert {"tomogram_path", "coords_path", "dataset_id", "class_available"} <= set(e)
        assert Path(e["coords_path"]).is_file()
        assert len(e["class_available"]) == len(v2main["table"])
        got.add(Path(e["tomogram_path"]).stem)
    assert got == val_stems
