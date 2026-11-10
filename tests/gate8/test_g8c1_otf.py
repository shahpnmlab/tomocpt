"""Gate 8c.1: tomogram store and on-the-fly crop sampler (spec 8c.1, G8c-C1..C5, C8 in
``test_g8c1_ddp_gloo.py``). CPU only, phantom tomograms, in-process chunking."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from gate2 import g2_helpers as G2
from gate8 import g8c_helpers as H


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("g8c1")
    df = H.write_inputs(root)
    return dict(
        root=root, df=df,
        chunks=H.build(df, root / "chunks"),
        store16=H.build(df, root / "store16", storage="tomogram", volume_dtype="float16"),
        store32=H.build(df, root / "store32", storage="tomogram", volume_dtype="float32"),
    )


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    """Sparse tomograms, empty chunks kept with probability ~0.2 (alpha 100): sampler statistics."""
    root = tmp_path_factory.mktemp("g8c1_sparse")
    df = H.write_sparse_inputs(root)
    return H.build(df, root / "store", storage="tomogram", alpha_drop_empty=100.0)


def _layout(d):
    return json.loads((Path(d) / "chunk_layout.json").read_text())


# ------------------------------------------------------------------------------------- C1
def test_G8c_C1_defaults_are_chunks():
    from tomocpt.defaultConfigs.otf_config import OnTheFlyConfig
    from tomocpt.mainConfig import mainConfig

    assert mainConfig.train.sampling == "chunks"
    assert mainConfig.train.otf == OnTheFlyConfig()
    o = OnTheFlyConfig()
    assert (o.volume_dtype, o.device, o.augment_device, o.samples_per_epoch, o.jitter, o.negative_mask_weight,
            o.seed, o.val_seed, o.val_samples) == ("float16", "auto", "cpu", None, True, 0.0, 42, 1234, None)


def test_G8c_C1_chunk_manifest_has_no_storage_key(built):
    assert "storage" not in _layout(built["chunks"])
    lay = _layout(built["store16"])
    assert lay["storage"] == "tomogram" and lay["volume_dtype"] == "float16"
    rest = {k: v for k, v in lay.items() if k not in ("storage", "volume_dtype", "alpha_drop_empty")}
    assert rest == _layout(built["chunks"])


def test_G8c_C1_storage_mismatch_is_refused(built):
    with pytest.raises(ValueError, match="storage mismatch"):
        G2.make_data(built["store16"])
    with pytest.raises(ValueError, match="storage mismatch"):
        H.otf_data(built["chunks"])


def test_G8c_C1_prepare_refuses_other_storage(tmp_path, built):
    from tomocpt.training.helpers import _prepare_data_if_needed
    from utils.tiny import applied_main_config, tiny_main_config

    cfg = tiny_main_config(tmp_path)
    cfg.train.data_pipeline_version = 2
    cfg.prepData.multiclass = True
    cfg.prepData.class_diameters = ",".join(f"{k}:{v}" for k, v in H.DIAMS.items())
    cfg.prepData.class_names = ",".join(f"{k}:{v}" for k, v in H.NAMES.items())
    for sec in (cfg.train, cfg.shared):
        sec.chunks_dir = built["chunks"]
    cfg.train.sampling = "on_the_fly"
    with applied_main_config(cfg) as mc, pytest.raises(ValueError, match="storage mismatch"):
        _prepare_data_if_needed(mc.train)
    for sec in (cfg.train, cfg.shared):
        sec.chunks_dir = built["store16"]
    cfg.train.sampling = "chunks"
    with applied_main_config(cfg) as mc, pytest.raises(ValueError, match="storage mismatch"):
        _prepare_data_if_needed(mc.train)
    cfg.train.sampling = "on_the_fly"
    with applied_main_config(cfg) as mc:
        _prepare_data_if_needed(mc.train)  # matches: nothing rebuilt, no error


# ------------------------------------------------------------------------------------- C2
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_G8c_C2_store_holds_what_chunking_chunks(built, dtype):
    from tomocpt.dataManager.metadata import read_chunk_meta
    from tomocpt.dataManager.tomostore import read_tomogram_store

    store_root = built["store32" if dtype == "float32" else "store16"]
    n_dirs, n_chunks = 0, 0
    for split in ("train", "val"):
        stores = {Path(s.path).name: s for s in read_tomogram_store(store_root / split)}
        chunk_dirs = {d.name: d for d in G2.chunk_dirs(built["chunks"], (split,))}
        assert set(chunk_dirs) <= set(stores), sorted(set(chunk_dirs) - set(stores))
        for name, cdir in chunk_dirs.items():
            st = stores[name]
            assert st.volume.dtype == np.dtype(dtype)
            assert read_chunk_meta(cdir) == st.meta
            n_dirs += 1
            for loc, (vp, pp) in H.chunk_locs(cdir).items():
                ref = H.as_np(tio.ScalarImage(vp).data[0])
                got = np.asarray(st.volume[loc[0]:loc[0] + 64, loc[1]:loc[1] + 64, loc[2]:loc[2] + 64])
                if dtype == "float32":
                    np.testing.assert_array_equal(got, ref)
                else:
                    np.testing.assert_array_equal(got, ref.astype(np.float16))
                    assert float(np.abs(got.astype(np.float32) - ref).max()) <= 1.5e-3
                pts = np.load(pp)
                np.testing.assert_array_equal(st.points_in_box(loc), pts)
                assert st.points.dtype == np.float32
                n_chunks += 1
    assert n_dirs >= 4 and n_chunks > 20
    # the 6 A/px tomogram was resampled and padded (z) in the store as in chunking
    shapes = [s.shape for split in ("train", "val") for s in read_tomogram_store(store_root / split)]
    assert any(sh[0] == 64 and sh[1] != 128 for sh in shapes), shapes


def test_G8c_C2_store_entries_cover_every_tomogram_and_level(built):
    from tomocpt.dataManager.tomostore import read_tomogram_store

    names = sorted(Path(s.path).name for sp in ("train", "val") for s in read_tomogram_store(built["store16"] / sp))
    assert names == sorted(f"{t}__L0s0" for t in ("TS_a", "TS_b", "TS_c", "TS_d"))
    # nothing chunk-like is written
    assert not list(built["store16"].rglob("*.nii.gz"))


def test_G8c_C2_negative_mask_identical(tmp_path_factory):
    from gate7 import test_g7_negative_masks as G7
    from tomocpt.dataManager.tomostore import read_tomogram_store

    ds = G7.dataset.__wrapped__(tmp_path_factory)
    _, mask_dir, root = ds
    chunks = G7._chunks(ds, "c", negative_masks=f"membrane:{mask_dir}")
    # same inputs, store instead of chunks
    from tomocpt.dataManager import chunking

    orig = chunking.do_chunking

    def _store(*a, **k):
        k["storage"] = "tomogram"
        k["chunkedDataDir"] = str(root / "store")
        return orig(*a, **k)

    chunking.do_chunking = _store
    try:
        G7._chunks(ds, "s", negative_masks=f"membrane:{mask_dir}")
    finally:
        chunking.do_chunking = orig
    n = 0
    for split in ("train", "val"):
        stores = {Path(s.path).name: s for s in read_tomogram_store(root / "store" / split)}
        for cdir in G2.chunk_dirs(chunks, (split,)):
            st = stores[cdir.name]
            assert st.has_negative_mask
            for loc, (vp, _) in H.chunk_locs(cdir).items():
                suffix = vp.name.split(".")[0][len("volumes"):]
                mp = cdir / "negative_masks" / f"negmask{suffix}.npy"
                ref = np.unpackbits(np.load(mp))[: 64 ** 3].reshape(64, 64, 64).astype(bool)
                np.testing.assert_array_equal(st.negative_crop(loc), ref)
                n += 1
    assert n > 0


def test_G8c_C2_cv_pool_store_written_once(tmp_path, built):
    from gate4c import test_g4c_chunks as G4c
    from tomocpt.training.cv import prepare_cv_chunks
    from utils.tiny import applied_main_config

    tdd = built["root"] / "training_data"
    cfg = G4c._cfg(tmp_path, tdd, tmp_path / "cv", folds=2, all_folds=True)
    cfg.prepData.class_diameters = ",".join(f"{k}:{v}" for k, v in H.DIAMS.items())
    cfg.prepData.class_names = ",".join(f"{k}:{v}" for k, v in H.NAMES.items())
    cfg.train.sampling = "on_the_fly"
    with applied_main_config(cfg) as mc, G2.load_common().inprocess_chunking():
        dirs = [Path(p) for p in prepare_cv_chunks(mc.train)]
    pool = (tmp_path / "cv" / "pool").resolve()
    assert _layout(pool)["storage"] == "tomogram"
    seen = set()
    for d in dirs:
        assert _layout(d)["storage"] == "tomogram"
        for split in ("train", "val"):
            for entry in (d / split).iterdir():
                assert entry.is_symlink() and pool in entry.resolve().parents
                seen.add(entry.name)
    assert len(list((pool / "train").glob("*/volume.npy"))) == len(seen) == 4


# ------------------------------------------------------------------------------------- C3
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_G8c_C3_crops_equal_chunk_samples(built, dtype):
    from tomocpt.dataManager.dataloading import VolumeDatsetIO
    from tomocpt.dataManager.otf import OnTheFlyDataset, crop_candidates
    from tomocpt.dataManager.tomostore import read_tomogram_store

    store_root = built["store32" if dtype == "float32" else "store16"]
    stores = read_tomogram_store(store_root / "train")
    by_name = {Path(s.path).name: i for i, s in enumerate(stores)}
    ds = OnTheFlyDataset(stores, transform=None, n_classes=2)
    ref = VolumeDatsetIO.get_dataset_v2(str(built["chunks"] / "train"), False, n_classes=2)
    atol = 0.0 if dtype == "float32" else 1.5e-3
    n = 0
    for i, (vp, tp, cdir) in enumerate(ref.filepath_tuples):
        si = by_name[Path(cdir).name]
        stem = Path(vp).name.split(".")[0]
        loc = tuple(int(v) for v in stem.split("_")[-3:])
        assert loc in {tuple(c) for c in crop_candidates(stores[si]).tolist()}
        got, exp = ds[(si, *loc)], ref[i]
        assert set(got) == set(exp)
        assert float((got["input_data"] - exp["input_data"]).abs().max()) <= atol
        assert got["target_data"].dtype == torch.float32
        torch.testing.assert_close(got["target_data"], exp["target_data"], atol=1e-6, rtol=0)
        for k in ("class_available", "class_d_vox", "apix_work", "dataset_id"):
            assert torch.equal(got[k], exp[k]), k
        n += 1
    assert n > 10


# ------------------------------------------------------------------------------------- C4
def test_G8c_C4_same_seed_same_crops_other_epoch_or_rank_differs(built):
    from tomocpt.dataManager.otf import draw_crops

    data = H.otf_data(built["store16"])
    kw = dict(jitter=True, stride=32, shapes=[s.shape for s in data.stores_train], size=64)
    a = draw_crops(data.train_index, data.train_weights, 200, 7, 0, 0, 2, **kw)
    b = draw_crops(data.train_index, data.train_weights, 200, 7, 0, 0, 2, **kw)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (100, 4)
    for other in (draw_crops(data.train_index, data.train_weights, 200, 7, 1, 0, 2, **kw),
                  draw_crops(data.train_index, data.train_weights, 200, 7, 0, 1, 2, **kw),
                  draw_crops(data.train_index, data.train_weights, 200, 8, 0, 0, 2, **kw)):
        assert not np.array_equal(a, other)
    # jitter stays inside the volume and within one stride of a grid corner
    for si, z, y, x in a.tolist():
        sh = data.stores_train[si].shape
        assert 0 <= z <= sh[0] - 64 and 0 <= y <= sh[1] - 64 and 0 <= x <= sh[2] - 64


def test_G8c_C4_sampler_epochs_and_length(built):
    data = H.otf_data(built["store16"])
    s = data.train_sampler()
    assert len(s) == data.samples_per_epoch
    e0 = list(s)
    assert e0 == list(s)  # same epoch, same crops
    s.set_epoch(1)
    e1 = list(s)
    assert len(e0) == len(e1) == data.samples_per_epoch and e0 != e1
    loader = data.train_dataloader()
    batch = next(iter(loader))
    assert tuple(batch["input_data"].shape) == (2, 1, 64, 64, 64)
    assert tuple(batch["target_data"].shape) == (2, 2, 64, 64, 64)
    assert batch["class_available"].dtype == torch.bool


def _expected_mix(stores, n_classes=2):
    """Independent re-derivation: per candidate, object (chunking's radius-grown rule) or empty,
    keep = 1 or 1 - drop_p; classes centred inside."""
    from tomocpt.dataManager.otf import crop_candidates

    keep, member = [], []
    for st in stores:
        avail = np.array(st.meta.class_available, bool)
        for c in crop_candidates(st).tolist():
            loc = np.asarray(c, np.float32)
            p = st.points
            local = p[:, :3] - loc
            r = p[:, 4:5] / 2.0
            near = np.all((local >= -r) & (local < 64 + r), axis=1)
            obj = bool(avail[p[near, 3].astype(int)].any())
            inside = np.all((local >= 0) & (local < 64), axis=1)
            m = np.zeros(n_classes, bool)
            for k in p[inside, 3].astype(int):
                m[k] = avail[k]
            keep.append(1.0 if obj else 1.0 - st.drop_p)
            member.append(m)
    return np.array(keep), np.array(member)


def test_G8c_C4_empty_fraction_matches_grid_keep_rate(sparse):
    data = H.otf_data(sparse, jitter=False)
    keep, member = _expected_mix(data.stores_train)
    empty = keep < 1.0
    p_empty = float(keep[empty].sum() / keep.sum())
    assert 0.02 < p_empty < 0.98, p_empty  # the check is not vacuous
    assert data.samples_per_epoch == max(1, round(float(keep.sum())))
    from tomocpt.dataManager.otf import draw_crops

    N = 10_000
    crops = draw_crops(data.train_index, data.train_weights, N, 3, 0, 0, 1, jitter=False)
    pos = {tuple(r): i for i, r in enumerate(data.train_index.tolist())}
    got = np.mean([empty[pos[tuple(r)]] for r in crops.tolist()])
    assert abs(got - p_empty) <= 3 * math.sqrt(p_empty * (1 - p_empty) / N), (got, p_empty)


def test_G8c_C4_class_frequency_matches_repeat_factors(sparse):
    from tomocpt.dataManager.otf import draw_crops

    data = H.otf_data(sparse, chunk_sampling="repeat_factor", jitter=False, threshold=0.5)
    keep, member = _expected_mix(data.stores_train)
    f = (keep[:, None] * member).sum(0) / keep.sum()
    r_c = np.where(f > 0, np.maximum(1.0, np.sqrt(0.5 / np.maximum(f, 1e-12))), 1.0)
    r = np.where(member.any(1), np.max(np.where(member, r_c[None], 0), axis=1), 1.0)
    w = keep * r
    np.testing.assert_allclose(data.train_weights, w, rtol=1e-12)
    assert r_c.max() > 1.0  # the rare class is up-weighted (not vacuous)
    N = 10_000
    crops = draw_crops(data.train_index, data.train_weights, N, 5, 0, 0, 1, jitter=False)
    pos = {tuple(row): i for i, row in enumerate(data.train_index.tolist())}
    rows = np.array([pos[tuple(row)] for row in crops.tolist()])
    for c in range(2):
        p = float((w * member[:, c]).sum() / w.sum())
        got = float(member[rows, c].mean())
        assert abs(got - p) <= 3 * math.sqrt(p * (1 - p) / N) + 1e-12, (c, got, p)


# ------------------------------------------------------------------------------------- C5
def test_G8c_C5_validation_set_is_fixed(built):
    a = H.otf_data(built["store16"])
    b = H.otf_data(built["store16"])
    np.testing.assert_array_equal(a.val_crops, b.val_crops)
    assert len(a.val_crops) > 0
    s = a.val_sampler()
    e0 = list(s)
    s.set_epoch(3)
    assert list(s) == e0 == [tuple(r) for r in a.val_crops.tolist()]
    x0 = [a.dataset_val[k]["input_data"] for k in e0[:4]]
    x1 = [b.dataset_val[k]["input_data"] for k in e0[:4]]
    assert all(torch.equal(u, v) for u, v in zip(x0, x1))
    # no augmentation on validation
    assert a.dataset_val.transform is None and a.dataset_val.zoom_aug_max == 1.0
    c = H.otf_data(built["store16"], val_seed=99)
    assert not np.array_equal(a.val_crops, c.val_crops)
    # only validation tomograms
    val_tomos = {s.meta.tomo_name for s in a.stores_val}
    train_tomos = {s.meta.tomo_name for s in a.stores_train}
    assert val_tomos and not (val_tomos & train_tomos)
    assert {a.stores_val[int(r[0])].meta.tomo_name for r in a.val_crops} <= val_tomos


def test_G8c_C5_validation_split_over_ranks(built, monkeypatch):
    from tomocpt.dataManager import otf

    data = H.otf_data(built["store16"], val_samples=7)
    full = [tuple(r) for r in data.val_crops.tolist()]
    assert len(full) == 7
    parts = []
    for rank in (0, 1):
        monkeypatch.setattr(otf, "_rank_world", lambda r=rank: (r, 2))
        s = data.val_sampler()
        parts.append(list(s))
        assert len(s) == len(parts[-1]) == 4
    assert set(parts[0]) | set(parts[1]) == set(full)
    assert parts[0][:3] == full[0::2][:3] and parts[1][:3] == full[1::2]
    # training streams of two ranks differ
    tr = []
    for rank in (0, 1):
        monkeypatch.setattr(otf, "_rank_world", lambda r=rank: (r, 2))
        tr.append(list(data.train_sampler()))
    assert tr[0] != tr[1] and len(tr[0]) == len(tr[1]) == math.ceil(data.samples_per_epoch / 2)
