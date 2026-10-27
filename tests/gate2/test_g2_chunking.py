"""Pipeline v2 chunking: G2-C5 (manifest table, mismatch at training start), G2-C8, G2-C9,
G2-C10, G2-C11, G2-C13."""

import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gate2 import g2_helpers as H
from gate2.conftest import MAIN_APIX, MAIN_DIAMETERS, MAIN_NAMES, MAIN_SHAPE
from utils.tiny import applied_main_config, tiny_main_config

SECTION_5_1 = {
    "input_data": (torch.float32, lambda B, C: (B, 1, 64, 64, 64)),
    "target_data": (torch.float32, lambda B, C: (B, C, 64, 64, 64)),
    "class_available": (torch.bool, lambda B, C: (B, C)),
    "class_d_vox": (torch.float32, lambda B, C: (B, C)),
    "apix_work": (torch.float32, lambda B, C: (B, 1)),
    "dataset_id": (torch.int64, lambda B, C: (B,)),
}


def _stem(p):
    return Path(str(p)).stem


def _dirs_of(chunks, stem):
    return [d for d in H.chunk_dirs(chunks) if d.name.startswith(stem + "__") or d.name == stem
            or f"__{stem}__" in d.name]


# ------------------------------------------------------------------------------------------
# G2-C8
# ------------------------------------------------------------------------------------------

def test_G2_C8_manifest_and_chunk_meta(v2main):
    from tomocpt.dataManager.metadata import ChunkMeta, read_chunk_layout

    chunks = v2main["chunks"]
    layout = read_chunk_layout(chunks)
    assert layout is not None
    assert layout["pipeline_version"] == 2
    for key in ("layout", "dir_names", "split", "band", "resample", "scales_per_level", "class_table",
                "val_tomograms"):
        assert key in layout, key
    assert H.same_table(H.table_from_manifest(layout["class_table"]), v2main["table"])
    assert (chunks / "done_labels_True.txt").is_file()

    dirs = H.chunk_dirs(chunks)
    assert dirs
    for d in dirs:
        assert re.search(r"__L\d+s\d+$", d.name), d.name
        assert (d / "chunk_meta.json").is_file() and len(list(d.glob("chunk_meta.json"))) == 1
        assert H.volume_files(d), d
    for d, m in H.metas(chunks).items():
        assert isinstance(m, ChunkMeta)
        assert m.pipeline_version == 2
        assert m.apix_native == pytest.approx(MAIN_APIX)
        assert m.apix_work == pytest.approx(9.375, rel=1e-9)   # band rule: 300 A / 32
        assert len(m.class_available) == len(m.class_d_vox) == 2
        for c, row in enumerate(v2main["table"]):
            assert m.class_d_vox[c] == pytest.approx(row.diameter_ang / m.apix_work, rel=1e-6)
        assert list(m.class_available) == [True, True]


def test_G2_C8_one_task_per_tomogram_when_rows_share_it(v2main_inputs, tmp_path):
    """Two tracking rows per tomogram (one dataset per species, as legacy multi-species data):
    each tomogram is chunked once, at one level, with the points of both classes."""
    tomos = v2main_inputs["tomos"]
    df = H.write_v2_inputs(tmp_path, tomos, MAIN_DIAMETERS, split_rows=True)
    assert len(df) == 4 and df["tomogram_path"].nunique() == 2
    chunks = H.run_v2_chunking(df, tmp_path / "chunks", v2main_inputs["table"])
    for t in tomos:
        dirs = _dirs_of(chunks, t["name"])
        assert len(dirs) == 1, [d.name for d in H.chunk_dirs(chunks)]
        classes = set()
        for p in H.points_files(dirs[0]):
            a = np.load(p)
            classes |= set(a[:, 3].astype(int).tolist()) if len(a) else set()
        assert classes == {0, 1}
    assert len(H.chunk_dirs(chunks)) == 2


def _check_batch(batch, C):
    assert set(SECTION_5_1) <= set(batch), set(batch)
    B = batch["input_data"].shape[0]
    for k, (dtype, shape) in SECTION_5_1.items():
        assert batch[k].dtype == dtype, (k, batch[k].dtype)
        assert tuple(batch[k].shape) == shape(B, C), (k, tuple(batch[k].shape))
    assert batch["target_data"].min() >= 0 and batch["target_data"].max() <= 1 + 1e-6


def test_G2_C8_batches_match_section_5_1(v2main):
    data = H.make_data(v2main["chunks"], batch_size=2)
    C = len(v2main["table"])
    diam = torch.tensor([r.diameter_ang for r in v2main["table"]], dtype=torch.float64)
    for loader in (data.val_dataloader(), data.train_dataloader()):
        n = 0
        for batch in loader:
            _check_batch(batch, C)
            expected = diam[None, :] / batch["apix_work"].double()
            torch.testing.assert_close(batch["class_d_vox"].double(), expected, rtol=1e-6, atol=0)
            n += 1
        assert n > 0


# ------------------------------------------------------------------------------------------
# G2-C9
# ------------------------------------------------------------------------------------------

def _tomo_points_working(inputs, tomo, apix_work, resample):
    """Tomogram-level points at the working scale with the exact map of the resample the
    chunks used (G2-C20), shifted by the v1 padding."""
    from tomocpt.dataManager.scaling import coordinate_map, working_shape
    from tomocpt.defaultConfigs.scale_config import ResampleMethod
    from tomocpt.labels.helpers import points_to_working

    row = inputs["df"][inputs["df"]["tomogram_path"].map(_stem) == tomo["name"]].iloc[0]
    coords = pd.read_csv(row["coords_path"])
    ws = tuple(int(s) for s in working_shape(tomo["shape"], tomo["apix"], apix_work))
    padded = tuple(max(H.CHUNK, s) for s in ws)
    pad = np.array([H.v1_pad_front(s) for s in ws], dtype=np.float64)
    cmap = coordinate_map(tuple(tomo["shape"]), ws, ResampleMethod(str(getattr(resample, "value", resample))))
    pts = np.asarray(points_to_working(coords, tomo["apix"], apix_work, inputs["table"], coord_map=cmap),
                     dtype=np.float64)
    assert pts.shape == (len(coords), 5)
    pts = pts.copy()
    pts[:, :3] += pad
    return pts, padded


def test_G2_C9_stored_points_render_the_tomogram_crop(v2main):
    """For every chunk, render_targets of its stored points equals the crop of a
    tomogram-level render_targets at the same scale (atol 1e-6), particles crossing the chunk
    border included. Each points file must match a distinct TorchIO grid position.

    The tomogram-level points use the exact resample map (G2-C20). v2main resamples 7 -> 9.375 A
    with working shape (47, 95, 95) from (64, 128, 128): N_out/N_in = 0.734 / 0.742 against
    apix_native/apix_work = 0.7467, so the approximate scaling is off by up to ~0.8 voxel."""
    from tomocpt.dataManager.scaling import working_shape

    ws = working_shape(MAIN_SHAPE, MAIN_APIX, 9.375)
    for n_in, n_out in zip(MAIN_SHAPE, ws):
        assert abs(n_out / n_in - MAIN_APIX / 9.375) > 1e-3
    from tomocpt.labels.helpers import render_targets

    C = len(v2main["table"])
    ms = H.metas(v2main["chunks"])
    n_cross = 0
    for tomo in v2main["tomos"]:
        dirs = _dirs_of(v2main["chunks"], tomo["name"])
        assert dirs, tomo["name"]
        for d in dirs:
            apix_work = ms[d].apix_work
            pts, padded = _tomo_points_working(v2main, tomo, apix_work, ms[d].resample)
            big = render_targets(pts.astype(np.float32), padded, C, border_policy="clip")
            starts = [H.grid_starts(s) for s in padded]
            files = H.points_files(d)
            assert len(files) == len(H.volume_files(d)) > 0
            used = set()
            for f in files:
                p = np.load(f)
                assert p.dtype == np.float32 and p.ndim == 2 and p.shape[1] == 5, (f, p.dtype, p.shape)
                small = render_targets(p, (64, 64, 64), C, border_policy="clip")
                best, best_off = np.inf, None
                for oz in starts[0]:
                    for oy in starts[1]:
                        for ox in starts[2]:
                            crop = big[:, oz:oz + 64, oy:oy + 64, ox:ox + 64]
                            err = float(np.abs(crop - small).max())
                            if err < best:
                                best, best_off = err, (oz, oy, ox)
                assert best <= 1e-6, (f, best)
                if small.any():
                    assert best_off not in used, f
                    used.add(best_off)
                for z, y, x, c, dv in p:
                    R = dv / 2.0
                    if min(z, y, x) < R or max(z, y, x) > 64 - R:
                        n_cross += 1
    assert n_cross > 0, "no stored particle crosses a chunk border; the test does not exercise it"


def test_G2_C9_points_file_is_small(tmp_path):
    """A chunk with >= 50 particles stores < 4 kB."""
    from tomocpt.labels.multiclass import build_class_table

    lattice = [8.0 + 12.0 * k + 6.0 for k in range(4)]  # 14, 26, 38, 50
    xyz = np.array([[x, y, z] for z in lattice for y in lattice for x in lattice])
    tomos = [dict(name=f"TS_{i + 1:02d}", apix=10.0, coords={1: xyz}, shape=(64, 64, 64),
                  volume=H.sphere_volume((64, 64, 64), xyz[:, ::-1], [5.0] * len(xyz), seed=i)) for i in range(2)]
    df = H.write_v2_inputs(tmp_path, tomos, {1: 100.0})
    chunks = H.run_v2_chunking(df, tmp_path / "chunks", build_class_table({1: 100.0}, {1: "p100"}))
    big = [f for d in H.chunk_dirs(chunks) for f in H.points_files(d) if len(np.load(f)) >= 50]
    assert big, "no chunk with >= 50 particles"
    for f in big:
        assert f.stat().st_size < 4096, (f, f.stat().st_size)


def _val_samples(chunks):
    data = H.make_data(chunks, batch_size=1)
    out = {}
    for b in data.val_dataloader():
        key = b["input_data"].numpy().tobytes()
        out[key] = {k: v.clone() for k, v in b.items()}
    return out


def test_G2_C9_dense_and_points_give_the_same_batches(v2main, v2main_dense):
    """target_storage='dense' and 'points' give the same batches without augmentation (atol
    1e-6). Samples are matched by their (identical) input volume."""
    a, b = _val_samples(v2main["chunks"]), _val_samples(v2main_dense["chunks"])
    common = set(a) & set(b)
    assert common
    for key in set(a) | set(b):
        side = a.get(key, b.get(key))
        if side["target_data"].max() > 0:
            assert key in common, "a non-empty chunk exists in only one of the two runs"
    for key in common:
        for k in SECTION_5_1:
            if k == "target_data":
                torch.testing.assert_close(a[key][k], b[key][k], rtol=0, atol=1e-6)
            else:
                assert torch.equal(a[key][k], b[key][k]), k


# ------------------------------------------------------------------------------------------
# G2-C10
# ------------------------------------------------------------------------------------------

def test_G2_C10_two_levels(tmp_path):
    """80 A and 400 A species at 10 A/px: chunks at two apix_work (10 and 12.5 A); at each
    level only the in-band class is available (80 A = 8 px is below the band at native size)."""
    from tomocpt.labels.multiclass import build_class_table

    diam = {1: 80.0, 2: 400.0}
    tomos = H.phantom_tomos(["TS_01", "TS_02"], (64, 128, 128), 10.0, ((1, 80.0, 6, False), (2, 400.0, 1, False)),
                            seed0=20)
    df = H.write_v2_inputs(tmp_path, tomos, diam)
    chunks = H.run_v2_chunking(df, tmp_path / "chunks", build_class_table(diam, {1: "small", 2: "big"}))
    ms = H.metas(chunks)
    by_apix = {}
    for d, m in ms.items():
        by_apix.setdefault(round(m.apix_work, 6), []).append((d, m))
    assert set(by_apix) == {10.0, 12.5}, sorted(by_apix)
    for d, m in by_apix[10.0]:
        assert list(m.class_available) == [True, False], d
        assert m.class_d_vox[0] == pytest.approx(8.0) and m.class_d_vox[1] == pytest.approx(40.0)
    for d, m in by_apix[12.5]:
        assert list(m.class_available) == [False, True], d
        assert m.class_d_vox[1] == pytest.approx(32.0)
    for t in tomos:
        levels = {re.search(r"__L(\d+)s\d+$", d.name).group(1) for d in _dirs_of(chunks, t["name"])}
        assert len(levels) == 2, t["name"]


# ------------------------------------------------------------------------------------------
# G2-C11
# ------------------------------------------------------------------------------------------

def _scales3(tmp_path, seed):
    from tomocpt.labels.multiclass import build_class_table

    diam = {1: 150.0, 2: 200.0}
    tomos = H.phantom_tomos(["TS_01", "TS_02"], (64, 96, 96), 10.0, ((1, 150.0, 3, False), (2, 200.0, 2, False)),
                            seed0=30)
    df = H.write_v2_inputs(tmp_path, tomos, diam)
    table = build_class_table(diam, {1: "a", 2: "b"})
    chunks = H.run_v2_chunking(df, tmp_path / "chunks", table, scales_per_level=3, seed=seed)
    return tomos, table, chunks


def test_G2_C11_scales_per_level(tmp_path):
    """scales_per_level=3: three directories per level (s0, s1, s2), distinct apix_work, every
    assigned class inside [lo_c, band_max], never below apix_native; same result for the same
    seed."""
    from tomocpt.dataManager.metadata import read_chunk_layout
    from tomocpt.dataManager.scaling import plan_levels

    tomos, table, chunks = _scales3(tmp_path / "a", seed=42)
    assert read_chunk_layout(chunks)["scales_per_level"] == 3
    levels = plan_levels(10.0, table, *H.BAND)
    assert len(levels) == 1
    ms = H.metas(chunks)
    for t in tomos:
        dirs = _dirs_of(chunks, t["name"])
        ks = sorted(int(re.search(r"__L0s(\d+)$", d.name).group(1)) for d in dirs)
        assert ks == [0, 1, 2], [d.name for d in dirs]
        apixes = [ms[d].apix_work for d in dirs]
        assert len({round(a, 9) for a in apixes}) == 3, apixes
        for d in dirs:
            m = ms[d]
            assert m.apix_work >= 10.0 - 1e-9
            k = int(re.search(r"s(\d+)$", d.name).group(1))
            assert m.scale_index == k and m.level == 0
            if k == 0:
                assert m.apix_work == pytest.approx(levels[0].apix_work, rel=1e-9)
            for c in levels[0].assigned:
                D = table[c].diameter_ang
                lo = min(H.BAND[0], D / 10.0)
                assert lo * (1 - 1e-6) <= m.class_d_vox[c] <= H.BAND[1] * (1 + 1e-6), (d, c, m.class_d_vox)
                assert m.class_d_vox[c] == pytest.approx(D / m.apix_work, rel=1e-6)

    _, _, chunks_b = _scales3(tmp_path / "b", seed=42)
    a = {d.relative_to(chunks).as_posix(): m.apix_work for d, m in ms.items()}
    b = {d.relative_to(chunks_b).as_posix(): m.apix_work for d, m in H.metas(chunks_b).items()}
    assert a == b


# ------------------------------------------------------------------------------------------
# G2-C13 and the training-start part of G2-C5
# ------------------------------------------------------------------------------------------

def _train_cfg(tmp_path, chunks_dir, version, prep_overrides=None):
    cfg = tiny_main_config(tmp_path)
    cfg.train.chunks_dir = Path(chunks_dir)
    cfg.shared.chunks_dir = Path(chunks_dir)
    cfg.train.data_pipeline_version = version
    cfg.train.n_epochs = 1
    cfg.train.OVERFIT_N_BATCHES = 1   # if a guard is missing, legacy code trains one batch, not an epoch
    for k, v in (prep_overrides or {}).items():
        setattr(cfg.prepData, k, v)
    return cfg


def _train(cfg):
    from tomocpt.training.run import train

    with applied_main_config(cfg) as mc:
        train(mc.train)


def test_G2_C13_v2_training_on_v1_chunks_raises(tmp_path, v1_run, chunks_copy):
    chunks = chunks_copy(v1_run["default_alpha"])
    cfg = _train_cfg(tmp_path, chunks, 2, dict(multiclass=True, class_diameters="1:120", class_names="1:p"))
    with pytest.raises(ValueError, match="chunk format"):
        _train(cfg)


def test_G2_C13_v2_training_without_manifest_raises(tmp_path, v1_run, chunks_copy):
    """A missing pipeline_version (here: no manifest at all) means 1."""
    chunks = chunks_copy(v1_run["default_alpha"])
    (chunks / "chunk_layout.json").unlink()
    cfg = _train_cfg(tmp_path, chunks, 2, dict(multiclass=True, class_diameters="1:120", class_names="1:p"))
    with pytest.raises(ValueError, match="chunk format"):
        _train(cfg)


def test_G2_C13_v1_training_on_v2_chunks_raises(tmp_path, v2main, chunks_copy):
    chunks = chunks_copy(v2main["chunks"])
    cfg = _train_cfg(tmp_path, chunks, 1)
    with pytest.raises(ValueError, match="chunk format"):
        _train(cfg)


def _main_prep():
    return dict(multiclass=True, class_diameters=",".join(f"{k}:{v}" for k, v in MAIN_DIAMETERS.items()),
                class_names=",".join(f"{k}:{v}" for k, v in MAIN_NAMES.items()))


def test_G2_C5_training_start_records_and_checks_the_class_table(tmp_path, v2main_inputs):
    """Training data prepared by _prepare_data_if_needed (v2) record exactly
    class_table_from_config(prepData) in chunk_layout.json; with a different table at
    training start, training raises ValueError."""
    from tomocpt.dataManager.metadata import read_chunk_layout
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.training.helpers import _prepare_data_if_needed

    root = tmp_path / "data"
    df = H.write_v2_inputs(root, v2main_inputs["tomos"], MAIN_DIAMETERS)
    assert len(df) == 2
    tdd = root / "training_data"
    chunks = tmp_path / "chunks"
    cfg = _train_cfg(tmp_path, chunks, 2, _main_prep())
    cfg.train.training_data_dir = tdd
    cfg.prepData.training_data_dir = tdd
    cfg.shared.training_data_dir = tdd

    common = H.load_common()
    with applied_main_config(cfg) as mc, common.inprocess_chunking():
        _prepare_data_if_needed(mc.train)
        expected = class_table_from_config(mc.prepData)
    layout = read_chunk_layout(chunks)
    assert layout["pipeline_version"] == 2
    assert H.same_table(H.table_from_manifest(layout["class_table"]), expected)
    assert H.same_table(expected, v2main_inputs["table"])

    # the same table again: the existing chunks are accepted
    with applied_main_config(cfg) as mc:
        _prepare_data_if_needed(mc.train)

    # a different table at training start
    bad = copy.deepcopy(cfg)
    bad.prepData.class_diameters = "1:120,2:250"
    with pytest.raises(ValueError):
        _train(bad)


# ------------------------------------------------------------------------------------------
# Slow: the real execution path (Dask cluster, GPU workers), not the in-process CPU patch
# ------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_G2_C8_gpu_cluster_matches_cpu(v2main, tmp_path):
    """do_chunking v2 with use_gpus=True through its normal cluster gives the same chunk
    directories, chunk_meta, manifest and points as the in-process CPU run of the fast tier
    (points atol 1e-4 voxel, volumes atol 1e-3 after robust normalisation)."""
    import torchio as tio

    from tomocpt.dataManager.chunking import do_chunking
    from tomocpt.dataManager.metadata import read_chunk_layout
    from tomocpt.defaultConfigs.scale_config import ScaleConfig

    gpu = tmp_path / "chunks_gpu"
    do_chunking(v2main["df"], chunkedDataDir=str(gpu), desired_particle_pixel_size=20, n_cpus=4,
                require_labels=True, use_gpus=True, pipeline_version=2, scale_cfg=ScaleConfig(),
                class_table=v2main["table"], seed=42)
    cpu = v2main["chunks"]
    rel = lambda root: sorted(p.relative_to(root).as_posix() for p in root.rglob("*")  # noqa: E731
                              if p.is_file() and not p.relative_to(root).as_posix().startswith(".monai_cache"))
    assert rel(gpu) == rel(cpu)
    lg, lc = read_chunk_layout(gpu), read_chunk_layout(cpu)
    assert lg["pipeline_version"] == lc["pipeline_version"] == 2
    assert H.same_table(H.table_from_manifest(lg["class_table"]), H.table_from_manifest(lc["class_table"]))
    mg, mc = H.metas(gpu), H.metas(cpu)
    for d in mc:
        a, b = mc[d], mg[gpu / d.relative_to(cpu)]
        assert a.apix_work == pytest.approx(b.apix_work, rel=1e-9)
        assert list(a.class_available) == list(b.class_available)
    for r in rel(cpu):
        if r.endswith(".npy"):
            np.testing.assert_allclose(np.load(gpu / r), np.load(cpu / r), atol=1e-4)
        elif r.endswith(".nii.gz"):
            np.testing.assert_allclose(tio.ScalarImage(gpu / r).data.numpy(), tio.ScalarImage(cpu / r).data.numpy(),
                                       atol=1e-3)
