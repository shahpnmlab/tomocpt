"""G2-C19 (T2.6): chunking workers use the caller's values.

These tests run ``do_chunking`` with its normal Dask ``LocalCluster`` (separate worker
processes, which import a fresh default ``mainConfig``), on the CPU (CUDA hidden through
``CUDA_VISIBLE_DEVICES``, inherited by the spawned workers). They must NOT use
``gate2_common.inprocess_chunking``: the defect only shows with separate processes.

Dataset: two 64 x 128 x 128 tomograms at 10 A/px with one 120 A particle each (12 px
diameter). With ``desired_particle_pixel_size=12`` nothing is resized, so the 64^3 / stride
32 grid has 1 x 3 x 3 = 9 chunks per tomogram, and every chunk equals the crop of the
robust-normalised native volume. The default target (20 px) would resize to 106 x 213 x 213
(108 grid chunks). With the legacy ALPHA = 1 almost every empty chunk is dropped; with
ALPHA = 1e9 none is.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import torchio as tio

from gate2 import g2_helpers as H
from utils.tiny import applied_main_config

SHAPE = (64, 128, 128)


@pytest.fixture(scope="module")
def sparse_v1(tmp_path_factory):
    from utils.phantoms import make_dataset

    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    root = tmp_path_factory.mktemp("g2_c19")
    with applied_main_config(None):
        ds = make_dataset(root / "data", n_tomos=2, shape=SHAPE, species=((1, 120.0, 1, False),), seed=5)
        tdd = root / "training_data"
        prepare_data(PrepdataConfig(**ds.legacy_prepdata_fields(tdd)))
    df = read_particles_csvs(str(tdd)).sort_values("tomogram_path").reset_index(drop=True)
    return df


def _chunk(df, out, monkeypatch, **kw):
    from tomocpt.dataManager.chunking import do_chunking

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    do_chunking(df, chunkedDataDir=str(out), desired_particle_pixel_size=12, n_cpus=2,
                require_labels=True, use_gpus=False, **kw)
    return out


def _volumes(out):
    vols = {}
    for p in sorted(Path(out).glob("*/*/volumes/*.nii.gz")):
        vols.setdefault(p.parent.parent.name, []).append(p)
    return vols


def _check_unresized(df, out, n_per_tomo, starts=(0, 32, 64)):
    """n_per_tomo chunks per tomogram, each equal to the crop of the normalised native volume."""
    from tomocpt.dataManager.preprocessing import load_mrc

    vols = _volumes(out)
    assert len(vols) == 2
    for tp in df["tomogram_path"]:
        stem = Path(tp).stem
        files = vols[stem]
        assert len(files) == n_per_tomo, (stem, len(files))
        native = load_mrc(tp, normalize=True)
        assert native.shape == SHAPE
        for f in files:
            z, y, x = (int(v) for v in re.search(r"_(\d+)_(\d+)_(\d+)\.nii\.gz$", f.name).groups())
            assert z == 0 and y in starts and x in starts, f.name
            chunk = tio.ScalarImage(f).data[0].numpy()
            np.testing.assert_allclose(chunk, native[z:z + 64, y:y + 64, x:x + 64], atol=1e-5)


def test_G2_C19_explicit_values_reach_separate_process_workers(sparse_v1, tmp_path, monkeypatch):
    """desired_particle_pixel_size=12 and alpha_drop_empty=1e9 passed explicitly: no resize,
    no chunk dropped (9 per tomogram)."""
    out = _chunk(sparse_v1, tmp_path / "chunks", monkeypatch, alpha_drop_empty=1e9)
    _check_unresized(sparse_v1, out, 9)


def test_G2_C19_parent_config_reaches_separate_process_workers(sparse_v1, tmp_path, monkeypatch):
    """New kwargs left at None: the parent's mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES
    (1e9) takes effect in the workers (no chunk dropped), and the explicit
    desired_particle_pixel_size argument is honoured."""
    from tomocpt.mainConfig import mainConfig

    mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES = 1e9   # restored by the autouse fixture
    out = _chunk(sparse_v1, tmp_path / "chunks", monkeypatch)
    _check_unresized(sparse_v1, out, 9)


def test_G2_C19_chunk_size_and_stride_kwargs(sparse_v1, tmp_path, monkeypatch):
    """Light check: chunk_stride=64 (chunk_size=64) gives the 1 x 2 x 2 grid."""
    out = _chunk(sparse_v1, tmp_path / "chunks", monkeypatch, alpha_drop_empty=1e9, chunk_size=64,
                 chunk_stride=64)
    _check_unresized(sparse_v1, out, 4, starts=(0, 64))
