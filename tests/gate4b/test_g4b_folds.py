"""G4b-C2 (``split_rows_by_fold``) and G4b-C3 (v2 chunking with ``cv``, refusals)."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gate2 import g2_helpers as G2
from gate4b import g4b_helpers as H
from utils.tiny import applied_main_config, tiny_main_config


def _df():
    """9 rows, 7 distinct stems; TS_03 and TS_05 appear in two datasets (rows share a stem)."""
    rows = []
    for ds, stems in (("dataset_1", ["TS_01", "TS_02", "TS_03", "TS_04", "TS_05", "TS_06", "TS_07"]),
                      ("dataset_2", ["TS_03", "TS_05"])):
        for s in stems:
            rows.append(dict(tomogram_path=f"/data/{ds}/{s}.mrc", label_path=f"/l/{ds}/{s}.mrc",
                             particle_diameter_angst=120.0, num_particles=5))
    df = pd.DataFrame(rows)
    df.index = [10 + 3 * i for i in range(len(df))]  # non-trivial index
    return df


def _split(df, K, k, **kw):
    from tomocpt.dataManager.chunking import split_rows_by_fold

    return split_rows_by_fold(df, K, k, **kw)


@pytest.mark.parametrize("K", [2, 3, 7])
def test_G4b_C2_folds_partition_rows_and_stems(K):
    df = _df()
    stems = df["tomogram_path"].map(H.stem)
    val_count = {s: 0 for s in stems.unique()}
    for k in range(K):
        tr, va = _split(df, K, k)
        assert set(tr.index).isdisjoint(va.index)
        assert sorted(set(tr.index) | set(va.index)) == sorted(df.index)
        assert len(tr) + len(va) == len(df)
        tr_st, va_st = set(stems[tr.index]), set(stems[va.index])
        assert tr_st.isdisjoint(va_st), "rows of one stem are split across train and validation"
        assert len(va) > 0
        for s in va_st:
            val_count[s] += 1
    assert all(v == 1 for v in val_count.values()), val_count


def test_G4b_C2_assignment_follows_the_spec_rule_and_is_deterministic():
    df = _df()
    stems = df["tomogram_path"].map(H.stem)
    K = 3
    exp = H.expected_folds(stems, K, seed=42)
    for k in range(K):
        tr, va = _split(df, K, k)
        tr2, va2 = _split(df, K, k, seed=42)
        assert list(va.index) == list(va2.index) and list(tr.index) == list(tr2.index)
        assert set(stems[va.index]) == {s for s, f in exp.items() if f == k}
    exp7 = H.expected_folds(stems, K, seed=7)
    for k in range(K):
        _, va = _split(df, K, k, seed=7)
        assert set(stems[va.index]) == {s for s, f in exp7.items() if f == k}


@pytest.mark.parametrize("K,k", [(1, 0), (0, 0), (8, 0), (3, -1), (3, 3)])
def test_G4b_C2_invalid_arguments_raise(K, k):
    with pytest.raises(ValueError):
        _split(_df(), K, k)


# ------------------------------------------------------------------------------------------
# G4b-C3
# ------------------------------------------------------------------------------------------

DIAMS = {1: 120.0, 2: 300.0}
NAMES = {1: "small", 2: "large"}
SPECIES = ((1, 120.0, 4, False), (2, 300.0, 1, False))


@pytest.fixture(scope="module")
def cv_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("g4b_cv")
    tomos = G2.phantom_tomos(["TS_a", "TS_b", "TS_c"], (64, 96, 96), 10.0, SPECIES, seed0=30)
    df = G2.write_v2_inputs(root, tomos, DIAMS)
    return dict(root=root, df=df, tdd=root / "training_data")


def _cfg(tmp_path, tdd, chunks, version=2, folds=0, fold=0):
    cfg = tiny_main_config(tmp_path)
    for sec in (cfg.train, cfg.shared):
        sec.chunks_dir = Path(chunks)
        sec.training_data_dir = Path(tdd)
    cfg.prepData.training_data_dir = Path(tdd)
    cfg.train.data_pipeline_version = version
    if version == 2:
        cfg.prepData.multiclass = True
        cfg.prepData.class_diameters = ",".join(f"{k}:{v}" for k, v in DIAMS.items())
        cfg.prepData.class_names = ",".join(f"{k}:{v}" for k, v in NAMES.items())
    cfg.train.cv_folds = folds
    cfg.train.cv_fold = fold
    return cfg


def _prepare(cfg):
    from tomocpt.training.helpers import _prepare_data_if_needed

    common = G2.load_common()
    with applied_main_config(cfg) as mc, common.inprocess_chunking():
        _prepare_data_if_needed(mc.train)


def test_G4b_C3_cv_chunks_record_fold_and_refuse_other_cv(tmp_path, cv_inputs):
    from tomocpt.dataManager.chunking import split_rows_by_fold
    from tomocpt.dataManager.metadata import read_chunk_layout

    K, k = 3, 1
    chunks = tmp_path / "chunks"
    cfg = _cfg(tmp_path, cv_inputs["tdd"], chunks, folds=K, fold=k)
    _prepare(cfg)

    layout = read_chunk_layout(chunks)
    assert layout["pipeline_version"] == 2
    assert layout["cv"] == {"folds": K, "fold": k}
    got = sorted(H.stem(e["tomogram_path"]) for e in layout["val_tomograms"])
    _, va = split_rows_by_fold(cv_inputs["df"], K, k)
    assert got == sorted(set(va["tomogram_path"].map(H.stem)))
    exp = H.expected_folds(cv_inputs["df"]["tomogram_path"].map(H.stem), K)
    assert got == sorted(s for s, f in exp.items() if f == k)
    # training chunks hold the other folds only
    train_dirs = {d.name for d in (chunks / "train").iterdir() if d.is_dir()}
    val_dirs = {d.name for d in (chunks / "val").iterdir() if d.is_dir()}
    assert not any(any(g in d for g in got) for d in train_dirs), (train_dirs, got)
    assert val_dirs and all(any(g in d for g in got) for d in val_dirs), (val_dirs, got)

    # the same request again: accepted
    _prepare(cfg)

    for folds, fold in ((K, 2), (2, 1), (0, 0)):
        bad = copy.deepcopy(cfg)
        bad.train.cv_folds, bad.train.cv_fold = folds, fold
        with pytest.raises(RuntimeError):
            _prepare(bad)


def test_G4b_C3_cv_requested_on_chunks_without_cv_raises(tmp_path, cv_inputs):
    """Existing v2 chunks without a ``cv`` entry differ from a cv request."""
    from tomocpt.dataManager.metadata import read_chunk_layout, write_chunk_layout

    chunks = tmp_path / "chunks"
    cfg = _cfg(tmp_path, cv_inputs["tdd"], chunks, folds=3, fold=0)
    _prepare(cfg)
    layout = read_chunk_layout(chunks)
    layout.pop("cv", None)
    write_chunk_layout(chunks, layout)
    with pytest.raises(RuntimeError):
        _prepare(cfg)


def test_G4b_C3_cv_with_pipeline_v1_raises(tmp_path, cv_inputs):
    cfg = _cfg(tmp_path, cv_inputs["tdd"], tmp_path / "chunks_v1", version=1, folds=2, fold=0)
    with pytest.raises(ValueError, match="(?i)cv|fold"):
        _prepare(cfg)
