"""G4c-C2: one chunk pool, fold chunk dirs of symlinks, equal to per-fold chunking."""

import copy
import json
import os
from pathlib import Path

import pytest

from gate2 import g2_helpers as G2
from gate4c import g4c_helpers as H
from utils.tiny import applied_main_config, tiny_main_config

DIAMS = {1: 120.0, 2: 300.0}
NAMES = {1: "small", 2: "large"}
SPECIES = ((1, 120.0, 4, False), (2, 300.0, 1, False))
K = 3


@pytest.fixture(scope="module")
def cv_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("g4c_cv")
    tomos = G2.phantom_tomos(["TS_a", "TS_b", "TS_c", "TS_d"], (64, 96, 96), 10.0, SPECIES, seed0=40)
    df = G2.write_v2_inputs(root, tomos, DIAMS)
    return dict(root=root, df=df, tdd=root / "training_data")


def _cfg(tmp_path, tdd, chunks, folds=K, fold=0, all_folds=False):
    cfg = tiny_main_config(tmp_path)
    for sec in (cfg.train, cfg.shared):
        sec.chunks_dir = Path(chunks)
        sec.training_data_dir = Path(tdd)
    cfg.prepData.training_data_dir = Path(tdd)
    cfg.train.data_pipeline_version = 2
    cfg.prepData.multiclass = True
    cfg.prepData.class_diameters = ",".join(f"{k}:{v}" for k, v in DIAMS.items())
    cfg.prepData.class_names = ",".join(f"{k}:{v}" for k, v in NAMES.items())
    cfg.train.cv_folds = folds
    cfg.train.cv_fold = fold
    cfg.train.cv_all_folds = all_folds
    return cfg


def _prepare(cfg):
    from tomocpt.training.helpers import _prepare_data_if_needed

    with applied_main_config(cfg) as mc, G2.load_common().inprocess_chunking():
        _prepare_data_if_needed(mc.train)


def _pool(cfg):
    from tomocpt.training.cv import prepare_cv_chunks

    with applied_main_config(cfg) as mc, G2.load_common().inprocess_chunking():
        return [Path(p) for p in prepare_cv_chunks(mc.train)]


@pytest.fixture(scope="module")
def pooled(tmp_path_factory, cv_inputs):
    tmp = tmp_path_factory.mktemp("g4c_pool")
    chunks = tmp / "cvchunks"
    cfg = _cfg(tmp, cv_inputs["tdd"], chunks, all_folds=True)
    dirs = _pool(cfg)
    return dict(tmp=tmp, chunks=chunks, dirs=dirs, cfg=cfg)


def test_G4c_C2_fold_dirs_are_named_and_ordered(pooled):
    assert pooled["dirs"] == [pooled["chunks"] / f"fold{k}" for k in range(K)]
    assert (pooled["chunks"] / "pool").is_dir()


@pytest.mark.parametrize("k", range(K))
def test_G4c_C2_fold_chunks_equal_per_fold_chunking(tmp_path, cv_inputs, pooled, k):
    ref = tmp_path / f"ref{k}"
    _prepare(_cfg(tmp_path, cv_inputs["tdd"], ref, fold=k))
    fold = pooled["dirs"][k]

    got_layout = json.loads((fold / "chunk_layout.json").read_text())
    ref_layout = json.loads((ref / "chunk_layout.json").read_text())
    assert got_layout == ref_layout
    assert got_layout["cv"] == {"folds": K, "fold": k}

    for split in ("train", "val"):
        got, exp = H.files_under(fold / split), H.files_under(ref / split)
        assert set(got) == set(exp), (split, sorted(set(got) ^ set(exp))[:10])
        assert exp, f"reference {split} split is empty (test is vacuous)"
        bad = [r for r in exp if not H.same_chunk_file(got[r], exp[r])]
        assert not bad, (split, bad[:10])
        # every per-tomogram entry is a symlink into the pool
        pool = (pooled["chunks"] / "pool").resolve()
        for entry in (fold / split).iterdir():
            assert entry.is_symlink(), entry
            assert pool in entry.resolve().parents, (entry, entry.resolve())
    for name in os.listdir(ref):
        if name.startswith("done"):
            assert (fold / name).is_file(), name


def test_G4c_C2_fold_dir_needs_no_chunking(pooled, monkeypatch):
    from tomocpt.dataManager import chunking
    from tomocpt.training import helpers

    def _boom(*a, **k):
        raise AssertionError("chunking ran on a complete fold chunks_dir")

    monkeypatch.setattr(chunking, "do_chunking", _boom)
    monkeypatch.setattr(helpers, "do_chunking", _boom, raising=False)
    for k, d in enumerate(pooled["dirs"]):
        _prepare(_cfg(pooled["tmp"], pooled["cfg"].train.training_data_dir, d, fold=k))


def test_G4c_C2_second_call_reuses_the_pool(pooled, monkeypatch):
    from tomocpt.dataManager import chunking

    def _boom(*a, **k):
        raise AssertionError("the pool was chunked again")

    for name in ("_do_chunking_v2", "chunk_pool_v2", "process_single_tomo_task_v2"):
        if hasattr(chunking, name) and name != "chunk_pool_v2":
            monkeypatch.setattr(chunking, name, _boom)
    monkeypatch.setattr(chunking, "_run_tasks", _boom)
    assert _pool(copy.deepcopy(pooled["cfg"])) == pooled["dirs"]


@pytest.mark.parametrize("change", ["class_table", "chunk_stride"])
def test_G4c_C2_pool_with_other_settings_raises(pooled, change):
    cfg = copy.deepcopy(pooled["cfg"])
    if change == "class_table":
        cfg.prepData.class_diameters = "1:120.0,2:350.0"
    else:
        cfg.train.CHUNK_STRIDE = 16
    with pytest.raises(ValueError):
        _pool(cfg)
