"""G2-C1 (pipeline v1 chunks unchanged) and the pipeline-v1 parts of G2-C18."""

import random

import numpy as np
import pytest
import torch

from gate2 import g2_helpers as H


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


# ------------------------------------------------------------------------------------------
# G2-C1
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["default_alpha", "keep_all"])
def test_G2_C1_v1_chunking_matches_base_golden(v1_run, variant):
    """do_chunking with every new argument at its default, on make_dataset(n_tomos=2), gives
    the same file list and identical decoded arrays as v2-gate-0-base.

    Golden: tests/golden/gate2_v1_chunks.json (make_goldens_gate2.py, base worktree).
    "default_alpha" keeps the legacy empty-chunk drop (seeded RNG); "keep_all" drops nothing,
    so every grid chunk of both tomograms is compared.
    """
    common = H.load_common()
    golden = H.read_json(H.GOLDEN_DIR / "gate2_v1_chunks.json")["variants"][variant]
    got = common.summarize_chunks(v1_run[variant])

    assert sorted(got) == sorted(golden), (
        f"file list differs: missing {sorted(set(golden) - set(got))[:5]}, "
        f"extra {sorted(set(got) - set(golden))[:5]}")
    for rel, ref in golden.items():
        cur = got[rel]
        if "sha256" in ref:
            assert cur["shape"] == ref["shape"] and cur["dtype"] == ref["dtype"], rel
            assert cur["sha256"] == ref["sha256"], (
                f"{rel}: decoded array differs (sum {cur['sum']} vs {ref['sum']}, max {cur['max']} vs {ref['max']})")
        elif "json" in ref:
            # The v1 manifest keeps its v1 keys and values. Invariant 4: a missing
            # pipeline_version means 1, so an explicit 1 is tolerated, nothing else.
            cur_json = dict(cur["json"])
            assert cur_json.pop("pipeline_version", 1) == 1, rel
            assert cur_json == ref["json"], rel
        else:
            assert cur == ref, rel


# ------------------------------------------------------------------------------------------
# G2-C18, pipeline v1 parts
# ------------------------------------------------------------------------------------------

def _first_batches(loader, n):
    out = []
    for i, b in enumerate(loader):
        if i >= n:
            break
        out.append({k: v.clone() for k, v in b.items()})
    return out


def test_G2_C18_v1_data_order_is_a_plain_shuffled_loader(v1_run):
    """Lock today's v1 train loader: same order (and samples) as a plain shuffled DataLoader
    over the same dataset, under the same global seed. Passes on v2-gate-0-base."""
    from torch.utils.data import DataLoader

    from tomocpt.dataManager.dataloading import Data
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    data = Data(data_dir=str(v1_run["default_alpha"]), return_labels=True, batch_size=2,
                workers_for_data=0, augmentation=AugmentationConfig())
    data.setup()
    _seed()
    got = _first_batches(data.train_dataloader(), 4)
    _seed()
    ref = _first_batches(DataLoader(data.dataset_training, batch_size=2, shuffle=True), 4)
    assert len(got) == len(ref) == 4
    for g, r in zip(got, ref):
        assert set(g) == {"input_data", "target_data"}
        for k in g:
            assert torch.equal(g[k], r[k]), k


def test_G2_C18_uniform_gives_the_legacy_chunk_order(v1_run):
    """chunk_sampling=uniform (with the other new Data arguments at their defaults) yields
    the same chunk order and samples as a Data built without the new arguments."""
    from tomocpt.dataManager.dataloading import Data
    from tomocpt.defaultConfigs.train_config import AugmentationConfig, ChunkSampling

    kw = dict(data_dir=str(v1_run["default_alpha"]), return_labels=True, batch_size=2,
              workers_for_data=0, augmentation=AugmentationConfig())
    legacy = Data(**kw)
    legacy.setup()
    uniform = Data(**kw, chunk_sampling=ChunkSampling.uniform, repeat_factor_threshold=0.1)
    uniform.setup()
    for _ in range(2):  # two epochs
        _seed()
        ref = _first_batches(legacy.train_dataloader(), 4)
        _seed()
        got = _first_batches(uniform.train_dataloader(), 4)
        for g, r in zip(got, ref):
            assert set(g) == set(r)
            for k in g:
                assert torch.equal(g[k], r[k]), k
    _seed()
    assert [b["input_data"].sum().item() for b in _first_batches(uniform.val_dataloader(), 3)] == \
           [b["input_data"].sum().item() for b in _first_batches(legacy.val_dataloader(), 3)]


def test_G2_C18_v1_with_repeat_factor_raises(v1_run):
    """Repeat-factor sampling is v2 only: pipeline v1 with chunk_sampling=repeat_factor -> ValueError."""
    from tomocpt.dataManager.dataloading import Data
    from tomocpt.defaultConfigs.train_config import AugmentationConfig, ChunkSampling

    with pytest.raises(ValueError):
        data = Data(data_dir=str(v1_run["default_alpha"]), return_labels=True, batch_size=2,
                    workers_for_data=0, augmentation=AugmentationConfig(), pipeline_version=1,
                    chunk_sampling=ChunkSampling.repeat_factor, repeat_factor_threshold=0.1)
        data.setup()
        next(iter(data.train_dataloader()))
