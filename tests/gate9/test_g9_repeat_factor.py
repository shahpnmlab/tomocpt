"""G9-C1 (sampling part), G9-C5: ``train.repeat_factor_max`` caps the per-class repeat factor in
the chunk-file sampler and in on-the-fly sampling; G9-C6: config backfill of the Gate 9 fields."""

import math

import numpy as np
import pytest

from gate8 import g8c_helpers as H

CHUNK_CLASSES = [{0, 1}, {1}] + [{0}] * 99 + [set()] * 99  # f = (0.5, 0.01)


def test_G9_C1_default_is_none():
    from tomocpt.defaultConfigs.train_config import TrainConfig

    assert TrainConfig().repeat_factor_max is None


def test_G9_C5_repeat_factors_helper():
    from tomocpt.dataManager.otf import repeat_factors

    f = np.array([0.5, 0.001, 0.0])
    np.testing.assert_allclose(repeat_factors(f, 0.1), [1.0, 10.0, 1.0])
    np.testing.assert_allclose(repeat_factors(f, 0.1, None), [1.0, 10.0, 1.0])
    np.testing.assert_allclose(repeat_factors(f, 0.1, 4.0), [1.0, 4.0, 1.0])
    np.testing.assert_allclose(repeat_factors(f, 0.1, 20.0), [1.0, 10.0, 1.0])
    with pytest.raises(ValueError):
        repeat_factors(f, 0.1, 0.5)


def test_G9_C5_chunk_weights_capped():
    from tomocpt.dataManager.dataloading import repeat_factor_weights

    ref = np.asarray(repeat_factor_weights(CHUNK_CLASSES, 2, 0.1))
    np.testing.assert_array_equal(ref, repeat_factor_weights(CHUNK_CLASSES, 2, 0.1, max_factor=None))
    capped = np.asarray(repeat_factor_weights(CHUNK_CLASSES, 2, 0.1, max_factor=2.0))
    assert ref[0] == pytest.approx(math.sqrt(10))
    np.testing.assert_allclose(capped[:2], 2.0)
    np.testing.assert_allclose(capped[2:], 1.0)


# ------------------------------------------------------------------ data modules
@pytest.fixture(scope="module")
def chunks(tmp_path_factory):
    root = tmp_path_factory.mktemp("g9_chunks")
    df = H.write_sparse_inputs(root)
    return H.build(df, root / "chunks", storage="chunks", alpha_drop_empty=100.0)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("g9_store")
    df = H.write_sparse_inputs(root)
    return H.build(df, root / "store", storage="tomogram", alpha_drop_empty=100.0)


def _chunk_data(chunks, cap=None, threshold=0.5):
    from tomocpt.dataManager.dataloading import Data
    from tomocpt.defaultConfigs.scale_config import ScaleConfig
    from tomocpt.defaultConfigs.train_config import ChunkSampling

    from gate2 import g2_helpers as G2

    kw = {} if cap is None else {"repeat_factor_max": cap}
    data = Data(data_dir=str(chunks), return_labels=True, batch_size=2, workers_for_data=0,
                augmentation=G2.no_spatial_aug(), pipeline_version=2, scale_cfg=ScaleConfig(),
                chunk_sampling=ChunkSampling.repeat_factor, repeat_factor_threshold=threshold, **kw)
    data.setup()
    return data


def _otf_data(store, cap=None, threshold=0.5):
    from tomocpt.dataManager.otf import OnTheFlyData
    from tomocpt.defaultConfigs.otf_config import OnTheFlyConfig
    from tomocpt.defaultConfigs.scale_config import ScaleConfig
    from tomocpt.defaultConfigs.train_config import ChunkSampling

    from gate2 import g2_helpers as G2

    kw = {} if cap is None else {"repeat_factor_max": cap}
    data = OnTheFlyData(data_dir=str(store), return_labels=True, batch_size=2, workers_for_data=0,
                        augmentation=G2.no_spatial_aug(), pipeline_version=2, scale_cfg=ScaleConfig(),
                        chunk_sampling=ChunkSampling.repeat_factor, repeat_factor_threshold=threshold,
                        otf=OnTheFlyConfig(jitter=False), **kw)
    data.setup()
    return data


def _check_cap(ref, capped, cap):
    f, r_ref = ref.class_frequencies, ref.class_repeat_factors
    assert r_ref.max() > cap > 1.0  # the cap bites (not vacuous)
    np.testing.assert_array_equal(capped.class_frequencies, f)
    np.testing.assert_allclose(capped.class_repeat_factors, np.minimum(r_ref, cap))


def test_G9_C5_chunk_sampler_capped(chunks):
    from tomocpt.dataManager.dataloading import repeat_factor_weights

    ref = _chunk_data(chunks)
    cap = 1.0 + 0.5 * (float(ref.class_repeat_factors.max()) - 1.0)
    capped = _chunk_data(chunks, cap=cap)
    _check_cap(ref, capped, cap)
    sets = capped.dataset_training.chunk_classes()
    np.testing.assert_allclose(capped.sample_weights, repeat_factor_weights(sets, 2, 0.5, max_factor=cap))
    np.testing.assert_array_equal(ref.sample_weights, repeat_factor_weights(sets, 2, 0.5))
    assert capped.sample_weights.max() == pytest.approx(cap)


def test_G9_C5_otf_sampler_capped(store):
    from tomocpt.dataManager.otf import candidate_weights

    ref = _otf_data(store)
    cap = 1.0 + 0.5 * (float(ref.class_repeat_factors.max()) - 1.0)
    capped = _otf_data(store, cap=cap)
    _check_cap(ref, capped, cap)
    np.testing.assert_array_equal(ref.train_index, capped.train_index)
    # weights: keep * r with the capped per-chunk factor
    _, w_ref, _ = candidate_weights(ref.stores_train, 2, "repeat_factor", 0.5)
    _, w_cap, _ = candidate_weights(ref.stores_train, 2, "repeat_factor", 0.5, repeat_factor_max=cap)
    np.testing.assert_array_equal(ref.train_weights, w_ref)
    np.testing.assert_array_equal(capped.train_weights, w_cap)
    assert np.all(w_cap <= w_ref + 1e-12) and np.any(w_cap < w_ref)


def test_G9_C1_otf_default_weights_unchanged(store):
    from tomocpt.dataManager.otf import candidate_weights

    a = candidate_weights(H_stores(store), 2, "repeat_factor", 0.5)
    b = candidate_weights(H_stores(store), 2, "repeat_factor", 0.5, repeat_factor_max=None)
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def H_stores(store):
    from tomocpt.dataManager.tomostore import read_tomogram_store

    return read_tomogram_store(str(store / "train"))


def test_G9_C5_factors_logged():
    from tomocpt.dataManager.dataloading import log_repeat_factors

    names = [f"c{i}" for i in range(15)]
    f = np.linspace(0.001, 0.5, 15)
    r = np.minimum(np.maximum(1.0, np.sqrt(0.1 / f)), 4.0)
    lines = log_repeat_factors(names, f, r, threshold=0.1, max_factor=4.0, top=10)
    text = "\n".join(lines)
    assert "c0" in text and "r_c" in text and "f_c" in text
    assert sum(ln.count("r_c =") for ln in lines) == 10  # top 10 classes only
    assert "capped" in text


# ------------------------------------------------------------------ G9-C6
@pytest.mark.parametrize("path", [("train", "lr_warmup_steps"), ("train", "repeat_factor_max"),
                                  ("train", "discriminative", "contrastive_warmup_steps"),
                                  ("train", "discriminative", "bank_max_age")])
def test_G9_C6_backfill(path):
    from tomocpt.configManager.configManager import backfill_missing_fields, safe_getattr
    from tomocpt.mainConfig import MainConfig

    cfg, defaults = MainConfig(), MainConfig()
    parent = cfg
    for name in path[:-1]:
        parent = getattr(parent, name)
    del parent.__dict__[path[-1]]
    backfill_missing_fields(cfg, defaults)
    assert path[-1] in parent.__dict__
    want = defaults
    for name in path:
        want = getattr(want, name)
    assert safe_getattr(parent, path[-1], object()) == want
