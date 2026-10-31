"""G6-C1: with no filament class, chunks, batches, model outputs and predictions equal Gate 4
(atol 1e-6).

Golden: ``tests/golden/gate6_c1.npz`` / ``.json``, written by ``make_goldens_gate6.py`` on the
``v2-dev`` code before Gate 6 (commit in the json). ``gate6_common`` runs the same pipeline on
the code under test: multi-class ``prepare_data`` and pipeline-v2 chunking of two tomograms
with two particle classes, the first validation and training batch of ``Data``, a tiny
conditioned v2 model with deterministic weights, and ``predict()`` with a tiny v2 checkpoint
(real network: per-class maps and STAR coordinates; Gate 4 oracle: coordinates).

Dense arrays are compared through a ``[::4, ::4, ::4]`` subsample (atol 1e-6) plus float64
sum, sum of squares and a seeded random projection of the full array (rtol 1e-6).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from utils.tiny import applied_main_config

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
ATOL = 1e-6


def _common():
    spec = importlib.util.spec_from_file_location("_gate6_common", GOLDEN / "gate6_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN / "gate6_c1.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads((GOLDEN / "gate6_c1.json").read_text())
    return arrays, meta


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    g6 = _common()
    root = tmp_path_factory.mktemp("g6_c1")
    with applied_main_config(None):
        chunks, _, _ = g6.prepare_and_chunk(root / "c")
        arrays, info = g6.chunk_arrays(chunks)
        arrays.update(g6.batch_arrays(chunks))
        arrays.update(g6.forward_arrays(root / "f"))
        arrays.update(g6.predict_arrays(root / "p"))
    return arrays, info


def _compare(golden_arrays, cur, prefix):
    keys = sorted(k for k in golden_arrays if k.startswith(prefix))
    assert keys, f"golden has no '{prefix}' arrays"
    missing = [k for k in keys if k not in cur]
    assert not missing, f"missing arrays: {missing[:10]}"
    extra = sorted(k for k in cur if k.startswith(prefix) and k not in golden_arrays)
    assert not extra, f"arrays not in the golden: {extra[:10]}"
    for k in keys:
        g, c = golden_arrays[k], np.asarray(cur[k])
        if g.dtype.kind in "US":
            assert str(c) == str(g), k
            continue
        assert c.shape == g.shape, (k, c.shape, g.shape)
        if k.endswith(("__sum", "__sumsq", "__proj")):
            np.testing.assert_allclose(c.astype(np.float64), g.astype(np.float64), rtol=1e-6, atol=1e-4, err_msg=k)
        elif g.dtype.kind == "b":
            assert np.array_equal(c, g), k
        else:
            np.testing.assert_allclose(c.astype(np.float64), g.astype(np.float64), rtol=0, atol=ATOL, err_msg=k)


def _subset(g, c, where="layout"):
    """Every key of the golden manifest is present with the same value (new keys are allowed)."""
    if isinstance(g, dict):
        assert isinstance(c, dict), where
        for k, v in g.items():
            assert k in c, f"{where}.{k} missing"
            _subset(v, c[k], f"{where}.{k}")
    elif isinstance(g, list):
        assert isinstance(c, list) and len(c) == len(g), where
        for i, (a, b) in enumerate(zip(g, c)):
            _subset(a, b, f"{where}[{i}]")
    elif isinstance(g, float):
        assert c == pytest.approx(g, abs=1e-9), where
    else:
        assert c == g, f"{where}: {c!r} != {g!r}"


def test_G6_C1_chunks_equal_gate4(golden, current):
    garr, meta = golden
    carr, info = current
    assert info["files"] == meta["info"]["files"], "chunk points/volume files differ"
    _subset(meta["info"]["layout"], info["layout"])
    _compare(garr, carr, "pts:")
    _compare(garr, carr, "vol:")


def test_G6_C1_batches_equal_gate4(golden, current):
    _compare(golden[0], current[0], "batch:val:")
    _compare(golden[0], current[0], "batch:train:")


def test_G6_C1_model_outputs_equal_gate4(golden, current):
    _compare(golden[0], current[0], "forward:")


def test_G6_C1_predictions_equal_gate4(golden, current):
    _compare(golden[0], current[0], "pred:")
    assert len(golden[0]["pred:oracle:small:coords"]) >= 3 and len(golden[0]["pred:oracle:big:coords"]) >= 1
