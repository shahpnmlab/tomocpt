"""Gate 2 fixtures.

Chunking runs are expensive (a few seconds each), so the datasets the tests share are built
once per session. Every chunking run goes through ``gate2_common.inprocess_chunking``: the
Dask cluster runs in-process with one thread, CUDA is hidden, and the chunking module gets a
seeded ``random.Random``. This keeps the fast tier on the CPU and makes runs reproducible.
"""

import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from utils.tiny import applied_main_config  # noqa: E402

from gate2 import g2_helpers as H  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: GPU or learning tests (run with -m slow)")


@pytest.fixture(autouse=True)
def _seed_and_restore_globals():
    """Seed everything; restore the global mainConfig / constants after each test."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    with applied_main_config(None):
        yield


def _seeded():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


# ------------------------------------------------------------------------------------------
# pipeline v1 (G2-C1 golden inputs; also the v1 chunks for C13 / C18)
# ------------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def v1_run(tmp_path_factory):
    """make_dataset(n_tomos=2) -> legacy prepare_data -> do_chunking (defaults), both golden
    variants. Returns ``{"root", "training_data_dir", "<variant>": chunks_dir}``."""
    common = H.load_common()
    root = tmp_path_factory.mktemp("g2_v1")
    with applied_main_config(None):
        _seeded()
        _, tdd = common.prepare_v1_dataset(root)
        out = {"root": root, "training_data_dir": tdd}
        for variant in common.VARIANTS:
            _seeded()
            chunks = root / f"chunks_{variant}"
            common.run_v1_chunking(tdd, chunks, variant)
            out[variant] = chunks
    return out


# ------------------------------------------------------------------------------------------
# pipeline v2 datasets
# ------------------------------------------------------------------------------------------

#: Main v2 dataset: two classes (120 A, 300 A) at 7 A/px. Band rule: apix_work = 300/32 =
#: 9.375 A (one level, resampled, then the Z axis is padded 47 -> 64). d_vox = 12.8 and 32.0.
MAIN_APIX = 7.0
MAIN_SHAPE = (64, 128, 128)
MAIN_DIAMETERS = {1: 120.0, 2: 300.0}
MAIN_NAMES = {1: "small", 2: "large"}
MAIN_SPECIES = ((1, 120.0, 8, False), (2, 300.0, 3, False))


@pytest.fixture(scope="session")
def v2main_inputs(tmp_path_factory):
    from tomocpt.labels.multiclass import build_class_table

    root = tmp_path_factory.mktemp("g2_v2main")
    tomos = H.phantom_tomos(["TS_01", "TS_02"], MAIN_SHAPE, MAIN_APIX, MAIN_SPECIES, seed0=10)
    df = H.write_v2_inputs(root, tomos, MAIN_DIAMETERS)
    table = build_class_table(MAIN_DIAMETERS, MAIN_NAMES)
    return dict(root=root, tomos=tomos, df=df, table=table, apix=MAIN_APIX, shape=MAIN_SHAPE,
                diameters=MAIN_DIAMETERS)


@pytest.fixture(scope="session")
def v2main(v2main_inputs):
    """v2main inputs chunked with target_storage='points' (defaults otherwise)."""
    with applied_main_config(None):
        _seeded()
        chunks = H.run_v2_chunking(v2main_inputs["df"], v2main_inputs["root"] / "chunks_points",
                                   v2main_inputs["table"])
    return dict(v2main_inputs, chunks=chunks)


@pytest.fixture(scope="session")
def v2main_dense(v2main_inputs):
    """The same inputs chunked with target_storage='dense'."""
    with applied_main_config(None):
        _seeded()
        chunks = H.run_v2_chunking(v2main_inputs["df"], v2main_inputs["root"] / "chunks_dense",
                                   v2main_inputs["table"], target_storage="dense")
    return dict(v2main_inputs, chunks=chunks)


@pytest.fixture
def chunks_copy(tmp_path):
    """Copy a chunks directory into ``tmp_path / 'chunks'`` (tests that train write into it)."""
    def _copy(src):
        dst = tmp_path / "chunks"
        shutil.copytree(src, dst)
        return dst
    return _copy
