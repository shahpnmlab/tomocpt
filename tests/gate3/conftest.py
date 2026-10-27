"""Gate 3 fixtures.

* ``tests/`` on ``sys.path`` (helpers are imported as ``utils.<module>``, never ``tests.utils``).
* Autouse: seed everything and restore the global ``mainConfig`` / ``tomocpt.constants``
  after each test (model constructors and checkpoint loading mutate them).
* Legacy checkpoints (``utils.legacy``, built with ``v2-gate-0-base`` code).
* ``v2_ckpts``: pipeline-v2 checkpoints written by the code under test (one-batch fits on
  CPU), built once per session on first use.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from utils.legacy import legacy_checkpoint, legacy_ssl_checkpoint  # noqa: E402
from utils.tiny import applied_main_config, tiny_main_config  # noqa: E402

from gate3 import g3_helpers as H  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: GPU or learning tests (run with -m slow)")


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


@pytest.fixture(autouse=True)
def _seed_and_restore_globals():
    _seed()
    with applied_main_config(None):
        yield


@pytest.fixture
def tiny_global(tmp_path):
    """The global mainConfig set to the tiny CPU (v1) config for the duration of a test."""
    with applied_main_config(tiny_main_config(tmp_path)) as cfg:
        yield cfg


@pytest.fixture(scope="session")
def legacy_ckpt():
    return legacy_checkpoint()


@pytest.fixture(scope="session")
def legacy_ssl_ckpt():
    return legacy_ssl_checkpoint()


@pytest.fixture(scope="session")
def v2_ckpts(tmp_path_factory):
    """``get(name) -> dict(path, x, logits, table)``; checkpoints built on first request.

    ``"c2"``: classes ``{1: A 150 A, 2: B 300 A}``, unconditioned.
    ``"c2_cond"``: the same classes, ``scale_conditioning=True``.
    ``logits`` maps an apix (or ``None``) to the eval-mode logits of the saved model on ``x``.
    """
    root = tmp_path_factory.mktemp("g3_v2_ckpts")
    cache = {}

    def get(name):
        if name not in cache:
            cond = {"c2": False, "c2_cond": True}[name]
            _seed()
            cache[name] = H.build_v2_checkpoint(root / name, H.CLASSES_AB, cond=cond)
        return cache[name]

    return get
