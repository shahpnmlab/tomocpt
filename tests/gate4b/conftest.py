"""Gate 4b fixtures.

* ``tests/`` on ``sys.path`` (helpers are imported as ``utils.<module>`` / ``gate4.<module>``).
* Autouse: seed everything and restore the global ``mainConfig`` / ``tomocpt.constants``.
* Tiny pipeline-v2 checkpoints written by the code under test (Gate 3 helper, one-batch CPU
  fits), built once per session on first use.
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

from utils.tiny import applied_main_config  # noqa: E402

from gate4 import g4_helpers as H4  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: GPU or learning tests (run with -m slow)")


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


@pytest.fixture(autouse=True)
def _seed_and_restore_globals():
    _seed()
    with applied_main_config(None):
        yield


#: Two classes, scale-conditioned (the Gate 4 ``two_level_cond`` checkpoint).
COND_CLASSES = ((1, "s100", 100.0), (2, "l500", 500.0))
#: Three classes, unconditioned (for EnsembleModel means, sigmoid and softmax heads).
ABC_CLASSES = ((1, "A", 150.0), (2, "B", 300.0), (3, "C", 200.0))


@pytest.fixture(scope="session")
def cond_ckpt(tmp_path_factory):
    root = tmp_path_factory.mktemp("g4b_cond")
    _seed(0)
    with applied_main_config(None):
        return H4.build_v2_ckpt(root, COND_CLASSES, cond=True)


@pytest.fixture(scope="session")
def abc_ckpts(tmp_path_factory):
    """Two tiny sigmoid checkpoints with different weights (different init seeds)."""
    out = []
    for s in (0, 1):
        root = tmp_path_factory.mktemp(f"g4b_abc{s}")
        _seed(s)
        with applied_main_config(None):
            out.append(H4.build_v2_ckpt(root, ABC_CLASSES, cond=False))
    return out
