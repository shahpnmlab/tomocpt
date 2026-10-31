"""Gate 6 fixtures.

* ``tests/`` on ``sys.path`` (helpers are imported as ``utils.<module>``, never ``tests.utils``).
* Autouse: seed everything and restore the global ``mainConfig`` / ``tomocpt.constants``
  after each test (model constructors, checkpoint loading and ``predict`` mutate them).
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
