"""Gate 3b fixtures: ``tests/`` on ``sys.path`` (helpers are ``utils.<module>`` and
``gate3.g3_helpers``), seeds, and the global ``mainConfig`` restored after each test."""

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


@pytest.fixture(autouse=True)
def _seed_and_restore_globals():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    with applied_main_config(None):
        yield
