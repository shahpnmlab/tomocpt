"""Gate 1 fixtures."""

import importlib.util
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

GOLDEN_DIR = TESTS_DIR / "golden"


def pytest_configure(config):
    # The implementer registers `slow` in pyproject.toml at Gate 1; harmless if both do.
    config.addinivalue_line("markers", "slow: GPU or learning tests (run with -m slow)")


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def golden_common():
    return _load_by_path("_gate1_common", GOLDEN_DIR / "gate1_common.py")


@pytest.fixture(autouse=True)
def _seed_and_restore_globals():
    """Seed everything and restore the global mainConfig / constants after each test
    (legacy model constructors and checkpoint loading mutate them)."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    with applied_main_config(None):
        yield


@pytest.fixture
def tiny_global(tmp_path):
    """The global mainConfig set to the tiny CPU config for the duration of a test."""
    with applied_main_config(tiny_main_config(tmp_path)) as cfg:
        yield cfg


@pytest.fixture(scope="session")
def legacy_ckpt():
    return legacy_checkpoint()


@pytest.fixture(scope="session")
def legacy_ssl_ckpt():
    return legacy_ssl_checkpoint()
