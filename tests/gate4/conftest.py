"""Gate 4 fixtures.

* ``tests/`` on ``sys.path`` (helpers are imported as ``utils.<module>``, never ``tests.utils``).
* Autouse: seed everything and restore the global ``mainConfig`` / ``tomocpt.constants``
  after each test (checkpoint loading and ``predict`` mutate them).
* ``legacy_ckpt``: tiny legacy checkpoint built with ``v2-gate-0-base`` code.
* ``v2_ckpts``: tiny pipeline-v2 checkpoints written by the code under test (Gate 3 helper,
  one-batch CPU fits), built once per session on first use.
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

from utils.legacy import legacy_checkpoint  # noqa: E402
from utils.tiny import applied_main_config  # noqa: E402

from gate4 import g4_helpers as H  # noqa: E402


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


@pytest.fixture(scope="session")
def legacy_ckpt():
    return legacy_checkpoint()


#: name -> (classes ((number, name, diameter_ang), ...), scale_conditioning)
V2_SPECS = {
    # G4-C4 (v2), C6, C7, C11: one level at 5 A/px (apix_work = 350 / 32 = 10.9375).
    "small_big": (((3, "small", 120.0), (7, "big", 350.0)), False),
    # G4-C3: two levels at 10 A/px (100 A at 10 A/px, 500 A at 15.625 A/px).
    "two_level_cond": (((1, "s100", 100.0), (2, "l500", 500.0)), True),
    # G4-C2 / infer.classes: at 5 A/px, level 1 = {c60} at 5 A/px, level 2 = {c300, c500} at 15.625.
    "three_cond": (((1, "c60", 60.0), (2, "c300", 300.0), (3, "c500", 500.0)), True),
}


@pytest.fixture(scope="session")
def v2_ckpts(tmp_path_factory):
    """``get(name) -> dict(path, x, logits, classes)`` (see :data:`V2_SPECS`)."""
    root = tmp_path_factory.mktemp("g4_v2_ckpts")
    cache = {}

    def get(name):
        if name not in cache:
            classes, cond = V2_SPECS[name]
            _seed()
            with applied_main_config(None):
                cache[name] = H.build_v2_ckpt(root / name, classes, cond=cond)
        return cache[name]

    return get
