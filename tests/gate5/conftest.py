"""Gate 5 fixtures.

* ``tests/`` on ``sys.path`` (helpers are imported as ``utils.<module>``, never ``tests.utils``).
* Autouse: seed everything and restore the global ``mainConfig`` / ``tomocpt.constants``
  after each test (model constructors and checkpoint loading mutate them).
* ``ssl_v2_fit``: one SSL v2 training step (tiny model, CPU) through a Lightning trainer,
  with the gradients after backward and a saved checkpoint; built once per session.
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

from gate5 import g5_helpers as H  # noqa: E402


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
def ssl_v2_fit(tmp_path_factory):
    """``get(name) -> dict`` (see ``g5_helpers.fit_ssl_v2_one_step``), built on first request.

    ``"default"``: ``train.ssl.version = 2`` and nothing else changed (``n_anchors = 32`` for speed).
    ``"recon_only"``: the same with ``contrast_weight = dense_weight = scale_weight = 0``.
    """
    root = tmp_path_factory.mktemp("g5_ssl_v2")
    cache = {}
    kws = {
        "default": dict(n_anchors=32),
        "recon_only": dict(n_anchors=32, contrast_weight=0.0, dense_weight=0.0, scale_weight=0.0),
    }

    def get(name):
        if name not in cache:
            _seed()
            with applied_main_config(None):
                cache[name] = H.fit_ssl_v2_one_step(root / name, **kws[name])
        return cache[name]

    return get
