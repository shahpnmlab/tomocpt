"""Gate 5b fixtures.

* ``tests/`` on ``sys.path`` (helpers are imported as ``utils.<module>`` and ``gate5.<module>``,
  never ``tests.<...>``).
* Autouse: seed everything and restore the global ``mainConfig`` / ``tomocpt.constants``
  after each test (model constructors and checkpoint loading mutate them).
* ``g5b_fit``: short SSL v2 fits (tiny model, CPU, fp32) with gradient and projection capture,
  built once per session (``g5b_helpers.fit_ssl_v2``).
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

from gate5b import g5b_helpers as HB  # noqa: E402


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


#: name -> (number of training steps, SSL settings on top of ``SSLConfig(version=2)``)
FITS = {
    # Every SSL v2 default (vicreg, queue, dense_grad_scale = 1, recon_weight = 1); a queue of 4
    # so that it wraps around within 3 steps of batch 2 (FIFO test).
    "default_q4": (3, dict(n_anchors=32, queue_size=4)),
    # Only the dense InfoNCE task reaches the network (no reconstruction, global or scale loss,
    # no VICReg): full dense gradient vs head-only dense gradient.
    "dense_only_full": (1, dict(n_anchors=32, recon_weight=0.0, contrast_weight=0.0, scale_weight=0.0,
                                collapse_reg="none", dense_grad_scale=1.0)),
    "dense_only_head": (1, dict(n_anchors=32, recon_weight=0.0, contrast_weight=0.0, scale_weight=0.0,
                                collapse_reg="none", dense_grad_scale=0.0)),
}


@pytest.fixture(scope="session")
def g5b_fit(tmp_path_factory):
    """``get(name) -> dict`` (see ``g5b_helpers.fit_ssl_v2``), built on first request."""
    root = tmp_path_factory.mktemp("g5b_ssl_v2")
    cache = {}

    def get(name):
        if name not in cache:
            steps, kw = FITS[name]
            _seed()
            with applied_main_config(None):
                cache[name] = HB.fit_ssl_v2(root / name, n_steps=steps, **kw)
        return cache[name]

    return get


#: Frozen Gate 5 tests that pin v2 behaviour superseded by Gate 5b (human ruling, 31 Oct 2026).
#: Each is replaced by a Gate 5b test (see README.md). Deselected whenever tests/gate5b is collected
#: (e.g. ``pytest tests``); running ``pytest tests/gate5`` alone still runs them.
SUPERSEDED_GATE5 = (
    # corr(input, clean target) > 0.5 -> test_G5b_C1_superseded_G5_C11_polarity (r > 0.1)
    "gate5/test_g5_views.py::test_G5_C11_inversion_applies_to_view_and_target",
)


def pytest_collection_modifyitems(config, items):
    keep, drop = [], []
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        (drop if any(nodeid.endswith(s) for s in SUPERSEDED_GATE5) else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
