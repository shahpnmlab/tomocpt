"""Gate 3c fixtures.

* ``tests/`` on ``sys.path`` (helpers are ``utils.<module>`` and ``gate3.g3_helpers``).
* Autouse: seeds; the global ``mainConfig`` / ``tomocpt.constants`` restored and the CLI
  snapshot cleared after each test, so neither leaks into other tests.
* ``ckpts``: two tiny picking checkpoints with different train settings (built once per session).
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
from utils.tiny import applied_main_config, tiny_main_config  # noqa: E402

from gate3 import g3_helpers as H  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: GPU or learning tests (run with -m slow)")


def _clear_snapshot():
    # Imported here: on code without the snapshot API the tests fail, the cleanup must not.
    try:
        from tomocpt.configManager.configManager import clear_cli_snapshot
    except ImportError:
        return
    clear_cli_snapshot()


@pytest.fixture(autouse=True)
def _seed_restore_and_clear():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    _clear_snapshot()
    try:
        with applied_main_config(None):
            yield
    finally:
        _clear_snapshot()


#: Train settings of the two checkpoints (all differ from each other and from the defaults).
A_TRAIN = dict(n_epochs=3, batch_size=4)
B_TRAIN = dict(n_epochs=9, batch_size=6)


def _save(model, path: Path):
    """A Lightning-format checkpoint without a Trainer (``load_from_checkpoint`` reads only
    ``state_dict``, ``hyper_parameters`` and the version)."""
    import pytorch_lightning as pl

    torch.save({"state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__}, path)


def _build(root: Path, v2: bool, train: dict) -> Path:
    from tomocpt.networks.pickingModel import BasePickingModel

    root.mkdir(parents=True, exist_ok=True)
    cfg = H.v2_config(root, H.CLASSES_AB) if v2 else tiny_main_config(root)
    for k, v in train.items():
        setattr(cfg.train, k, v)
    torch.manual_seed(0)
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        path = root / "model.ckpt"
        _save(model, path)
    return path


@pytest.fixture(scope="session")
def ckpts(tmp_path_factory):
    """``{"A": v1 tiny picker (A_TRAIN), "B": v2 two-class picker (B_TRAIN, classes A/B),
    "legacy": the v2-gate-0-base tiny picker}``."""
    root = tmp_path_factory.mktemp("g3c_ckpts")
    with applied_main_config(None):
        out = {"A": _build(root / "A", v2=False, train=A_TRAIN),
               "B": _build(root / "B", v2=True, train=B_TRAIN),
               "legacy": legacy_checkpoint()}
    assert A_TRAIN == {"n_epochs": 3, "batch_size": 4} and B_TRAIN == {"n_epochs": 9, "batch_size": 6}
    return out
