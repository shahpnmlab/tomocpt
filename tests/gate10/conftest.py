"""Gate 10 fixtures: ``tests/`` on ``sys.path``; restore the global ``mainConfig`` and ``sys.argv``."""

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from utils.tiny import applied_main_config  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_globals(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tomocpt"])
    with applied_main_config(None):
        yield
