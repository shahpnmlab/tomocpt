"""G4-C9 (slow): ``benchmarks/synthetic_multisize.py --tiny`` runs every arm and writes
``report.json`` with the keys of the T4.6 contract (exact key names: ``tests/gate4/README.md``).

The script is run as a subprocess from the repository root with ``--tiny`` only (the spec
gives no output option); the test takes the ``benchmarks/out/*/report.json`` written after
it started (``benchmarks/out/`` is git-ignored).
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ARMS = ("a", "b", "c", "d")
CALIBRATED_ARMS = ("b", "c", "d")
SOURCES = {"class_thresholds", "calibration", "confidence_threshold"}
STAT_KEYS = ("fbeta", "precision", "recall")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_class_entry(e, where, mode):
    for k in STAT_KEYS:
        assert k in e, f"{where}: missing {k!r}"
        assert set(e[k]) >= {"mean", "sd"}, f"{where}.{k}: needs 'mean' and 'sd'"
        assert _num(e[k]["mean"]) and 0.0 <= e[k]["mean"] <= 1.0, f"{where}.{k}.mean"
        assert _num(e[k]["sd"]) and e[k]["sd"] >= 0.0, f"{where}.{k}.sd"
    assert "threshold" in e and "threshold_source" in e, f"{where}: threshold and its source"
    assert e["threshold_source"] in SOURCES, f"{where}: threshold_source {e['threshold_source']!r}"
    assert _num(e["threshold"]), f"{where}: threshold {e['threshold']!r}"
    if mode == "global_0.3":
        assert e["threshold"] == pytest.approx(0.3) and e["threshold_source"] == "confidence_threshold", where


def check_report(rep):
    assert rep.get("benchmark") == "synthetic_multisize"
    assert rep.get("tiny") is True
    names = [c["name"] for c in rep["classes"]]
    assert len(names) >= 2 and all({"number", "name", "diameter_ang"} <= set(c) for c in rep["classes"])
    assert set(rep["arms"]) >= set(ARMS), f"arms: {sorted(rep['arms'])}"
    for arm in ARMS:
        a = rep["arms"][arm]
        assert _num(a["wall_time_per_tomogram_s"]) and a["wall_time_per_tomogram_s"] >= 0, arm
        g = a["peak_gpu_memory_mb"]
        assert g is None or (_num(g) and g >= 0), arm
        assert isinstance(a["cross_class_suppression"], dict), arm
        assert all(isinstance(k, str) and isinstance(v, int) and v >= 0
                   for k, v in a["cross_class_suppression"].items()), arm
        modes = ("global_0.3", "calibrated") if arm in CALIBRATED_ARMS else ("global_0.3",)
        for mode in modes:
            pc = a["scoring"][mode]["per_class"]
            assert set(pc) == set(names), f"{arm}/{mode}: classes {sorted(pc)}"
            for n in names:
                _check_class_entry(pc[n], f"arms.{arm}.scoring.{mode}.per_class.{n}", mode)


@pytest.mark.slow
def test_G4_C9_synthetic_multisize_tiny_report(tmp_path):
    script = REPO / "benchmarks" / "synthetic_multisize.py"
    assert script.is_file(), f"{script} is missing"
    start = time.time() - 1.0
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run([sys.executable, str(script), "--tiny"], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=1500)
    assert res.returncode == 0, f"exit {res.returncode}\n{res.stdout[-3000:]}\n{res.stderr[-6000:]}"
    reports = [p for p in (REPO / "benchmarks" / "out").glob("*/report.json") if p.stat().st_mtime >= start]
    assert reports, "no benchmarks/out/<run>/report.json written by this run"
    report = max(reports, key=lambda p: p.stat().st_mtime)
    assert (report.parent / "report.md").is_file(), "report.md next to report.json"
    check_report(json.loads(report.read_text()))
