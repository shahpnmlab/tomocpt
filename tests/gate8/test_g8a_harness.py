"""Gate 8a: profiling harness (spec 08_AGENT_PERFORMANCE.md, 8a pass criteria).

G8a-C1: step micro-benchmark JSON keys, deterministic loss sequence on CPU, CLI output.
G8a-C2: golden comparison (self = zero, perturbed copy = exact max difference).
G8a-C3 is an orchestrator check (git diff), not a test.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.tiny import tiny_main_config  # noqa: E402

STEP_KEYS = ("steps_per_s", "ms_forward", "ms_loss", "ms_backward", "ms_optimizer", "ms_data_wait",
             "peak_mem_bytes", "commit", "config_hash", "losses")


def _run(tmp_path, **kw):
    from benchmarks.perf.step import run_step_benchmark

    cfg = tiny_main_config(tmp_path)
    return run_step_benchmark(cfg, "synthetic", classes=2, batch=2, steps=4, warmup=1, seed=0,
                              device="cpu", **kw)


def test_G8a_C1_step_json_keys_and_determinism(tmp_path):
    a = _run(tmp_path / "a")
    for k in STEP_KEYS:
        assert k in a, f"missing key {k}"
    assert len(a["losses"]) == 4
    assert all(np.isfinite(v) for v in a["losses"])
    assert a["steps_per_s"] > 0
    for k in ("ms_forward", "ms_loss", "ms_backward", "ms_optimizer", "ms_data_wait"):
        assert a[k] >= 0
    json.dumps(a)  # JSON-serialisable
    b = _run(tmp_path / "b")
    assert a["losses"] == b["losses"], "same seed, CPU fp32: the loss sequence must repeat exactly"
    assert a["config_hash"] == b["config_hash"]


def test_G8a_C1_step_cli_writes_json(tmp_path):
    yaml = tmp_path / "tiny.yaml"
    yaml.write_text(
        "train:\n"
        "  batch_size: 2\n  n_cpus_for_dataloading: 0\n  use_gpus: false\n"
        f"  model_dir: {tmp_path / 'model'}\n  chunks_dir: {tmp_path / 'chunks'}\n"
        "  network:\n    TORCH_FLOAT_PRECISION: '32'\n"
        "    SwinUNETR:\n      feature_size: 12\n      use_checkpoint: false\n"
        "      drop_rate: 0.0\n      attn_drop_rate: 0.0\n      dropout_path_rate: 0.0\n"
    )
    out = tmp_path / "step.json"
    cmd = [sys.executable, "-m", "benchmarks.perf.step", "--config", str(yaml), "--data", "synthetic",
           "--classes", "2", "--batch", "2", "--steps", "2", "--warmup", "1", "--seed", "0",
           "--device", "cpu", "--out", str(out)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads(out.read_text())
    for k in STEP_KEYS:
        assert k in d
    assert len(d["losses"]) == 2


def _goldens():
    g = REPO / "tests" / "golden"
    return g / "gate7_lock.json", g / "gate1_forward.npz"


def test_G8a_C2_compare_self_is_zero():
    from benchmarks.perf.compare_goldens import compare

    for p in _goldens():
        r = compare(p, p)
        assert r["max_abs"] == 0.0 and r["max_rel"] == 0.0, p
        assert r["missing"] == [] and r["added"] == [] and r["mismatched"] == [], p
        assert len(r["keys"]) > 0
        assert all(v["max_abs"] == 0.0 for v in r["keys"].values())


def test_G8a_C2_compare_perturbed_json(tmp_path):
    from benchmarks.perf.compare_goldens import compare

    src, _ = _goldens()
    d = json.loads(src.read_text())
    rec = d["records"]["sigmoid_uncond"]
    old = rec["loss"]
    rec["loss"] = old + 0.25
    rec["logits_sha256"] = "0" * 64
    new = tmp_path / "lock.json"
    new.write_text(json.dumps(d))
    r = compare(src, new)
    key = "records/sigmoid_uncond/loss"
    assert r["keys"][key]["max_abs"] == pytest.approx(0.25, abs=1e-12)
    assert r["max_abs"] == pytest.approx(0.25, abs=1e-12)
    assert "records/sigmoid_uncond/logits_sha256" in r["mismatched"]


def test_G8a_C2_compare_perturbed_npz(tmp_path):
    from benchmarks.perf.compare_goldens import compare

    _, src = _goldens()
    z = dict(np.load(src))
    name = next(k for k, v in z.items() if np.issubdtype(np.asarray(v).dtype, np.floating) and np.asarray(v).size > 1)
    arr = np.array(z[name], dtype=z[name].dtype, copy=True)
    flat = arr.reshape(-1)
    flat[1] = flat[1] + np.asarray(0.5, dtype=arr.dtype)
    z[name] = arr
    extra = "g8_extra_key"
    z[extra] = np.zeros(3)
    dropped = next(k for k in z if k not in (name, extra))
    del z[dropped]
    new = tmp_path / "golden.npz"
    np.savez(new, **z)
    r = compare(src, new)
    expected = float(abs(np.float64(flat[1]) - np.float64(np.load(src)[name].reshape(-1)[1])))
    assert r["keys"][name]["max_abs"] == pytest.approx(expected, rel=1e-12)
    assert r["max_abs"] >= expected
    assert dropped in r["missing"] and extra in r["added"]
