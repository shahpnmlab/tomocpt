"""Gate 8c.1, G8c-C8 (spec 3.6): on gloo, two ranks draw different crop streams (each the
``draw_crops`` stream of its rank and epoch) and a two-epoch on-the-fly fit finishes with one
checkpoint. CPU only; slow marker (two subprocess fits)."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gate8 import g8c_helpers as H

WORKER = Path(__file__).with_name("ddp_gloo_otf_worker.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_G8c_C8_gloo_two_ranks_disjoint_streams_and_fit(tmp_path):
    from tomocpt.dataManager.otf import draw_crops

    store = H.build(H.write_inputs(tmp_path / "in"), tmp_path / "store", storage="tomogram")
    out = tmp_path / "out"
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()), PYTHONWARNINGS="ignore")
    env.pop("LOCAL_RANK", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    r = subprocess.run([sys.executable, str(WORKER), str(store), str(out), "2"], env=env, capture_output=True,
                       text=True, timeout=900, cwd=str(WORKER.parents[2]))
    assert r.returncode == 0, r.stderr[-4000:]
    assert (out / "fit_done.txt").read_text().split() == ["2", "4"]  # 8 crops / 2 ranks / batch 2, 2 epochs
    assert len(list((out / "ckpt").glob("*.ckpt"))) == 1
    crops = [json.loads((out / f"crops_rank{k}.json").read_text()) for k in (0, 1)]
    data = H.otf_data(store, samples_per_epoch=8, val_samples=4)
    for e in ("0", "1"):
        a, b = crops[0][e], crops[1][e]
        assert len(a) == len(b) == 4 and a != b
        for rank, got in ((0, a), (1, b)):
            exp = draw_crops(data.train_index, data.train_weights, 8, 42, int(e), rank, 2, jitter=True, stride=32,
                             shapes=[s.shape for s in data.stores_train], size=64)
            np.testing.assert_array_equal(np.array(got), exp)
    assert crops[0]["0"] != crops[0]["1"]
