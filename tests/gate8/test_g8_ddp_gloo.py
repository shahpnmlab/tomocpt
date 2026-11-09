"""Gate 8, DDP check on gloo (spec 3.6, G8b-C7): 2 CPU processes, tiny v2 model, no dropout, every
``(b, c)`` available. After one step the all-reduced gradients equal the single-process gradients on
the concatenated batch (fp32, ``rtol 1e-5``); a two-epoch fit finishes and saves one checkpoint.
Run at the defaults and with every CPU-capable 8b option on. NCCL refuses two ranks on one GPU, so
GPU DDP is not measured here.
"""
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

WORKER = Path(__file__).with_name("ddp_gloo_worker.py")
OPTIONS = {
    "defaults": [],
    "8b": ["network.SwinUNETR.use_checkpoint=False", "particles.loss_checkpoint=False",
           "particles.hessian_block=0"],
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(mode, out, devices, options):
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()), PYTHONWARNINGS="ignore")
    env.pop("LOCAL_RANK", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    r = subprocess.run([sys.executable, str(WORKER), mode, str(out), str(devices), *options],
                       env=env, capture_output=True, text=True, timeout=600, cwd=str(WORKER.parents[2]))
    assert r.returncode == 0, r.stderr[-4000:]


@pytest.mark.slow
@pytest.mark.parametrize("opts", list(OPTIONS))
def test_G8_ddp_gloo_gradients_equal_single_process(tmp_path, opts):
    _run("grads", tmp_path, 1, OPTIONS[opts])
    _run("grads", tmp_path, 2, OPTIONS[opts])
    one = torch.load(tmp_path / "grads_1.pt")
    two = torch.load(tmp_path / "grads_2.pt")
    assert one.pop("__none__") == [] and two.pop("__none__") == []
    assert one.keys() == two.keys()
    # rtol 1e-5 per tensor (an elementwise relative bound is meaningless on near-zero elements; the
    # two runs sum the batch in a different order): max |difference| <= 1e-5 max |g|, and the norms.
    for n in one:
        scale = float(one[n].abs().max())
        assert float((two[n] - one[n]).abs().max()) <= 1e-5 * scale + 1e-12, n
        assert abs(float(two[n].norm()) - float(one[n].norm())) <= 1e-5 * float(one[n].norm()) + 1e-12, n


@pytest.mark.slow
@pytest.mark.parametrize("opts", list(OPTIONS))
def test_G8_ddp_gloo_two_epoch_fit_saves_a_checkpoint(tmp_path, opts):
    _run("fit", tmp_path, 2, OPTIONS[opts])
    assert (tmp_path / "fit_done.txt").read_text().split() == ["2", "4"]
    ckpts = list((tmp_path / "ckpt").glob("*.ckpt"))
    assert len(ckpts) == 1, ckpts
