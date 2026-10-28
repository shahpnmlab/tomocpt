"""G5-C1: SSL v1 lock against the ``v2-gate-0-base`` golden (``g5_c1_golden.json``)."""

import json

import numpy as np
import torch

from gate5 import g5_helpers as H
from utils.tiny import applied_main_config, tiny_main_config

GOLDEN = H.HERE / "g5_c1_golden.json"


def test_G5_C1_ssl_v1_step_matches_base_golden(tmp_path):
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    g5 = H.load_common()
    gold = json.loads(GOLDEN.read_text())
    x = g5.ssl_batch_input()
    assert float(x.astype(np.float64).sum()) == gold["x_sum"]

    cfg = g5.configure_tiny_ssl(tiny_main_config(tmp_path))
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        with applied_main_config(cfg):
            torch.manual_seed(0)
            model = SelfSupervisedModel(lr=1e-3, train_continue=None)
            # The v1 model keeps its parameters (names) exactly; the golden weights are keyed by name.
            assert sorted(n for n, _ in model.named_parameters()) == gold["param_names"]
            g5.reinit_by_name_(model)
            sums = g5.ssl_v1_forward_sums(model, x)
            step = g5.ssl_v1_step(model, x)
    finally:
        torch.set_num_threads(threads)
    np.testing.assert_allclose(sums, gold["forward_sums"], rtol=1e-5)
    # total loss, rotation, contrastive, reconstruction
    np.testing.assert_allclose(step, gold["step"], rtol=1e-5)
