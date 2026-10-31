"""G5b-C1: SSL v1 unchanged under the Gate 5b defaults.

The Gate 5 C1 golden (``tests/gate5/g5_c1_golden.json``, from ``v2-gate-0-base``) is re-checked
by calling the frozen Gate 5 test function (imported inside the test, so pytest does not collect it
twice). The rest of the earlier suite is covered by running ``pytest tests -m "not slow"``.
"""

from gate5 import g5_helpers as H
from utils.tiny import applied_main_config, tiny_main_config


def test_G5b_C1_ssl_v1_step_matches_base_golden(tmp_path):
    from gate5 import test_g5_v1_lock as lock

    lock.test_G5_C1_ssl_v1_step_matches_base_golden(tmp_path)


def test_G5b_C1_ssl_v1_model_has_no_gate5b_modules(tmp_path):
    """A default (``version = 1``) SSL model has no queue buffer and no SSL v2 head."""
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    g5 = H.load_common()
    cfg = g5.configure_tiny_ssl(tiny_main_config(tmp_path))
    with applied_main_config(cfg):
        model = SelfSupervisedModel(lr=1e-3, train_continue=None)
    assert H.resolved(model.hparams.config.train.ssl).version == 1
    assert not [n for n, _ in model.named_buffers() if "queue" in n.lower()]
    heads = H.head_classes()
    assert not any(isinstance(m, heads) for m in model.modules())
    assert model.rotation_head is not None
