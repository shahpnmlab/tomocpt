"""G7-C1 (defaults: no head, config defaults), G7-C7 second half (both options off: outputs and
loss bit-identical to Gate 6), and the ``forward_features`` contract (invariant 7).

Golden: ``tests/golden/gate7_lock.json``, written by ``make_goldens_gate7.py`` with the code at
``v2-gate-6-passed`` (commit in the json). Logits are compared by a SHA-256 of their bytes and the
first-step loss by its float hex (bit-exact); the gradient sum with rtol 1e-5 (CPU backward is not
bit-reproducible between runs).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def _common():
    if str(GOLDEN) not in sys.path:
        sys.path.insert(0, str(GOLDEN))
    spec = importlib.util.spec_from_file_location("_gate7_common", GOLDEN / "gate7_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def golden():
    return json.loads((GOLDEN / "gate7_lock.json").read_text())["records"]


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    g7 = _common()
    root = tmp_path_factory.mktemp("g7_lock")
    return {case: g7.lock_record(root / case, case) for case in g7.CASES}


# --------------------------------------------------------------------------------------
# G7-C1
# --------------------------------------------------------------------------------------

def test_G7_C1_config_defaults():
    from tomocpt.defaultConfigs.discriminative_config import DiscriminativeConfig
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.defaultConfigs.train_config import TrainConfig

    d = DiscriminativeConfig()
    expected = dict(contrastive_weight=0.0, embed_dim=64, temperature=0.1, anchors_per_class=32,
                    anchor_min_target=0.5,  # human decision 1 Nov 2026 (spec said 0.99)
                    bank_size=1024, hard_negatives=64, hard_negative_min_score=0.3,
                    hard_negative_max_score=0.9, exclusion_radius=1.0, mask_negatives=64,
                    init_from_ssl=True, value_loss="huber", focal_alpha=2.0, focal_beta=4.0)
    for k, v in expected.items():
        got = getattr(d, k)
        assert str(getattr(got, "value", got)) == str(v), (k, got, v)
    assert isinstance(TrainConfig().discriminative, DiscriminativeConfig)
    assert PrepdataConfig().negative_masks is None


@pytest.mark.parametrize("case", ["sigmoid_uncond", "softmax_cond"])
def test_G7_C1_defaults_create_no_head_and_no_new_state(current, golden, case):
    from tomocpt.networks.ssl.heads import DenseProjectionHead

    model = current[case]["_model"]
    assert getattr(model, "contrastive_head", None) is None
    heads = [n for n, m in model.named_modules() if isinstance(m, DenseProjectionHead)]
    assert not heads, f"projection head modules with the defaults: {heads}"
    assert current[case]["state_dict_keys"] == golden[case]["state_dict_keys"]


# --------------------------------------------------------------------------------------
# G7-C7 (both options off)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["sigmoid_uncond", "softmax_cond"])
def test_G7_C7_both_off_is_bit_identical_to_gate6(current, golden, case):
    cur, gold = current[case], golden[case]
    assert cur["logits_sha256"] == gold["logits_sha256"], (cur["logits_sum"], gold["logits_sum"])
    assert cur["loss_hex"] == gold["loss_hex"], (cur["loss"], gold["loss"])
    assert cur["grads_abs_sum"] == pytest.approx(gold["grads_abs_sum"], rel=1e-5)


@pytest.mark.parametrize("case", ["sigmoid_uncond", "softmax_cond"])
def test_G7_C7_explicit_off_values_equal_defaults(tmp_path, golden, case):
    """``contrastive_weight = 0`` and ``value_loss = "huber"`` set explicitly (with other
    discriminative fields changed) are the defaults."""
    g7 = _common()
    cfg = g7.lock_config(tmp_path / "cfg", case)
    d = cfg.train.discriminative
    d.contrastive_weight = 0.0
    d.value_loss = "huber"
    d.embed_dim = 32
    d.hard_negatives = 7
    rec = g7.lock_record(tmp_path / case, case, cfg=cfg)
    assert rec["loss_hex"] == golden[case]["loss_hex"], (rec["loss"], golden[case]["loss"])
    assert rec["logits_sha256"] == golden[case]["logits_sha256"]
    assert rec["state_dict_keys"] == golden[case]["state_dict_keys"]


# --------------------------------------------------------------------------------------
# forward_features (invariant 7)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("cond", [False, True])
def test_G7_forward_features_matches_forward(cond):
    from utils.tiny import tiny_swin_kwargs

    from tomocpt.networks.swinunetr import MySwinUNETR, ScaleConditionedSwinUNETR

    kw = tiny_swin_kwargs()
    kw["out_channels"] = 3
    torch.manual_seed(0)
    net = (ScaleConditionedSwinUNETR if cond else MySwinUNETR)(**kw).eval()
    if cond:
        with torch.no_grad():  # make FiLM non-trivial so the conditioned path is exercised
            for p in net.cond.parameters():
                p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(2, 1, 64, 64, 64)
    apix_values = (None, torch.tensor([[10.0], [20.0]])) if cond else (None,)
    for apix in apix_values:
        with torch.no_grad():
            ref = net(x) if apix is None else net(x, apix=apix)
            out = net.forward_features(x) if apix is None else net.forward_features(x, apix=apix)
        assert isinstance(ref, tuple) and len(ref) == 2
        assert len(out) == 3
        logits, hidden4, dec = out
        assert torch.equal(logits, ref[0]) and torch.equal(hidden4, ref[1])
        assert tuple(dec.shape) == (2, kw["feature_size"], 32, 32, 32)
    if cond:
        with torch.no_grad():
            a = net.forward_features(x, apix=torch.tensor([[10.0], [10.0]]))[2]
            b = net.forward_features(x, apix=torch.tensor([[20.0], [20.0]]))[2]
        assert not torch.allclose(a, b), "decoder2 features must be FiLM-conditioned"
