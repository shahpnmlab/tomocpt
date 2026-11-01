"""G7-C7 (both options on: every trainable parameter gets a gradient and is optimised) and G7-C8
(SSL v2 dense head -> picker contrastive head).

C7 batch: ``gate7_common.lock_items`` (particle + tube targets, class table 1 "ribo" particle
120 A, 5 "mt" filament 250 A) plus a ``negative_mask [1, 64, 64, 64]`` slab per sample, so the
anchor, hard-negative and mask-negative paths all run. The first loss must differ from the Gate 6
loss (the options are wired in) and be finite.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from gate3 import g3_helpers as G3
from gate5 import g5_helpers as H5
from gate7 import g7_helpers as H
from utils.tiny import applied_main_config

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def _common():
    if str(GOLDEN) not in sys.path:
        sys.path.insert(0, str(GOLDEN))
    spec = importlib.util.spec_from_file_location("_gate7_common_m", GOLDEN / "gate7_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _items_with_mask(g7):
    items = g7.lock_items()
    for b, it in enumerate(items):
        m = torch.zeros(1, 64, 64, 64, dtype=torch.bool)
        m[0, 50 + b:54 + b] = True
        it["negative_mask"] = m
    return items


@pytest.mark.parametrize("case", ["sigmoid_uncond", "softmax_cond"])
@pytest.mark.parametrize("options", ["both", "contrastive", "focal"])
def test_G7_C7_options_on_every_parameter_gets_a_gradient(tmp_path, case, options):
    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.networks.ssl.heads import DenseProjectionHead

    g7 = _common()
    gold = json.loads((GOLDEN / "gate7_lock.json").read_text())["records"][case]
    head, cond = g7.CASES[case]
    disc = {}
    if options in ("both", "contrastive"):
        disc.update(contrastive_weight=0.1)
    if options in ("both", "focal"):
        disc.update(value_loss="focal")
    cfg = H.disc_config(tmp_path, g7.PARTICLES, g7.FILAMENTS, class_head=head, cond=cond, **disc)
    with applied_main_config(cfg):
        torch.manual_seed(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        if "contrastive_weight" in disc:
            assert isinstance(model.contrastive_head, DenseProjectionHead)
        else:
            assert getattr(model, "contrastive_head", None) is None
        loss, grads, trainer = G3.first_step_loss_and_grads(model, _items_with_mask(g7), tmp_path / "tb")
    assert torch.isfinite(torch.tensor(loss)), loss
    assert loss != gold["loss"], "the first loss equals Gate 6: the option is not wired into training_step"
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, f"parameters without a gradient: {missing[:10]} ({len(missing)} in all)"
    not_opt = H.trainable_not_in_optimizer(model, trainer)
    assert not not_opt, f"trainable parameters not in the optimizer: {not_opt[:10]}"
    if "contrastive_weight" in disc:
        head_grads = [g for n, g in grads.items() if n.startswith("contrastive_head.")]
        assert head_grads and any(torch.count_nonzero(g) > 0 for g in head_grads), \
            "the batch has anchors: the contrastive head must get a non-zero gradient"


def test_G7_C7_batch_without_anchors_still_gives_the_head_a_gradient(tmp_path):
    """No target anywhere (no anchors): the contrastive term is 0 but the head is in the graph (DDP)."""
    from tomocpt.networks.pickingModel import BasePickingModel

    g7 = _common()
    items = g7.lock_items()
    for it in items:
        it["target_data"] = torch.zeros_like(it["target_data"])
    cfg = H.disc_config(tmp_path, g7.PARTICLES, g7.FILAMENTS, class_head="softmax", contrastive_weight=0.1)
    with applied_main_config(cfg):
        torch.manual_seed(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        loss, grads, _ = G3.first_step_loss_and_grads(model, items, tmp_path / "tb")
    assert torch.isfinite(torch.tensor(loss))
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, missing[:10]


# --------------------------------------------------------------------------------------
# G7-C8
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ssl_fit(tmp_path_factory):
    root = tmp_path_factory.mktemp("g7_ssl")
    with applied_main_config(None):
        return H5.fit_ssl_v2_one_step(root)


@pytest.mark.parametrize("init_from_ssl", [True, False])
def test_G7_C8_ssl_dense_head_initialises_the_contrastive_head(tmp_path, ssl_fit, init_from_ssl):
    from tomocpt.training.helpers import _create_model

    ssl_sd = torch.load(ssl_fit["path"], map_location="cpu", weights_only=False)["state_dict"]
    src = {k[len("dense_head."):]: v for k, v in ssl_sd.items() if k.startswith("dense_head.")}
    assert src, "the SSL v2 checkpoint has a dense head (train.ssl.dense_weight > 0 by default)"
    embed_dim = int(src["proj.weight"].shape[0])

    cfg = G3.v2_config(tmp_path, ((1, "A", 150.0), (2, "B", 300.0)), cond=True)
    cfg.train.particles.class_head = "softmax"
    cfg.train.discriminative.contrastive_weight = 0.1
    cfg.train.discriminative.embed_dim = embed_dim
    cfg.train.discriminative.init_from_ssl = init_from_ssl
    with applied_main_config(cfg) as g:
        g.train.resume_from = None
        g.train.fine_tune_from = ssl_fit["path"]
        g.train.use_distillation = False
        torch.manual_seed(0)
        model, resume, _ = _create_model(g.train)
    assert resume is None
    head_sd = model.contrastive_head.state_dict()
    assert set(head_sd) == set(src), (sorted(head_sd), sorted(src))
    same = all(torch.equal(head_sd[k], src[k]) for k in src)
    assert same is init_from_ssl, f"init_from_ssl={init_from_ssl}: head equal to the SSL dense head = {same}"
