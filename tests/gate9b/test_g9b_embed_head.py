"""Gate 9b (spec ``.claude/agents/09b_AGENT_EMBED_HEAD.md``): the class head uses the contrastive
embedding. Option 3: cosine-softmax CE on the embedding at the anchors
(``train.discriminative.embed_ce_weight``). Option 2: prototype residual on the particle logits
(``train.discriminative.prototype_head``). All CPU.

Public names used: ``tomocpt.training.discriminative.{PrototypeClassifier, prototype_ce_loss}``,
``BasePickingModel.{prototype_classifier, _forward_embed, discriminative_terms, embed_ce_weight_at}``.
"""

import pytest
import torch

from gate3 import g3_helpers as G3
from gate6 import g6_helpers as G6
from utils.tiny import applied_main_config

PARTICLES = ((1, "A", 120.0), (2, "B", 200.0))
FILAMENTS = ((9, "actin", 70.0),)
D_VOX = [12.0, 20.0, 7.0]
P = 2


def _cfg(tmp_path, scope="all", **disc):
    cfg = G6.v2_config_with_filaments(tmp_path, PARTICLES, FILAMENTS, class_head="softmax")
    cfg.train.particles.softmax_scope = scope
    for k, v in disc.items():
        setattr(cfg.train.discriminative, k, v)
    return cfg


def _model(seed=0):
    from tomocpt.networks.pickingModel import BasePickingModel

    torch.manual_seed(seed)
    return BasePickingModel(lr=1e-3, train_continue=None)


def _items(n=2, available=None, empty=False):
    items = []
    for b in range(n):
        it = G3.v2_item(500 + b, D_VOX, apix=10.0, available=available, n_per_class=3)
        if empty:
            it["target_data"] = torch.zeros_like(it["target_data"])
        m = torch.zeros(1, 64, 64, 64, dtype=torch.bool)
        m[0, 2 + b:5 + b] = True
        it["negative_mask"] = m
        items.append(it)
    return items


# --------------------------------------------------------------------------------------
# C1 defaults
# --------------------------------------------------------------------------------------

def test_G9b_C1_defaults():
    from tomocpt.defaultConfigs.discriminative_config import DiscriminativeConfig

    d = DiscriminativeConfig()
    assert d.embed_ce_weight == 0.0
    assert d.prototype_head is False
    assert d.prototype_scale == 10.0


@pytest.mark.parametrize("contrastive", [0.0, 0.1])
def test_G9b_C1_no_prototype_module_by_default(tmp_path, contrastive):
    with applied_main_config(_cfg(tmp_path, contrastive_weight=contrastive)):
        model = _model()
    assert model.prototype_classifier is None
    assert not any("prototype" in k for k in model.state_dict())
    assert (model.contrastive_head is None) == (contrastive == 0.0)


def test_G9b_C1_not_cli_flags():
    from tomocpt.configManager.cli_options import CLI_FLAGS

    flat = {str(f).split(".")[-1] for names in CLI_FLAGS.values() for f in names}
    for name in ("embed_ce_weight", "prototype_head", "prototype_scale"):
        assert name not in flat


@pytest.mark.parametrize("field,value", [("embed_ce_weight", 0.1), ("prototype_head", True)])
def test_G9b_C1_creates_the_projection_head_without_supcon(tmp_path, field, value):
    """Either option alone creates the projection head and the prototypes, but no memory bank."""
    with applied_main_config(_cfg(tmp_path, **{field: value})):
        model = _model()
    assert model.contrastive_head is not None
    assert model.prototype_classifier is not None
    assert model.prototype_classifier.prototypes.shape[0] == P + 1
    assert (model.prototype_classifier.alpha is not None) == (field == "prototype_head")
    assert getattr(model, "contrastive_bank", None) is None


def test_G9b_C1_needs_pipeline_v2(tmp_path):
    from utils.tiny import tiny_main_config

    cfg = tiny_main_config(tmp_path)
    cfg.train.discriminative.embed_ce_weight = 0.1
    with applied_main_config(cfg), pytest.raises(ValueError):
        _model()


# --------------------------------------------------------------------------------------
# C2 loss value
# --------------------------------------------------------------------------------------

def test_G9b_C2_prototype_ce_loss_matches_reference():
    from tomocpt.training.discriminative import PrototypeClassifier, prototype_ce_loss

    clf = PrototypeClassifier(2, 3, scale=10.0)
    with torch.no_grad():
        clf.prototypes.copy_(torch.tensor([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]]))
    emb = torch.tensor([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 5.0]])
    rows = torch.tensor([0, 1, 2, 2])  # 2 = background
    allowed = torch.ones(4, 3, dtype=torch.bool)
    allowed[3, 1] = False  # class 1 unavailable for anchor 3
    cos = clf.cosine(emb)
    loss = prototype_ce_loss(cos, rows, allowed, 10.0)

    e = torch.nn.functional.normalize(emb.double(), dim=1)
    p = torch.nn.functional.normalize(clf.prototypes.detach().double(), dim=1)
    ref_cos = e @ p.T
    assert torch.allclose(cos.double(), ref_cos, atol=1e-6)
    total = 0.0
    for i in range(4):
        logits = 10.0 * ref_cos[i]
        keep = [k for k in range(3) if bool(allowed[i, k])]
        lse = torch.logsumexp(logits[keep], 0)
        total += float(lse - logits[int(rows[i])])
    assert float(loss) == pytest.approx(total / 4, rel=1e-5)


def test_G9b_C2_no_anchors_gives_zero_with_a_gradient():
    from tomocpt.training.discriminative import PrototypeClassifier, prototype_ce_loss

    clf = PrototypeClassifier(2, 4)
    cos = clf.cosine(torch.zeros(0, 4))
    loss = prototype_ce_loss(cos, torch.zeros(0, dtype=torch.long), torch.ones(0, 3, dtype=torch.bool), 10.0,
                             zero_from=clf.prototypes)
    assert float(loss) == 0.0
    loss.backward()
    assert clf.prototypes.grad is not None


def test_G9b_C2_cosine_map_equals_cosine():
    from tomocpt.training.discriminative import PrototypeClassifier

    torch.manual_seed(0)
    clf = PrototypeClassifier(3, 5)
    emb_map = torch.randn(2, 5, 4, 4, 4)
    cm = clf.cosine_map(emb_map)  # [2, 4, 4, 4, 4]
    flat = emb_map.permute(0, 2, 3, 4, 1).reshape(-1, 5)
    ref = clf.cosine(flat).reshape(2, 4, 4, 4, 4).permute(0, 4, 1, 2, 3)
    assert torch.allclose(cm, ref, atol=1e-6)


# --------------------------------------------------------------------------------------
# C3 training step, DDP safety
# --------------------------------------------------------------------------------------

OPTIONS = {
    "embed_ce": dict(contrastive_weight=0.1, embed_ce_weight=0.5),
    "prototype": dict(contrastive_weight=0.1, prototype_head=True),
    "both": dict(contrastive_weight=0.1, embed_ce_weight=0.5, prototype_head=True),
    "embed_ce_only": dict(embed_ce_weight=0.5),
    "prototype_only": dict(prototype_head=True),
}


@pytest.mark.parametrize("scope", ["all", "particles"])
@pytest.mark.parametrize("option", list(OPTIONS))
def test_G9b_C3_every_parameter_gets_a_gradient(tmp_path, scope, option):
    from gate7 import g7_helpers as H7

    with applied_main_config(_cfg(tmp_path / "base", scope=scope, contrastive_weight=0.1)):
        base_loss, _, _ = G3.first_step_loss_and_grads(_model(), _items(), tmp_path / "tb0")
    with applied_main_config(_cfg(tmp_path, scope=scope, **OPTIONS[option])):
        model = _model()
        loss, grads, trainer = G3.first_step_loss_and_grads(model, _items(), tmp_path / "tb")
    assert torch.isfinite(torch.tensor(loss))
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, f"parameters without a gradient: {missing[:10]}"
    assert not H7.trainable_not_in_optimizer(model, trainer)
    if "embed_ce_weight" in OPTIONS[option]:
        assert loss != base_loss
        assert torch.count_nonzero(grads["prototype_classifier.prototypes"]) > 0
    if OPTIONS[option].get("prototype_head"):
        assert torch.count_nonzero(grads["prototype_classifier.alpha"]) > 0


@pytest.mark.parametrize("option", ["embed_ce", "both"])
def test_G9b_C3_batch_without_anchors(tmp_path, option):
    with applied_main_config(_cfg(tmp_path, **OPTIONS[option])):
        loss, grads, _ = G3.first_step_loss_and_grads(_model(), _items(empty=True), tmp_path / "tb")
    assert torch.isfinite(torch.tensor(loss))
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, missing[:10]


def test_G9b_C3_embed_ce_reaches_backbone_and_prototypes(tmp_path):
    with applied_main_config(_cfg(tmp_path, embed_ce_weight=1.0)):
        model = _model()
        batch = G3.collate(_items())
        logits, _, _, emb_map = model._forward_embed(batch["input_data"])
        y_pred, _ = model._probabilities(logits, batch["target_data"], batch)
        terms = model.discriminative_terms(emb_map, y_pred, batch)
    assert terms["contrastive"] is None
    assert terms["n_anchors"] > 0
    terms["embed_ce"].backward()
    assert torch.count_nonzero(model.prototype_classifier.prototypes.grad) > 0
    assert torch.count_nonzero(model.contrastive_head.proj.weight.grad) > 0
    swin = [p.grad for n, p in model.model.named_parameters() if n.startswith("swinViT.") and p.grad is not None]
    assert swin and any(torch.count_nonzero(g) > 0 for g in swin)


def test_G9b_C3_embed_ce_follows_the_contrastive_ramp(tmp_path):
    with applied_main_config(_cfg(tmp_path, contrastive_weight=0.1, embed_ce_weight=0.4,
                                  contrastive_warmup_steps=10)):
        model = _model()
    assert model.embed_ce_weight_at(0) == 0.0
    assert model.embed_ce_weight_at(5) == pytest.approx(0.2)
    assert model.embed_ce_weight_at(10) == pytest.approx(0.4)
    with applied_main_config(_cfg(tmp_path, embed_ce_weight=0.4)):
        model = _model()
    assert model.embed_ce_weight_at(0) == 0.4


def test_G9b_C3_supcon_unchanged_by_embed_ce(tmp_path):
    """One anchor draw for both terms: the SupCon value is the same with and without the CE."""
    vals = []
    for disc in (dict(contrastive_weight=0.1), dict(contrastive_weight=0.1, embed_ce_weight=0.5)):
        with applied_main_config(_cfg(tmp_path, **disc)):
            model = _model()
            batch = G3.collate(_items())
            with torch.no_grad():
                logits, feats = model._forward_features(batch["input_data"])
                y_pred, _ = model._probabilities(logits, batch["target_data"], batch)
                torch.manual_seed(3)
                loss_c, _ = model.contrastive_loss(feats, y_pred, batch, update_bank=False)
                vals.append(float(loss_c))
    assert vals[0] == vals[1]


# --------------------------------------------------------------------------------------
# C4 prototype residual
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("scope", ["all", "particles"])
def test_G9b_C4_zero_alpha_gives_identical_logits(tmp_path, scope):
    with applied_main_config(_cfg(tmp_path, scope=scope, prototype_head=True)):
        model = _model().eval()
    x = G3.fixed_input(2, seed=4)
    with torch.no_grad():
        raw = model.model.forward_features(x)[0]
        out = model(x)
        p = model.predict_step(x)
    assert out.shape == raw.shape
    assert torch.equal(out.float(), raw.float())
    from tomocpt.training.multiclass_loss import class_probabilities

    assert torch.equal(p, class_probabilities(raw, "softmax", None, model._softmax_channels()))


@pytest.mark.parametrize("scope", ["all", "particles"])
def test_G9b_C4_residual_on_particle_channels_only(tmp_path, scope):
    with applied_main_config(_cfg(tmp_path, scope=scope, prototype_head=True)):
        model = _model().eval()
    with torch.no_grad():
        model.prototype_classifier.alpha.fill_(1.0)
    x = G3.fixed_input(1, seed=5)
    with torch.no_grad():
        raw = model.model.forward_features(x)[0].float()
        out = model(x).float()
        p = model.predict_step(x)
        p_all, _ = model._forward_all(x)[0], None
    assert torch.equal(out[:, P:], raw[:, P:])  # filament channel untouched
    assert not torch.allclose(out[:, :P], raw[:, :P])
    assert torch.equal(p_all, out)  # every forward path adds the same residual
    if scope == "particles":
        assert torch.allclose(p[:, P:], torch.sigmoid(raw[:, P:]))
    # residual = alpha * s * (cos_k - cos_bg), trilinear from the decoder2 grid
    emb = model._forward_embed(x)[3]
    cos = model.prototype_classifier.cosine_map(emb)
    r = 10.0 * (cos[:, :P] - cos[:, P:])
    up = torch.nn.functional.interpolate(r, size=raw.shape[2:], mode="trilinear", align_corners=False)
    assert torch.allclose(out[:, :P] - raw[:, :P], up, atol=1e-5)


def test_G9b_C4_output_contract(tmp_path):
    with applied_main_config(_cfg(tmp_path, prototype_head=True)):
        model = _model().eval()
    x = G3.fixed_input(1, seed=6)
    with torch.no_grad():
        net_out = model.model(x)
    assert isinstance(net_out, tuple) and len(net_out) == 2
    assert model(x).shape == (1, 3, 64, 64, 64)


# --------------------------------------------------------------------------------------
# C5 level masking
# --------------------------------------------------------------------------------------

def test_G9b_C5_unavailable_class_is_masked(tmp_path):
    avail = [True, False, True]
    with applied_main_config(_cfg(tmp_path, embed_ce_weight=1.0)):
        model = _model()
        batch = G3.collate(_items(available=avail))
        with torch.no_grad():
            logits, _, _, emb_map = model._forward_embed(batch["input_data"])
            y_pred, _ = model._probabilities(logits, batch["target_data"], batch)
            torch.manual_seed(1)
            a = float(model.discriminative_terms(emb_map, y_pred, batch)["embed_ce"])
            model.prototype_classifier.prototypes[1].mul_(-3.0).add_(0.7)
            torch.manual_seed(1)
            b = float(model.discriminative_terms(emb_map, y_pred, batch)["embed_ce"])
            model.prototype_classifier.prototypes[0].mul_(-3.0).add_(0.7)
            torch.manual_seed(1)
            c = float(model.discriminative_terms(emb_map, y_pred, batch)["embed_ce"])
    assert a == pytest.approx(b, rel=1e-6)  # class 1 unavailable: its prototype plays no part
    assert c != pytest.approx(b, rel=1e-6)


# --------------------------------------------------------------------------------------
# C6 checkpoints
# --------------------------------------------------------------------------------------

def _source_ckpt(root):
    """Contrastive-only v2 picker written with the defaults of this gate (an "old" checkpoint)."""
    root.mkdir(parents=True, exist_ok=True)
    with applied_main_config(_cfg(root, contrastive_weight=0.1)):
        model = _model()
        trainer = G3.cpu_trainer(root / "tb")
        trainer.fit(model, datamodule=G3.datamodule(_items()))
        path = root / "src.ckpt"
        trainer.save_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def source_ckpt(tmp_path_factory):
    with applied_main_config(None):
        return _source_ckpt(tmp_path_factory.mktemp("g9b_src"))


def test_G9b_C6_old_checkpoint_loads_strictly_for_predict(source_ckpt):
    from tomocpt.networks.inference import InferenceModel

    with applied_main_config(None):
        model = InferenceModel.load_from_checkpoint(str(source_ckpt), map_location="cpu", strict=True)
    assert model.prototype_classifier is None


def test_G9b_C6_fine_tune_with_both_options(tmp_path, source_ckpt):
    from tomocpt.networks.inference import InferenceModel
    from tomocpt.training.helpers import _create_model

    x = G3.fixed_input(1, seed=7)
    with applied_main_config(None):
        src = InferenceModel.load_from_checkpoint(str(source_ckpt), map_location="cpu", strict=True).eval()
        with torch.no_grad():
            src_logits = src(x).float()
    src_sd = torch.load(source_ckpt, map_location="cpu", weights_only=False)["state_dict"]

    cfg = _cfg(tmp_path, contrastive_weight=0.1, embed_ce_weight=0.5, prototype_head=True)
    with applied_main_config(cfg) as g:
        g.train.resume_from = None
        g.train.fine_tune_from = source_ckpt
        g.train.use_distillation = False
        torch.manual_seed(0)
        model, resume, _ = _create_model(g.train)
        groups = G3.param_groups(model)
    assert resume is None
    for k, v in model.contrastive_head.state_dict().items():
        assert torch.equal(v, src_sd["contrastive_head." + k]), k
    assert torch.equal(model.contrastive_bank.queue, src_sd["contrastive_bank.queue"])
    assert float(model.prototype_classifier.alpha) == 0.0
    new = {id(model.prototype_classifier.prototypes), id(model.prototype_classifier.alpha)}
    for mult, params in groups:
        if new & {id(p) for p in params}:
            assert mult == 1.0
    assert sum(len(new & {id(p) for p in params}) for _, params in groups) == 2
    model.eval()
    with torch.no_grad():
        out = model(x).float()
    assert torch.equal(out, src_logits)


def test_G9b_C6_new_checkpoint_round_trip(tmp_path):
    from tomocpt.networks.inference import InferenceModel

    with applied_main_config(_cfg(tmp_path, embed_ce_weight=0.5, prototype_head=True)):
        model = _model()
        trainer = G3.cpu_trainer(tmp_path / "tb")
        trainer.fit(model, datamodule=G3.datamodule(_items()))
        path = tmp_path / "new.ckpt"
        trainer.save_checkpoint(path)
    with applied_main_config(None):
        loaded = InferenceModel.load_from_checkpoint(str(path), map_location="cpu", strict=True)
    assert loaded.prototype_classifier is not None and loaded.prototype_classifier.alpha is not None


# --------------------------------------------------------------------------------------
# C7 backfill
# --------------------------------------------------------------------------------------

def test_G9b_C7_backfill_of_new_fields():
    from tomocpt.configManager.configManager import backfill_missing_fields
    from tomocpt.mainConfig import MainConfig

    old = MainConfig()
    for name in ("embed_ce_weight", "prototype_head", "prototype_scale"):
        old.train.discriminative.__dict__.pop(name, None)
    filled = backfill_missing_fields(old, MainConfig())
    d = filled.train.discriminative
    assert "embed_ce_weight" in d.__dict__ and d.embed_ce_weight == 0.0
    assert "prototype_head" in d.__dict__ and d.prototype_head is False
    assert "prototype_scale" in d.__dict__ and d.prototype_scale == 10.0
