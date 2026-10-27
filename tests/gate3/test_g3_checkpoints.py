"""G3-C7 (class-table change), G3-C8, G3-C9 (v2 part), G3-C10, G3-C11: model creation,
fine-tuning, resuming, distillation and inference loading with pipeline-v2 models."""

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as H
from utils.legacy import legacy_reference
from utils.tiny import applied_main_config

ATOL = 1e-6


def _fine_tune_cfg(cfg, ckpt):
    cfg.train.resume_from = None
    cfg.train.fine_tune_from = ckpt
    cfg.train.use_distillation = False
    return cfg


# --------------------------------------------------------------------------------------
# G3-C7: class-table change through _create_model
# --------------------------------------------------------------------------------------

def test_G3_C7_class_table_change_copies_matching_channels(tmp_path, v2_ckpts):
    """Fine-tune a v2 checkpoint (classes 1: A, 2: B) into the table (2: B, 3: C): the channel
    of class 2 is copied (matched by class number), the new class 3 starts from the mean of the
    old channels, everything else is loaded; a new run starts (no optimizer state)."""
    from tomocpt.training.helpers import _create_model

    src = v2_ckpts("c2")
    old = src["logits"][None]  # [1, 2, ...] channels (A, B)
    cfg = H.v2_config(tmp_path, ((2, "B", 300.0), (3, "C", 200.0)))
    with applied_main_config(cfg) as g:
        _fine_tune_cfg(g, src["path"])
        model, resume, _ = _create_model(g.train)
        assert resume is None
        model.eval()
        with torch.no_grad():
            new = model(src["x"])
    assert tuple(new.shape) == (1, 2, 64, 64, 64)
    np.testing.assert_allclose(new[:, 0].numpy(), old[:, 1].numpy(), rtol=0, atol=1e-5)
    np.testing.assert_allclose(new[:, 1].numpy(), old.mean(dim=1).numpy(), rtol=0, atol=1e-5)


def test_G3_C7_same_class_table_keeps_every_channel(tmp_path, v2_ckpts):
    from tomocpt.training.helpers import _create_model

    src = v2_ckpts("c2")
    cfg = H.v2_config(tmp_path, H.CLASSES_AB)
    with applied_main_config(cfg) as g:
        _fine_tune_cfg(g, src["path"])
        model, resume, _ = _create_model(g.train)
        model.eval()
        with torch.no_grad():
            new = model(src["x"])
    np.testing.assert_allclose(new.numpy(), src["logits"][None].numpy(), rtol=0, atol=1e-5)


# --------------------------------------------------------------------------------------
# G3-C8: legacy checkpoint -> v2 fine-tune through _create_model
# --------------------------------------------------------------------------------------

V2_TARGETS = {
    "C1_cond": (((1, "A", 150.0),), True),
    "C3": (H.CLASSES_ABC, False),
    "C3_cond": (H.CLASSES_ABC, True),
}


@pytest.mark.parametrize("target", sorted(V2_TARGETS))
def test_G3_C8_legacy_checkpoint_into_v2_model(tmp_path, legacy_ckpt, target):
    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.training.helpers import _create_model

    classes, cond = V2_TARGETS[target]
    C = len(classes)
    ref = legacy_reference(legacy_ckpt)
    x = torch.from_numpy(ref["x"])
    saved = {k[len("model."):]: v for k, v in H.state_dict_of(legacy_ckpt)["state_dict"].items()
             if k.startswith("model.")}

    cfg = H.v2_config(tmp_path, classes, cond=cond)
    with applied_main_config(cfg) as g:
        _fine_tune_cfg(g, legacy_ckpt)
        model, resume, _ = _create_model(g.train)
        assert isinstance(model, BasePickingModel)
        assert resume is None  # a new run: no optimizer state from the legacy run
        sd = model.model.state_dict()
        new_keys, old_keys = set(sd), set(saved)
        assert old_keys - new_keys == set()
        extra = new_keys - old_keys
        assert all(k.startswith("cond.") for k in extra), sorted(extra)
        assert bool(extra) == cond
        for k in old_keys:
            if k.startswith("out."):
                assert sd[k].shape[0] == C, k
            else:
                assert torch.equal(sd[k].cpu(), saved[k].cpu()), k
        model.eval()
        with torch.no_grad():
            outs = [model(x)]
            if cond:
                outs += [model(x, apix=H.apix_tensor(a)) for a in (7.0, 10.0, 20.0)]
    for logits in outs:
        assert tuple(logits.shape) == (1, C, 64, 64, 64)
        for c in range(C):
            np.testing.assert_allclose(logits[:, c].numpy(), ref["logits"][:, 0], rtol=0, atol=ATOL)


def test_G3_C8_ssl_checkpoint_into_v2_model(tmp_path, legacy_ssl_ckpt):
    """SSL checkpoint -> v2 picking model follows the same rules (backbone loaded, the
    one-channel ``out`` copied into every channel)."""
    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.training.helpers import _create_model

    ref = legacy_reference(legacy_ssl_ckpt)
    cfg = H.v2_config(tmp_path, H.CLASSES_ABC, cond=True)
    with applied_main_config(cfg) as g:
        _fine_tune_cfg(g, legacy_ssl_ckpt)
        model, resume, _ = _create_model(g.train)
        assert isinstance(model, BasePickingModel)
        assert resume is None
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(ref["x"]), apix=H.apix_tensor(12.0))
    assert logits.shape[1] == 3
    for c in range(3):
        np.testing.assert_allclose(logits[:, c].numpy(), ref["logits"][:, 0], rtol=0, atol=ATOL)


# --------------------------------------------------------------------------------------
# G3-C9: parameter groups (v2)
# --------------------------------------------------------------------------------------

def _check_groups(model, fine_tune):
    names = {id(p): n for n, p in model.named_parameters()}
    groups = H.param_groups(model)
    seen = [id(p) for _, params in groups for p in params]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(seen) == len(set(seen)), "a parameter is in more than one group"
    assert set(seen) == trainable, "a trainable parameter is in no group"
    if not fine_tune:
        assert len(groups) == 1
        return
    for mult, params in groups:
        for p in params:
            n = names[id(p)]
            inner = n[len("model."):] if n.startswith("model.") else n
            if inner.startswith("swinViT."):
                want = 0.01
            elif inner.startswith("out.") or inner.startswith("cond."):
                want = 1.0
            elif "encoder" in inner:
                want = 0.1
            else:
                want = 1.0
            assert mult == want, (n, mult)


@pytest.mark.parametrize("fine_tune", [False, True])
def test_G3_C9_v2_parameter_groups(tmp_path, fine_tune):
    from tomocpt.networks.pickingModel import BasePickingModel

    cfg = H.v2_config(tmp_path, H.CLASSES_ABC, cond=True)
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue="prev.ckpt" if fine_tune else None)
        assert any(n.startswith("model.cond.") for n, _ in model.named_parameters())
        _check_groups(model, fine_tune)


def test_G3_C9_groups_after_fine_tune_from_legacy(tmp_path, legacy_ckpt):
    from tomocpt.training.helpers import _create_model

    cfg = H.v2_config(tmp_path, H.CLASSES_ABC, cond=True)
    with applied_main_config(cfg) as g:
        _fine_tune_cfg(g, legacy_ckpt)
        model, _, _ = _create_model(g.train)
        _check_groups(model, fine_tune=True)


# --------------------------------------------------------------------------------------
# G3-C10: errors
# --------------------------------------------------------------------------------------

def _v1_batch():
    b = H.v2_batch([15.0], B=2, seed=1)
    return {"input_data": b["input_data"], "target_data": b["target_data"]}


def test_G3_C10_conditioning_without_apix_raises(tmp_path):
    from tomocpt.networks.pickingModel import BasePickingModel

    cfg = H.v2_config(tmp_path, H.CLASSES_AB, cond=True)
    full = H.v2_batch([15.0, 30.0], B=2, seed=2)
    no_apix = {k: v for k, v in full.items() if k != "apix_work"}
    v1_like = {k: full[k] for k in ("input_data", "target_data")}
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        for batch in (no_apix, v1_like):
            with pytest.raises(ValueError):
                model.training_step(batch, 1)
            with pytest.raises(ValueError):
                model.validation_step(batch, 1)


@pytest.mark.parametrize("change", ["class_table", "scale_conditioning"])
def test_G3_C10_resume_with_a_different_architecture_raises(tmp_path, v2_ckpts, change):
    from tomocpt.training.helpers import _create_model

    src = v2_ckpts("c2")
    if change == "class_table":
        cfg = H.v2_config(tmp_path, H.CLASSES_ABC, cond=False)
    else:
        cfg = H.v2_config(tmp_path, H.CLASSES_AB, cond=True)
    with applied_main_config(cfg) as g:
        g.train.resume_from = src["path"]
        g.train.fine_tune_from = None
        g.train.use_distillation = False
        with pytest.raises(ValueError):
            _create_model(g.train)


def test_G3_C10_resume_with_the_same_architecture_works(tmp_path, v2_ckpts):
    from tomocpt.training.helpers import _create_model

    src = v2_ckpts("c2_cond")
    cfg = H.v2_config(tmp_path, H.CLASSES_AB, cond=True)
    with applied_main_config(cfg) as g:
        g.train.resume_from = src["path"]
        g.train.fine_tune_from = None
        g.train.use_distillation = False
        model, resume, _ = _create_model(g.train)
        assert str(resume) == str(src["path"])
        model.eval()
        with torch.no_grad():
            logits = model(src["x"], apix=H.apix_tensor(20.0))
    np.testing.assert_allclose(logits.numpy(), src["logits"][20.0].numpy(), rtol=0, atol=1e-5)


def test_G3_C10_resume_legacy_checkpoint_with_conditioning_raises(tiny_global, legacy_ckpt):
    from tomocpt.training.helpers import _create_model

    tiny_global.train.network.scale_conditioning = True
    tiny_global.train.resume_from = legacy_ckpt
    tiny_global.train.fine_tune_from = None
    tiny_global.train.use_distillation = False
    with pytest.raises(ValueError):
        _create_model(tiny_global.train)


STUDENTS = {
    "v2_single": (((1, "A", 150.0),), False),
    "v2_multi": (H.CLASSES_AB, False),
    "v2_cond": (((1, "A", 150.0),), True),
}


@pytest.mark.parametrize("student", sorted(STUDENTS))
def test_G3_C10_distillation_with_v2_student_raises(tmp_path, legacy_ckpt, student):
    from tomocpt.networks.distillationModel import DistillationPickingModel

    classes, cond = STUDENTS[student]
    cfg = H.v2_config(tmp_path, classes, cond=cond)
    with applied_main_config(cfg):
        with pytest.raises(NotImplementedError):
            DistillationPickingModel(teacher_checkpoint_path=str(legacy_ckpt), lr=1e-3)


def test_G3_C10_distillation_through_create_model_raises(tmp_path, legacy_ckpt):
    from tomocpt.training.helpers import _create_model

    cfg = H.v2_config(tmp_path, H.CLASSES_AB, cond=False)
    with applied_main_config(cfg) as g:
        g.train.resume_from = None
        g.train.fine_tune_from = legacy_ckpt
        g.train.use_distillation = True
        with pytest.raises(NotImplementedError):
            _create_model(g.train)


def test_G3_C10_unet_with_scale_conditioning_raises():
    from tomocpt.defaultConfigs.network_config import ModelType, NetworkConfig

    net = NetworkConfig(model_type=ModelType.UNET, scale_conditioning=True)
    with pytest.raises(NotImplementedError):
        net.build_model(require_labels=True, img_size=64)


# --------------------------------------------------------------------------------------
# G3-C11: InferenceModel loads a v2 checkpoint strictly
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["c2", "c2_cond"])
def test_G3_C11_inference_model_loads_v2_checkpoint(v2_ckpts, name):
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.networks.inference import InferenceModel

    src = v2_ckpts(name)
    model = InferenceModel.load_from_checkpoint(src["path"], map_location="cpu", strict=True).eval()
    table = class_table_from_config(model.hparams.config.prepData)
    assert [r.name for r in table] == ["A", "B"]
    assert bool(model.hparams.config.train.network.scale_conditioning) == (name == "c2_cond")
    x = src["x"]
    with torch.no_grad():
        for apix, want in src["logits"].items():
            if apix is None:
                logits = model(x)
                pred = model.predict_step(x, 0)
            else:
                logits = model(x, apix=H.apix_tensor(apix))
                pred = model.predict_step(x, 0, apix=H.apix_tensor(apix))
            assert tuple(logits.shape) == (1, 2, 64, 64, 64)
            np.testing.assert_allclose(logits.numpy(), want.numpy(), rtol=0, atol=1e-5)
            np.testing.assert_allclose(pred.numpy(), torch.sigmoid(want).numpy(), rtol=0, atol=1e-6)
        if name == "c2_cond":
            # the saved FiLM layers were trained for one step: the voxel size matters
            assert float((src["logits"][10.0] - src["logits"][20.0]).abs().max()) > 0
