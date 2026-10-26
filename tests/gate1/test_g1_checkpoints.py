"""G1-C4, G1-C5, G1-C6, G1-C11: checkpoints written by ``v2-gate-0-base`` keep working.

The legacy checkpoints are tiny models built with base code (``utils.legacy``); the
reference outputs were computed by the base code right after saving.
"""

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from utils.legacy import legacy_reference, zenodo_checkpoint
from utils.tiny import random_chunk_datamodule

ATOL = 1e-6


def _state_dict(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _trainer(tmp_path, **kw):
    kw.setdefault("max_epochs", 1)
    return pl.Trainer(accelerator="cpu", devices=1, logger=TensorBoardLogger(tmp_path / "tb"),
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0, **kw)


def test_G1_C4_legacy_picking_checkpoint_loads(legacy_ckpt):
    from tomocpt.networks.inference import InferenceModel
    from tomocpt.networks.pickingModel import BasePickingModel

    ref = legacy_reference(legacy_ckpt)
    x = torch.from_numpy(ref["x"])

    model = BasePickingModel.load_from_checkpoint(legacy_ckpt, map_location="cpu").eval()
    with torch.no_grad():
        out = model.model(x)
        logits = model(x)
    assert isinstance(out, tuple) and len(out) == 2  # (logits, hidden_states_out[4])
    assert tuple(logits.shape) == (1, 1, 64, 64, 64)
    np.testing.assert_allclose(logits.numpy(), ref["logits"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(out[1].numpy(), ref["hidden4"], rtol=0, atol=ATOL)

    inf = InferenceModel.load_from_checkpoint(legacy_ckpt, map_location="cpu", strict=True).eval()
    with torch.no_grad():
        inf_logits = inf(x)
    np.testing.assert_allclose(inf_logits.numpy(), ref["logits"], rtol=0, atol=ATOL)
    # predict reads the target particle size from the checkpoint config
    assert inf.hparams.config.prepData.desired_particle_pixel_size == 20


def test_G1_C4_legacy_ssl_checkpoint_loads(legacy_ssl_ckpt):
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    ref = legacy_reference(legacy_ssl_ckpt)
    model = SelfSupervisedModel.load_from_checkpoint(legacy_ssl_ckpt, map_location="cpu").eval()
    with torch.no_grad():
        logits, hidden4 = model.model(torch.from_numpy(ref["x"]))
        rot, contr, rec = model(torch.from_numpy(ref["x"]).repeat(2, 1, 1, 1, 1))
    np.testing.assert_allclose(logits.numpy(), ref["logits"], rtol=0, atol=ATOL)
    np.testing.assert_allclose(hidden4.numpy(), ref["hidden4"], rtol=0, atol=ATOL)
    assert tuple(rot.shape) == (2, 4)
    assert tuple(rec.shape) == (2, 1, 64, 64, 64)


def test_G1_C4_ssl_to_picking_through_create_model(tiny_global, legacy_ssl_ckpt):
    from tomocpt.defaultConfigs.train_config import TrainingModes
    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.training.helpers import _create_model

    cfg = tiny_global
    cfg.train.mode = TrainingModes.picking
    cfg.train.resume_from = None
    cfg.train.fine_tune_from = legacy_ssl_ckpt
    cfg.train.use_distillation = False
    model, resume_from_ckpt, _ = _create_model(cfg.train)

    assert isinstance(model, BasePickingModel)
    assert resume_from_ckpt is None
    saved = _state_dict(legacy_ssl_ckpt)["state_dict"]
    backbone = model.model.state_dict()
    assert backbone, "empty backbone"
    for k, v in backbone.items():
        assert torch.equal(v.cpu(), saved["model." + k]), k

    ref = legacy_reference(legacy_ssl_ckpt)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(ref["x"]))
    np.testing.assert_allclose(logits.numpy(), ref["logits"], rtol=0, atol=ATOL)


def test_G1_C5_legacy_checkpoint_resumes(tiny_global, legacy_ckpt, tmp_path):
    from tomocpt.training.helpers import _create_model

    cfg = tiny_global
    cfg.train.resume_from = legacy_ckpt
    cfg.train.fine_tune_from = None
    cfg.train.use_distillation = False
    model, resume_from_ckpt, _ = _create_model(cfg.train)
    assert str(resume_from_ckpt) == str(legacy_ckpt)

    saved = _state_dict(legacy_ckpt)
    trainer = _trainer(tmp_path, max_epochs=saved["epoch"] + 2)
    trainer.fit(model, datamodule=random_chunk_datamodule(), ckpt_path=resume_from_ckpt)

    assert trainer.current_epoch == saved["epoch"] + 2
    assert trainer.global_step > saved["global_step"]
    # optimizer state restored with the checkpoint's grouping (from scratch: one group)
    assert len(trainer.optimizers[0].param_groups) == len(saved["optimizer_states"][0]["param_groups"])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_G1_C6_distillation_with_legacy_teacher(tiny_global, legacy_ckpt, tmp_path):
    from tomocpt.networks.distillationModel import DistillationPickingModel
    from tomocpt.training.helpers import _create_model

    cfg = tiny_global
    cfg.train.resume_from = None
    cfg.train.fine_tune_from = legacy_ckpt
    cfg.train.use_distillation = True
    model, resume_from_ckpt, _ = _create_model(cfg.train)
    assert isinstance(model, DistillationPickingModel)
    assert resume_from_ckpt is None
    assert all(not p.requires_grad for p in model.teacher.parameters())

    student_before = {k: v.clone() for k, v in model.model.state_dict().items()}
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    trainer.fit(model, datamodule=random_chunk_datamodule())
    assert trainer.global_step == 1
    assert np.isfinite(float(trainer.callback_metrics["loss"]))

    saved = _state_dict(legacy_ckpt)["state_dict"]
    for k, v in model.teacher.model.state_dict().items():
        assert torch.equal(v.cpu(), saved["model." + k]), f"teacher changed: {k}"
    changed = [k for k, v in model.model.state_dict().items() if not torch.equal(v, student_before[k])]
    assert changed, "the student did not train"


def test_G1_C11_zenodo_checkpoint_loads_for_predict():
    path = zenodo_checkpoint()
    if path is None:
        pytest.skip("set TOMOCPT_ZENODO_CKPT to the public pretrained checkpoint to run this test")
    from tomocpt.networks.inference import InferenceModel

    model = InferenceModel.load_from_checkpoint(path, map_location="cpu", strict=True).eval()
    target_px = model.hparams.config.prepData.desired_particle_pixel_size
    assert isinstance(target_px, (int, float)) and target_px > 0
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 64, 64, 64))
    assert tuple(logits.shape) == (1, 1, 64, 64, 64)
    assert torch.isfinite(logits).all()


@pytest.mark.slow
def test_G1_C4_legacy_checkpoint_on_gpu(legacy_ckpt):
    """predict runs on GPU: the legacy checkpoint loads there and matches the CPU reference.
    TF32 is switched off for the comparison; atol 1e-4 leaves a wide margin over the
    observed fp32 GPU-vs-CPU kernel differences."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from tomocpt.networks.inference import InferenceModel

    ref = legacy_reference(legacy_ckpt)
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        model = InferenceModel.load_from_checkpoint(legacy_ckpt, map_location="cuda", strict=True).eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(ref["x"]).cuda()).float().cpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
    np.testing.assert_allclose(logits.numpy(), ref["logits"], rtol=0, atol=1e-4)
