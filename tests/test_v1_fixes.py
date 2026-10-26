"""Regression tests for small v1 defects fixed before the scale-free work.

- IMOD ``prepare_data`` raised NameError (undefined ``class_id``).
- ``SelfSupervisedModel`` could not be built at all (``require_labels`` was passed twice).
- The optimizer never saw the SwinUNETR ``out`` layer during fine-tuning, nor the
  self-supervised heads in any run.
- Resuming a fine-tuning run rebuilt the optimizer without the fine-tuning groups.
- The "recons" TensorBoard panel showed the input instead of the reconstruction.

Models here are tiny SwinUNETRs (``feature_size=12``) so everything runs on CPU.
"""

import copy

import mrcfile
import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from tomocpt.mainConfig import mainConfig
from tomocpt.networks import baseModel
from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

CHUNK = 64


@pytest.fixture
def tiny_config():
    """Shrink the global config to a CPU-sized model and restore it afterwards."""
    saved_train = copy.deepcopy(mainConfig.train)
    mainConfig.train.batch_size = 2
    mainConfig.train.network.SwinUNETR.feature_size = 12
    mainConfig.train.network.SwinUNETR.use_checkpoint = False
    mainConfig.train.use_gpus = False
    yield mainConfig
    for name in vars(saved_train):
        setattr(mainConfig.train, name, getattr(saved_train, name))


def _optimizer_param_ids(model):
    """ids of the parameters the optimizer would update, one entry per group."""
    parameters = model._get_model_parameters_for_optimizer()
    parameters = parameters if isinstance(parameters, list) else list(parameters)
    if parameters and isinstance(parameters[0], dict):
        return [[id(p) for p in g["params"]] for g in parameters]
    return [[id(p) for p in parameters]]


def _assert_every_trainable_param_once(model):
    groups = _optimizer_param_ids(model)
    flat = [i for g in groups for i in g]
    assert len(flat) == len(set(flat)), "a parameter is in more than one group"
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in set(flat)]
    assert not missing, f"never optimised: {missing}"
    assert set(flat) == trainable


# --- IMOD ------------------------------------------------------------------

def test_prepare_picking_imod_writes_tracking_csv_and_labels(tmp_path, monkeypatch):
    from tomocpt.labels import helpers, mod

    tomo_dir = tmp_path / "tomos"
    tomo_dir.mkdir()
    with mrcfile.new(tomo_dir / "TS_01.mrc") as mrc:
        mrc.set_data(np.zeros((40, 50, 60), dtype=np.float32))
        mrc.voxel_size = 10.0

    mod_dir = tmp_path / "mods"
    mod_dir.mkdir()
    (mod_dir / "TS_01.mod").write_bytes(b"")  # content comes from the stubbed reader
    monkeypatch.setattr(
        mod.imodmodel, "read", lambda path: pd.DataFrame({"x": [30.0], "y": [25.0], "z": [20.0]})
    )

    out_dir = tmp_path / "out"
    helpers.prepare_picking_imod(mod_dir, tomo_dir, out_dir, particle_diameter_angst=100.0)

    tracking = pd.read_csv(out_dir / "imod_picking_tracking.csv")
    assert len(tracking) == 1
    label_path = tracking.loc[0, "label_path"]
    assert "class_all" in label_path
    with mrcfile.open(label_path) as mrc:
        label = mrc.data
    assert label.shape == (40, 50, 60)
    assert label[20, 25, 30] == pytest.approx(1.0)


# --- Parameter groups ------------------------------------------------------

def test_picking_from_scratch_optimizer_is_unchanged(tiny_config):
    """Training from scratch already optimised everything; keep the order identical."""
    model = BasePickingModel(lr=1e-3, train_continue=None)
    assert _optimizer_param_ids(model) == [[id(p) for p in model.model.parameters()]]


def test_picking_finetune_optimises_out_layer_at_full_lr(tiny_config):
    model = BasePickingModel(lr=1e-3, train_continue="pretrained.ckpt")
    _assert_every_trainable_param_once(model)

    parameters = model._get_model_parameters_for_optimizer()
    out_ids = {id(p) for p in model.model.out.parameters()}
    (out_group,) = [g for g in parameters if out_ids & {id(p) for p in g["params"]}]
    assert out_group["lr_mult"] == 1.0
    assert {g["lr_mult"] for g in parameters} == {0.01, 0.1, 1.0}


@pytest.mark.parametrize("train_continue", [None, "pretrained.ckpt"])
def test_self_supervised_model_builds_and_optimises_its_heads(tiny_config, train_continue):
    model = SelfSupervisedModel(lr=1e-3, train_continue=train_continue)
    _assert_every_trainable_param_once(model)
    head_ids = {id(p) for p in model.rotation_head.parameters()} | {
        id(p) for p in model.contrastive_head.parameters()
    }
    assert head_ids <= {i for g in _optimizer_param_ids(model) for i in g}


# --- SSL TensorBoard panel -------------------------------------------------

class _RecordingExperiment:
    def __init__(self):
        self.images = {}

    def add_image(self, tag, img, global_step=None):
        self.images[tag] = img


def test_ssl_recons_panel_shows_the_reconstruction(tiny_config, monkeypatch):
    model = SelfSupervisedModel(lr=1e-3)
    b = 2
    x = torch.zeros(b, 1, CHUNK, CHUNK, CHUNK)
    x1, x2 = x + 1, x + 2
    imgs_recon = torch.cat([x + 3, x + 4])
    zero = torch.tensor(0.0)
    monkeypatch.setattr(
        model, "_step", lambda batch, idx: (zero, (zero, zero, zero), (x, x1, x2), (None, None), imgs_recon)
    )
    monkeypatch.setattr(model, "log", lambda *a, **k: None)
    experiment = _RecordingExperiment()
    monkeypatch.setattr(SelfSupervisedModel, "logger", property(lambda self: type("L", (), {"experiment": experiment})()))

    model.training_step({"input_data": x}, 0)
    model.validation_step({"input_data": x}, 0)

    expected = torchvision.utils.make_grid(imgs_recon[:b, :1, CHUNK // 2])
    for tag in ("recons_train", "recons_val"):
        assert torch.equal(experiment.images[tag], expected)


# --- Resume ----------------------------------------------------------------

class _RandomChunks(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {
            "input_data": torch.randn(1, CHUNK, CHUNK, CHUNK, generator=g),
            "target_data": (torch.rand(1, CHUNK, CHUNK, CHUNK, generator=g) > 0.99).float(),
        }


class _RandomData(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(_RandomChunks(), batch_size=2)

    def val_dataloader(self):
        return DataLoader(_RandomChunks(), batch_size=2)


def _fit(model, tmp_path, max_epochs, ckpt_path=None):
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator="cpu", devices=1, logger=TensorBoardLogger(tmp_path / "tb"),
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=_RandomData(), ckpt_path=ckpt_path)
    return trainer


def _resume_through_create_model(tmp_path, ckpt):
    from tomocpt.training.helpers import _create_model

    mainConfig.train.resume_from = str(ckpt)
    mainConfig.train.fine_tune_from = None
    mainConfig.train.use_distillation = False
    model, resume_from_ckpt, _ = _create_model(mainConfig.train)
    assert resume_from_ckpt == str(ckpt)
    trainer = _fit(model, tmp_path, max_epochs=2, ckpt_path=resume_from_ckpt)
    return model, trainer


def test_resuming_a_finetune_run_keeps_its_parameter_groups(tiny_config, tmp_path):
    first = BasePickingModel(lr=1e-3, train_continue="pretrained.ckpt")
    trainer = _fit(first, tmp_path, max_epochs=1)
    ckpt = tmp_path / "finetune.ckpt"
    trainer.save_checkpoint(ckpt)
    n_groups = len(trainer.optimizers[0].param_groups)
    assert n_groups == 4

    resumed, trainer = _resume_through_create_model(tmp_path, ckpt)
    assert resumed.hparams.train_continue == "pretrained.ckpt"
    assert len(trainer.optimizers[0].param_groups) == n_groups


def test_resuming_a_run_saved_before_the_fix_keeps_its_old_groups(tiny_config, tmp_path, monkeypatch):
    """An optimizer state saved with the old grouping (no `out` group) still resumes."""
    with monkeypatch.context() as m:
        m.setattr(baseModel.BaseModel, "_resume_needs_legacy_groups", lambda *a: True)
        first = BasePickingModel(lr=1e-3, train_continue="pretrained.ckpt")
        trainer = _fit(first, tmp_path, max_epochs=1)
    assert len(trainer.optimizers[0].param_groups) == 3
    ckpt = tmp_path / "legacy_finetune.ckpt"
    trainer.save_checkpoint(ckpt)

    _, trainer = _resume_through_create_model(tmp_path, ckpt)
    assert len(trainer.optimizers[0].param_groups) == 3


# --- Self-supervised training end to end -----------------------------------

def test_ssl_fit_logs_the_metric_the_scheduler_and_early_stopping_monitor(tiny_config, tmp_path):
    from pytorch_lightning.callbacks import EarlyStopping

    model = SelfSupervisedModel(lr=1e-3)
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=TensorBoardLogger(tmp_path / "tb"),
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, callbacks=[EarlyStopping(monitor="val_loss_smooth")],
    )
    trainer.fit(model, datamodule=_RandomData())
    assert "val_loss_smooth" in trainer.callback_metrics


def test_ssl_checkpoint_transfers_to_picking(tiny_config, tmp_path):
    from tomocpt.defaultConfigs.train_config import TrainingModes
    from tomocpt.training.helpers import _create_model

    mainConfig.train.mode = TrainingModes.selfSupervised
    ssl = SelfSupervisedModel(lr=1e-3)
    trainer = _fit(ssl, tmp_path, max_epochs=1)
    ckpt = tmp_path / "ssl.ckpt"
    trainer.save_checkpoint(ckpt)

    mainConfig.train.mode = TrainingModes.picking
    mainConfig.train.resume_from = None
    mainConfig.train.fine_tune_from = str(ckpt)
    mainConfig.train.use_distillation = False
    model, resume_from_ckpt, _ = _create_model(mainConfig.train)

    assert isinstance(model, BasePickingModel) and resume_from_ckpt is None
    ssl_backbone = ssl.model.state_dict()
    for k, v in model.model.state_dict().items():
        assert torch.equal(v, ssl_backbone[k]), k
    _assert_every_trainable_param_once(model)
    _fit(model, tmp_path, max_epochs=1)
