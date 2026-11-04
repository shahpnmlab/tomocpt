"""G4e-C1..C5: early stopping fields and schedule-free RAdam."""

import builtins
import copy

import pytest
import torch

from gate3 import g3_helpers as G3
from utils.tiny import applied_main_config

CLASSES = ((1, "A", 150.0), (2, "B", 300.0))


def _cfg(tmp_path, **train):
    cfg = G3.v2_config(tmp_path, CLASSES, cond=False)
    for k, v in train.items():
        setattr(cfg.train, k, v)
    return cfg


def _model(train_continue=None):
    from tomocpt.networks.pickingModel import BasePickingModel

    torch.manual_seed(0)
    return BasePickingModel(lr=1e-3, train_continue=train_continue)


def _callbacks(mc, model):
    from pytorch_lightning.callbacks import ModelCheckpoint

    from tomocpt.training.helpers import _setup_callbacks

    return _setup_callbacks(mc.train, model, ModelCheckpoint(monitor="val_loss"))


def _of(cbs, cls):
    return [c for c in cbs if isinstance(c, cls)]


def test_G4e_C1_defaults(tmp_path):
    from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging

    from tomocpt.defaultConfigs.train_config import TrainConfig

    t = TrainConfig()
    assert t.early_stop_min_delta == 0.005 and t.early_stop_patience == 10 and t.schedule_free is False
    with applied_main_config(_cfg(tmp_path)) as mc:
        model = _model()
        cbs = _callbacks(mc, model)
        es = _of(cbs, EarlyStopping)
        assert len(es) == 1
        assert es[0].monitor == "val_loss_smooth" and es[0].patience == 10 and es[0].mode == "min"
        assert abs(es[0].min_delta) == pytest.approx(0.005)
        assert len(_of(cbs, StochasticWeightAveraging)) == 1
        conf = model.configure_optimizers()
    assert type(conf["optimizer"]) is torch.optim.RAdam
    assert isinstance(conf["lr_scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert conf["monitor"] == "val_loss_smooth"


def test_G4e_C2_early_stopping_fields(tmp_path):
    from pytorch_lightning.callbacks import EarlyStopping

    with applied_main_config(_cfg(tmp_path, early_stop_min_delta=1e-4, early_stop_patience=15)) as mc:
        es = _of(_callbacks(mc, _model()), EarlyStopping)
    assert len(es) == 1 and es[0].patience == 15
    assert abs(es[0].min_delta) == pytest.approx(1e-4)


def test_G4e_C3_schedule_free_optimizer_no_scheduler_no_swa(tmp_path):
    import schedulefree
    from pytorch_lightning.callbacks import StochasticWeightAveraging

    with applied_main_config(_cfg(tmp_path, schedule_free=True)) as mc:
        model = _model()
        conf = model.configure_optimizers()
        cbs = _callbacks(mc, model)
    opt = conf["optimizer"] if isinstance(conf, dict) else conf
    assert isinstance(opt, schedulefree.RAdamScheduleFree)
    assert not (isinstance(conf, dict) and conf.get("lr_scheduler") is not None)
    assert not _of(cbs, StochasticWeightAveraging)
    assert {float(g["lr"]) for g in opt.param_groups} == {pytest.approx(1e-3)}
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_G4e_C3_schedule_free_keeps_fine_tune_lr_multipliers(tmp_path):
    import schedulefree

    with applied_main_config(_cfg(tmp_path, schedule_free=False)):
        ref = _model(train_continue="finetune").configure_optimizers()["optimizer"]
    with applied_main_config(_cfg(tmp_path, schedule_free=True)):
        conf = _model(train_continue="finetune").configure_optimizers()
    opt = conf["optimizer"] if isinstance(conf, dict) else conf
    assert isinstance(opt, schedulefree.RAdamScheduleFree)
    got = sorted(float(g["lr"]) for g in opt.param_groups)
    want = sorted(float(g["lr"]) for g in ref.param_groups)
    assert len(want) > 1 and got == pytest.approx(want)


def test_G4e_C3_other_optimizer_target_raises(tmp_path):
    cfg = _cfg(tmp_path, schedule_free=True)
    cfg.train.optimizer._target_ = "torch.optim.AdamW"
    with applied_main_config(cfg), pytest.raises(ValueError):
        _model().configure_optimizers()


def test_G4e_C4_fit_switches_modes_and_saves_averaged_weights(tmp_path):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    from tomocpt.networks.inference import InferenceModel

    seen = {"train": set(), "val": set()}

    class _Spy(pl.Callback):
        @staticmethod
        def _mode(trainer):
            return [bool(g["train_mode"]) for g in trainer.optimizers[0].param_groups]

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            seen["train"].update(self._mode(trainer))

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            if trainer.optimizers:
                seen["val"].update(self._mode(trainer))

    items = [G3.v2_item(300 + i, [15.0, 30.0], apix=10.0) for i in range(4)]
    with applied_main_config(_cfg(tmp_path, schedule_free=True)):
        model = _model()
        ck = ModelCheckpoint(dirpath=tmp_path / "ck", monitor="val_loss", filename="weights", save_top_k=1)
        trainer = G3.cpu_trainer(tmp_path / "tb", callbacks=[ck, _Spy()], max_epochs=2, limit_train_batches=2,
                                 limit_val_batches=1, enable_checkpointing=True)
        trainer.fit(model, datamodule=G3.datamodule(items))
        assert seen["train"] == {True} and seen["val"] == {False}
        opt = trainer.optimizers[0]
        assert not any(g["train_mode"] for g in opt.param_groups), "fit must end in eval mode"
        eval_w = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt.train()
        train_w = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt.eval()
        assert any(not torch.equal(eval_w[k], train_w[k]) for k in eval_w if eval_w[k].is_floating_point())
        assert ck.best_model_path, "ModelCheckpoint saved nothing"
        # checkpoints are written after validation (eval mode): the final one equals eval weights
        trainer.save_checkpoint(tmp_path / "final.ckpt")
        final = torch.load(tmp_path / "final.ckpt", map_location="cpu", weights_only=False)["state_dict"]
        for k in eval_w:
            assert torch.equal(final[k], eval_w[k]), k
    with applied_main_config(None):
        m = InferenceModel.load_from_checkpoint(str(ck.best_model_path), map_location="cpu", strict=True)
    assert m is not None


def test_G4e_C5_missing_package_raises_import_error(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "schedulefree" or name.startswith("schedulefree."):
            raise ImportError("No module named 'schedulefree'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with applied_main_config(_cfg(tmp_path, schedule_free=True)):
        with pytest.raises(ImportError, match="schedulefree"):
            _model().configure_optimizers()
    with applied_main_config(_cfg(tmp_path)):
        conf = _model().configure_optimizers()
    assert type(conf["optimizer"]) is torch.optim.RAdam
