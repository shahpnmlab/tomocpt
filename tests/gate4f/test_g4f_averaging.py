"""G4f-C2..C4: weight averaging callback."""

import pytest
import torch

from gate4f import g4f_helpers as H
from utils.tiny import applied_main_config


def _avg_callback(mc, model):
    from pytorch_lightning.callbacks import ModelCheckpoint

    from tomocpt.training.averaging import WeightAveraging
    from tomocpt.training.helpers import _setup_callbacks

    cbs = _setup_callbacks(mc.train, model, ModelCheckpoint(monitor="val_loss"))
    found = [c for c in cbs if isinstance(c, WeightAveraging)]
    return found[0] if found else None, cbs


def _model():
    from tomocpt.networks.pickingModel import BasePickingModel

    torch.manual_seed(0)
    return BasePickingModel(lr=1e-3, train_continue=None)


def _fit(tmp_path, model, callbacks, epochs=3, batches=3):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    ck = ModelCheckpoint(dirpath=tmp_path / "ck", monitor="val_loss", filename="weights", save_last=True)
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=TensorBoardLogger(str(tmp_path / "tb")),
                         callbacks=[*callbacks, ck], max_epochs=epochs, limit_train_batches=batches,
                         limit_val_batches=1, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0, precision="32")
    from gate3 import g3_helpers as G3
    trainer.fit(model, datamodule=G3.datamodule(H.items(6), batch_size=2))
    return trainer, ck


def test_G4f_C2_ema_decay_schedule():
    from tomocpt.training.averaging import ema_decay_at

    assert ema_decay_at(0, 0.999) == pytest.approx(0.1)
    assert ema_decay_at(10, 0.999) == pytest.approx(11 / 20)
    assert ema_decay_at(10 ** 6, 0.999) == pytest.approx(0.999)


def test_G4f_C2_ema_fit(tmp_path):
    import pytorch_lightning as pl

    from tomocpt.training.averaging import ema_decay_at

    KEY = "model.out.conv.conv.weight"
    rec = {"post_step": [], "train_seen": [], "val_seen": [], "ema_at_val": []}

    with applied_main_config(H.v2_cfg(tmp_path, weight_averaging="ema", ema_decay=0.9)) as mc:
        model = _model()
        avg, _ = _avg_callback(mc, model)
        assert avg is not None

        class _Spy(pl.Callback):
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                rec["train_seen"].append(pl_module.state_dict()[KEY].clone())

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                rec["post_step"].append(pl_module.state_dict()[KEY].clone())

            def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                rec["val_seen"].append(pl_module.state_dict()[KEY].clone())
                rec["ema_at_val"].append(avg.averaged[KEY].clone())

        trainer, ck = _fit(tmp_path, model, [avg, _Spy()])

    # hand recursion over the post-step weights (the first average is the first post-step weight
    # blended into the initial weights with d_0)
    ema = None
    init = rec["train_seen"][0]
    ema = init.clone()
    for t, w in enumerate(rec["post_step"]):
        d = ema_decay_at(t, 0.9)
        ema = d * ema + (1 - d) * w
    torch.testing.assert_close(avg.averaged[KEY], ema, rtol=0, atol=1e-6)
    assert avg.n_updates == len(rec["post_step"])
    # validation saw the average, training the raw weights
    for v, e in zip(rec["val_seen"], rec["ema_at_val"]):
        torch.testing.assert_close(v, e, rtol=0, atol=0)
    for i in range(1, len(rec["train_seen"])):
        torch.testing.assert_close(rec["train_seen"][i], rec["post_step"][i - 1], rtol=0, atol=0)
    # end of fit and the last checkpoint hold the average
    torch.testing.assert_close(model.state_dict()[KEY], avg.averaged[KEY], rtol=0, atol=0)
    last = torch.load(ck.last_model_path, map_location="cpu", weights_only=False)
    torch.testing.assert_close(last["state_dict"][KEY], rec["ema_at_val"][-1], rtol=0, atol=0)

    # state round trip
    from tomocpt.training.averaging import WeightAveraging

    other = WeightAveraging("ema", ema_decay=0.9)
    other.load_state_dict(avg.state_dict())
    assert other.n_updates == avg.n_updates
    torch.testing.assert_close(other.averaged[KEY], avg.averaged[KEY])


@pytest.mark.parametrize("seq, patience, rel, start", [
    ([1.0, 0.9, 0.8, 0.7999, 0.7998, 0.7997], 2, 1e-3, 4),   # improvements below 0.1 % from index 3
    ([1.0, 0.5, 0.25, 0.125, 0.0625], 1, 1e-3, None),         # always improving: never
    ([1.0, 1.0], 0, 1e-3, 0),                                 # patience 0: the first validation
    ([10.0, 9.0, 8.995, 8.993, 8.0], 2, 1e-3, 3),             # < 0.1 % below the best twice; stays on after 8.0
])
def test_G4f_C3_plateau_trigger(seq, patience, rel, start):
    from tomocpt.training.averaging import PlateauTrigger

    trig = PlateauTrigger(patience=patience, rel_delta=rel)
    fired = [trig.step(v) for v in seq]
    if start is None:
        assert not any(fired)
    else:
        assert fired.index(True) == start and all(fired[start:])


def test_G4f_C3_swa_plateau_mean_of_epoch_end_weights(tmp_path):
    import pytorch_lightning as pl

    KEY = "model.out.conv.conv.weight"
    epoch_end = []
    with applied_main_config(H.v2_cfg(tmp_path, weight_averaging="swa_plateau", swa_plateau_patience=0)) as mc:
        model = _model()
        avg, cbs = _avg_callback(mc, model)
        assert avg is not None
        from pytorch_lightning.callbacks import StochasticWeightAveraging
        assert not any(isinstance(c, StochasticWeightAveraging) for c in cbs)

        class _Spy(pl.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if batch_idx == trainer.num_training_batches - 1:
                    epoch_end.append(pl_module.state_dict()[KEY].clone())

        trainer, _ = _fit(tmp_path, model, [_Spy(), avg], epochs=4)
    # trigger fires at the first validation (patience 0); epochs 1..3 are averaged
    assert avg.active and avg.n_updates == 3
    want = torch.stack(epoch_end[1:]).mean(0)
    torch.testing.assert_close(avg.averaged[KEY], want, rtol=0, atol=1e-6)
    torch.testing.assert_close(model.state_dict()[KEY], avg.averaged[KEY], rtol=0, atol=0)


def test_G4f_C4_defaults_and_schedule_free_combinations(tmp_path):
    from pytorch_lightning.callbacks import StochasticWeightAveraging

    from tomocpt.defaultConfigs.train_config import TrainConfig

    t = TrainConfig()
    assert t.weight_averaging == "swa" and t.ema_decay == 0.999
    assert t.swa_plateau_patience == 5 and t.swa_plateau_rel_delta == 1e-3
    with applied_main_config(H.v2_cfg(tmp_path)) as mc:
        avg, cbs = _avg_callback(mc, _model())
        assert avg is None and sum(isinstance(c, StochasticWeightAveraging) for c in cbs) == 1
    with applied_main_config(H.v2_cfg(tmp_path, weight_averaging="none")) as mc:
        avg, cbs = _avg_callback(mc, _model())
        assert avg is None and not any(isinstance(c, StochasticWeightAveraging) for c in cbs)
    for mode in ("ema", "swa_plateau"):
        with applied_main_config(H.v2_cfg(tmp_path, weight_averaging=mode, schedule_free=True)) as mc:
            with pytest.raises(ValueError):
                _avg_callback(mc, _model())
    with applied_main_config(H.v2_cfg(tmp_path, weight_averaging="bogus")) as mc:
        with pytest.raises(ValueError):
            _avg_callback(mc, _model())
