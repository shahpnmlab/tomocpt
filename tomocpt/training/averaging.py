"""Weight averaging without a fixed start epoch (Gate 4f, TODO 3b.3a).

``WeightAveraging("ema")`` keeps an exponential moving average of the weights from the first step;
``WeightAveraging("swa_plateau")`` starts an equal-weight average of the epoch-end weights once
``val_loss_smooth`` stops improving by a relative margin (:class:`PlateauTrigger`). In both modes
every validation, and so ``val_loss``, ``val_loss_smooth``, early stopping and ``ModelCheckpoint``,
runs on the averaged weights: they are swapped into the model at the start of validation and the
raw weights are swapped back before the next training batch. At the end of fit the model holds
the average. (Lightning's ``StochasticWeightAveraging`` only transfers its average when training
reaches ``max_epochs``, after the last validation, so it never reaches a checkpoint when early
stopping fires.)
"""
from typing import Dict, Optional

import pytorch_lightning as pl
import torch

MODES = ("ema", "swa_plateau")


def ema_decay_at(t: int, max_decay: float) -> float:
    """Decay of the ``t``-th EMA update (0-based): ``min(max_decay, (1 + t) / (10 + t))``."""
    return min(float(max_decay), (1.0 + t) / (10.0 + t))


class PlateauTrigger:
    """Fires (and stays on) once ``patience`` consecutive values fail to improve on the best by
    more than ``rel_delta * |best|``."""

    def __init__(self, patience: int, rel_delta: float):
        self.patience, self.rel_delta = int(patience), float(rel_delta)
        self.best: Optional[float] = None
        self.since = 0
        self.fired = False

    def step(self, value: float) -> bool:
        value = float(value)
        if self.best is None or value < self.best - self.rel_delta * abs(self.best):
            self.best = value if self.best is None else min(self.best, value)
            self.since = 0
        else:
            self.since += 1
        if self.since >= self.patience:
            self.fired = True
        return self.fired

    def state_dict(self) -> dict:
        return {"best": self.best, "since": self.since, "fired": self.fired}

    def load_state_dict(self, s: dict) -> None:
        self.best, self.since, self.fired = s["best"], int(s["since"]), bool(s["fired"])


#: State entries that are not weights (the Gate 7 contrastive memory bank is a queue of embeddings).
_NOT_AVERAGED = ("contrastive_bank.",)


def _float_state(module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()
            if v.is_floating_point() and not k.startswith(_NOT_AVERAGED)}


class WeightAveraging(pl.Callback):
    def __init__(self, mode: str, ema_decay: float = 0.999, plateau_patience: int = 5,
                 plateau_rel_delta: float = 1e-3):
        if mode not in MODES:
            raise ValueError(f"WeightAveraging mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.ema_decay = float(ema_decay)
        self.trigger = PlateauTrigger(plateau_patience, plateau_rel_delta)
        self.averaged: Optional[Dict[str, torch.Tensor]] = None
        self.n_updates = 0
        self._raw: Optional[Dict[str, torch.Tensor]] = None  # raw weights while the average is swapped in

    @property
    def active(self) -> bool:
        return self.averaged is not None if self.mode == "ema" else self.trigger.fired

    # -- weights in the model -------------------------------------------------------------
    @staticmethod
    def _load(module, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            own = module.state_dict()
            for k, v in state.items():
                own[k].copy_(v)

    def _swap_in(self, module) -> None:
        if self.averaged is None or self._raw is not None:
            return
        self._raw = _float_state(module)
        self._load(module, self.averaged)

    def _swap_out(self, module) -> None:
        if self._raw is None:
            return
        self._load(module, self._raw)
        self._raw = None

    # -- hooks ------------------------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        if self.mode == "ema" and self.averaged is None:
            self.averaged = _float_state(pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._swap_out(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.mode != "ema":
            return
        d = ema_decay_at(self.n_updates, self.ema_decay)
        with torch.no_grad():
            for k, v in pl_module.state_dict().items():
                if k in self.averaged:
                    a = self.averaged[k]
                    if a.device != v.device:
                        a = self.averaged[k] = a.to(v.device)
                    a.mul_(d).add_(v.detach().to(a.dtype), alpha=1.0 - d)
        self.n_updates += 1

    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self.mode == "swa_plateau" and self.trigger.fired and self._raw is None:
            # the raw weights now are the end-of-epoch weights
            w = _float_state(pl_module)
            if self.averaged is None:
                self.averaged, self.n_updates = w, 1
            else:
                self.n_updates += 1
                with torch.no_grad():
                    for k, v in w.items():
                        self.averaged[k].add_((v - self.averaged[k]) / self.n_updates)
        self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.mode != "swa_plateau" or self.trigger.fired:
            return
        v = trainer.callback_metrics.get("val_loss_smooth")
        if v is not None:
            self.trigger.step(float(v))

    def on_fit_end(self, trainer, pl_module):
        # end of fit: the model holds the average
        if self.averaged is not None and self._raw is None:
            self._swap_in(pl_module)
        self._raw = None

    # -- resume -----------------------------------------------------------------------------
    def state_dict(self) -> dict:
        cpu = (lambda s: None if s is None else {k: v.detach().cpu() for k, v in s.items()})
        return {"mode": self.mode, "averaged": cpu(self.averaged), "raw": cpu(self._raw),
                "n_updates": self.n_updates, "trigger": self.trigger.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.averaged = state.get("averaged")
        self._raw = state.get("raw")
        self.n_updates = int(state.get("n_updates", 0))
        if state.get("trigger"):
            self.trigger.load_state_dict(state["trigger"])
