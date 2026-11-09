"""Gate 8 training-step micro-benchmark (spec 08_AGENT_PERFORMANCE.md, Section 3.3 and 8a).

Builds the model with ``training.helpers._create_model`` and runs Lightning ``fit`` exactly as
``tomocpt train`` does (precision, gradient clipping, weight averaging callback, TensorBoard image
logging), for ``warmup + steps`` optimiser steps, without validation, checkpoints or early stopping.
Phase times come from synchronised timers around the Lightning hooks:

- ``ms_forward``: start of ``training_step`` to its end, minus the loss;
- ``ms_loss``: ``multiclass_loss`` / ``loss`` (and the Gate 7 contrastive / Gate 4f offset terms);
- ``ms_backward``: ``on_before_backward`` → ``on_after_backward``;
- ``ms_optimizer``: the rest up to ``on_train_batch_end`` (clipping, optimiser step, EMA update);
- ``ms_data_wait``: ``on_train_batch_end`` of the previous step → ``on_train_batch_start``
  (the loader's next batch, its host→device copy).

The LR plateau scheduler is dropped (it acts only at epoch ends on ``val_loss_smooth``).

Usage::

    python -m benchmarks.perf.step --config recipe.yaml --data chunks --steps 300 --warmup 50 \\
        --set train.network.SwinUNETR.use_checkpoint=false --out before.json
"""
import argparse
import copy
import math
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from benchmarks.perf.common import (applied_config, config_hash, environment, git_commit, load_config,
                                    synthetic_class_table, write_json)

#: Distinct synthetic samples (cycled), so building the data costs nothing per step.
N_SYNTHETIC_SAMPLES = 8


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _SyntheticV2(torch.utils.data.Dataset):
    """Pipeline-v2 samples (Section 5.1 keys): noise input, soft spheres per class at random places."""

    def __init__(self, length: int, d_vox, seed: int, size: int = 64):
        from tomocpt.labels.helpers import render_targets

        self.length = int(length)
        g = np.random.default_rng(seed)
        C = len(d_vox)
        self.items = []
        for _ in range(N_SYNTHETIC_SAMPLES):
            n = 12
            pts = np.zeros((n, 5), np.float32)
            pts[:, :3] = g.uniform(0, size, (n, 3))
            pts[:, 3] = g.integers(0, C, n)
            pts[:, 4] = np.asarray(d_vox, np.float32)[pts[:, 3].astype(int)]
            self.items.append({
                "input_data": torch.from_numpy(g.standard_normal((1, size, size, size)).astype(np.float32)).clamp_(-3, 3),
                "target_data": torch.from_numpy(render_targets(pts, (size,) * 3, C, "clip")),
                "class_available": torch.ones(C, dtype=torch.bool),
                "class_d_vox": torch.as_tensor(d_vox, dtype=torch.float32),
                "apix_work": torch.tensor([10.0], dtype=torch.float32),
                "dataset_id": torch.tensor(0, dtype=torch.int64),
            })

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.items[i % N_SYNTHETIC_SAMPLES]


def _synthetic_datamodule(n_samples: int, d_vox, batch: int, seed: int):
    import pytorch_lightning as pl

    ds = _SyntheticV2(n_samples, d_vox, seed)

    class _Data(pl.LightningDataModule):
        def train_dataloader(self):
            return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    return _Data()


def _chunks_datamodule(cfg):
    from tomocpt.dataManager.dataloading import Data

    data = Data(
        data_dir=cfg.chunks_dir, return_labels=(cfg.mode == "picking"), batch_size=cfg.batch_size,
        workers_for_data=cfg.n_cpus_for_dataloading, augmentation=cfg.augmentation,
        pipeline_version=cfg.data_pipeline_version, scale_cfg=cfg.scale,
        chunk_sampling=cfg.chunk_sampling, repeat_factor_threshold=cfg.repeat_factor_threshold)
    data.setup()
    return data


class _Timer:
    """Synchronised phase timers (see the module docstring)."""

    def __init__(self, device, warmup: int):
        self.device, self.warmup = device, int(warmup)
        self.step = 0
        self.rows = []
        self.losses = []
        self._t = {}
        self._loss_ms = 0.0
        self._last_end = None
        self.t_first = self.t_last = None

    def mark(self, name):
        _sync(self.device)
        self._t[name] = time.perf_counter()

    def add_loss(self, ms):
        self._loss_ms += ms

    def batch_start(self):
        self.mark("start")
        self._wait = 0.0 if self._last_end is None else (self._t["start"] - self._last_end) * 1e3
        self._loss_ms = 0.0
        if self.step == self.warmup:
            self.t_first = self._last_end if self._last_end is not None else self._t["start"]
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

    def batch_end(self, loss):
        self.mark("end")
        t = self._t
        if self.step >= self.warmup:
            fwd_total = (t.get("step_end", t["start"]) - t["start"]) * 1e3
            self.rows.append({
                "forward": max(0.0, fwd_total - self._loss_ms), "loss": self._loss_ms,
                "backward": (t.get("bwd_end", t["end"]) - t.get("bwd_start", t["end"])) * 1e3,
                "optimizer": (t["end"] - t.get("bwd_end", t["end"])) * 1e3,
                "data_wait": self._wait,
            })
            self.losses.append(float(loss))
            self.t_last = t["end"]
        self._last_end = t["end"]
        self._t = {}
        self.step += 1


def _instrument(model, timer: _Timer):
    """Wrap ``training_step`` and the loss methods of one model instance with the timer."""
    orig_step = model.training_step

    def training_step(*a, **k):
        out = orig_step(*a, **k)
        timer.mark("step_end")
        return out

    model.training_step = training_step
    for name in ("multiclass_loss", "loss", "contrastive_loss", "_offset_term"):
        if not hasattr(model, name):
            continue
        fn = getattr(model, name)

        def wrapped(*a, _fn=fn, **k):
            _sync(timer.device)
            t0 = time.perf_counter()
            out = _fn(*a, **k)
            _sync(timer.device)
            timer.add_loss((time.perf_counter() - t0) * 1e3)
            return out

        setattr(model, name, wrapped)

    orig_conf = model.configure_optimizers

    def configure_optimizers():
        conf = orig_conf()
        return {"optimizer": conf["optimizer"]} if isinstance(conf, dict) else conf

    model.configure_optimizers = configure_optimizers


def _callback(timer: _Timer):
    import pytorch_lightning as pl

    class _TimerCallback(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            timer.batch_start()

        def on_before_backward(self, trainer, pl_module, loss):
            timer.mark("bwd_start")

        def on_after_backward(self, trainer, pl_module):
            timer.mark("bwd_end")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            timer.batch_end(loss.detach().float().item())

    return _TimerCallback()


def _summary(rows):
    out = {}
    for k in ("forward", "loss", "backward", "optimizer", "data_wait"):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"ms_{k}"] = float(v.mean()) if len(v) else 0.0
        out[f"ms_{k}_p90"] = float(np.percentile(v, 90)) if len(v) else 0.0
    return out


def run_step_benchmark(config=None, data: str = "synthetic", classes: int = 2, batch: int = 16,
                       steps: int = 300, warmup: int = 50, seed: int = 0, overrides: Iterable[str] = (),
                       device: str = "auto", logger: str = "tensorboard", workers: Optional[int] = None) -> dict:
    """Run the micro-benchmark; returns the result dict (Section 3.3 keys plus ``losses``).

    ``config``: ``MainConfig``, YAML path or ``None``. ``data``: ``synthetic`` (``classes`` particle
    classes, pipeline v2), ``chunks`` (the config's ``train.chunks_dir``) or ``chunks:<dir>``.
    ``device``: ``auto`` | ``cuda`` | ``cpu``. ``logger``: ``tensorboard`` (as training; panels are
    part of the step) or ``none``. ``workers`` overrides ``train.n_cpus_for_dataloading``."""
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    cfg = load_config(config, overrides)
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu")
    cfg.train.batch_size = int(batch)
    cfg.train.use_gpus = dev.type == "cuda"
    if workers is not None:
        cfg.train.n_cpus_for_dataloading = int(workers)
    d_vox = None
    if data == "synthetic":
        diam = synthetic_class_table(cfg, classes)
        d_vox = [d / 10.0 for d in diam]  # synthetic data at 10 Å/px
    elif data == "chunks" or data.startswith("chunks:"):
        if data.startswith("chunks:"):
            cfg.train.chunks_dir = Path(data.split(":", 1)[1])
    else:
        raise ValueError(f"data must be 'synthetic', 'chunks' or 'chunks:<dir>', got {data!r}")

    tmp = tempfile.TemporaryDirectory(prefix="tomocpt_perf_")
    cfg.train.model_dir = Path(tmp.name) / "model"
    total = int(warmup) + int(steps)
    result = {}
    try:
        with applied_config(cfg) as mc:
            tc = mc.train
            torch.set_float32_matmul_precision(tc.network.TORCH_MATMUL_PRECISION)
            pl.seed_everything(int(seed), workers=True)
            from tomocpt.training.helpers import _create_model, _setup_callbacks

            model, _, checkpointer = _create_model(tc)
            if data == "synthetic":
                dm = _synthetic_datamodule(total * int(batch), d_vox, int(batch), int(seed))
                n_batches = total
            else:
                dm = _chunks_datamodule(tc)
                n_batches = len(dm.train_dataloader())
            from pytorch_lightning.callbacks import StochasticWeightAveraging
            from tomocpt.training.averaging import WeightAveraging

            callbacks = [c for c in _setup_callbacks(tc, model, checkpointer)
                         if isinstance(c, (WeightAveraging, StochasticWeightAveraging))]
            timer = _Timer(dev, warmup)
            _instrument(model, timer)
            callbacks.append(_callback(timer))
            tb = TensorBoardLogger(save_dir=tmp.name, name="", version="") if logger == "tensorboard" else False
            trainer = pl.Trainer(
                accelerator="gpu" if dev.type == "cuda" else "cpu", devices=1, max_steps=total,
                max_epochs=max(1, math.ceil(total / max(1, n_batches))),
                limit_val_batches=0, num_sanity_val_steps=0, logger=tb, enable_checkpointing=False,
                enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks,
                precision=tc.network.TORCH_FLOAT_PRECISION.value, gradient_clip_val=1.0,
                gradient_clip_algorithm="norm", deterministic=False)
            trainer.fit(model, datamodule=dm)
            timed = len(timer.rows)
            wall = (timer.t_last - timer.t_first) if timed else float("nan")
            result.update(_summary(timer.rows))
            result.update({
                "steps_per_s": float(timed / wall) if timed and wall > 0 else 0.0,
                "ms_step": float(wall / timed * 1e3) if timed else 0.0,
                "peak_mem_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else 0,
                "losses": timer.losses,
                "timed_steps": timed,
                "commit": git_commit(),
                "config_hash": config_hash(mc),
                "settings": {"data": data, "classes": int(classes) if data == "synthetic" else len(model.class_table()),
                             "batch": int(batch), "steps": int(steps), "warmup": int(warmup), "seed": int(seed),
                             "device": str(dev), "overrides": list(overrides or ()), "logger": logger,
                             "workers": int(tc.n_cpus_for_dataloading), "batches_per_epoch": int(n_batches)},
                "env": environment(),
            })
    finally:
        tmp.cleanup()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="recipe YAML (as tomocpt train --config_file)")
    p.add_argument("--data", default="synthetic", help="synthetic | chunks | chunks:<dir>")
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--logger", default="tensorboard", choices=("tensorboard", "none"))
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    p.add_argument("--profile", action="store_true", help="also write the module / kernel profile")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = run_step_benchmark(a.config, a.data, a.classes, a.batch, a.steps, a.warmup, a.seed,
                             overrides=a.overrides, device=a.device, logger=a.logger, workers=a.workers)
    if a.profile:
        from benchmarks.perf.profile_model import profile_network

        res["profile"] = profile_network(a.config, a.data, a.classes, a.batch, a.seed, a.overrides, a.device)
    write_json(a.out, res)
    keys = ("steps_per_s", "ms_step", "ms_forward", "ms_loss", "ms_backward", "ms_optimizer", "ms_data_wait")
    print({k: round(res[k], 3) for k in keys}, f"peak {res['peak_mem_bytes'] / 1e9:.2f} GB")
    return res


if __name__ == "__main__":
    main()
