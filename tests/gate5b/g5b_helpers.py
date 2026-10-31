"""Helpers for the Gate 5b tests (no tests here).

New public names (spec ``05b_AGENT_SSL_FIX.md``, "Public names") are imported inside the
functions, so collection works on code that does not have them yet. Gate 5 helpers
(``tests/gate5/g5_helpers.py``, frozen) are reused.
"""

from __future__ import annotations

import functools
import shutil
from pathlib import Path

import numpy as np
import torch

from gate5 import g5_helpers as H
from utils.tiny import applied_main_config


# --------------------------------------------------------------------------------------
# Fourier bands and correlation
# --------------------------------------------------------------------------------------

def radius_grid(shape) -> np.ndarray:
    """``|k|`` as a fraction of Nyquist (1.0 = Nyquist along an axis)."""
    k = [np.fft.fftfreq(n) for n in shape]
    return np.sqrt(k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2) / 0.5


def band(v, lo=None, hi=None) -> np.ndarray:
    """Demeaned ``v`` with only the Fourier coefficients with ``lo <= |k|/Nyquist <= hi`` kept."""
    v = H.to_np(v)
    F = np.fft.fftn(v - v.mean())
    r = radius_grid(v.shape)
    if lo is not None:
        F[r < lo] = 0
    if hi is not None:
        F[r > hi] = 0
    return np.fft.ifftn(F).real


def corr(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def flip_mask_axes(m: int) -> tuple:
    return tuple(a for a in range(3) if m >> a & 1)


def best_flip(ref, other) -> tuple:
    """Axes ``f`` maximising ``corr(ref, flip(other, f))``, and that correlation."""
    ref, other = H.to_np(ref), H.to_np(other)
    scores = [(corr(ref, np.flip(other, flip_mask_axes(m))), flip_mask_axes(m)) for m in range(8)]
    r, axes = max(scores)
    return axes, r


# --------------------------------------------------------------------------------------
# short SSL v2 fits (CPU)
# --------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _source(seed):
    return H.make_source(seed)


def queue_buffers(model):
    """``[(name, buffer)]``: floating 2-D buffers of ``model`` whose name contains ``queue``."""
    return [(n, b) for n, b in model.named_buffers()
            if "queue" in n.lower() and b.is_floating_point() and b.ndim == 2]


def queue_rows(buf: torch.Tensor, dim: int) -> torch.Tensor:
    """The queue as ``[Q, dim]`` rows (a ``[dim, Q]`` buffer, as in MoCo, is transposed)."""
    if buf.shape[1] == dim:
        return buf
    if buf.shape[0] == dim:
        return buf.t()
    raise AssertionError(f"queue buffer of shape {tuple(buf.shape)} has no axis of the projection size {dim}")


def fit_ssl_v2(root: Path, n_steps: int = 1, **ssl_kw) -> dict:
    """Tiny SSL v2 ``SelfSupervisedModel`` (global config = tiny CPU config with
    ``train.ssl = SSLConfig(version=2, **ssl_kw)``), ``n_steps`` training batches of 2
    ``make_training_pair`` items (distinct sources), then one validation batch (eval mode; the
    projection hook records training-mode calls only).

    Returns ``dict(model, cfg, grads=[{name: grad or None} per step],
    projections=[[ProjectionHead outputs (detached, float32)] per step])``.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
    from tomocpt.networks.ssl.heads import ProjectionHead

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = H.ssl_v2_main_config(root, **ssl_kw)
    grads, projections = [], []

    class _Grab(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            projections.append([])

        def on_after_backward(self, trainer, pl_module):
            grads.append({n: None if p.grad is None else p.grad.detach().clone()
                          for n, p in pl_module.named_parameters() if p.requires_grad})

    def _hook(mod, inp, out):
        if mod.training and projections:
            projections[-1].append(out.detach().float().clone())

    with applied_main_config(cfg) as g:
        ssl = g.train.ssl
        items = [H.training_pair(_source(200 + i), ssl, 300 + i) for i in range(2 * n_steps + 2)]
        torch.manual_seed(0)
        model = SelfSupervisedModel(lr=1e-3, train_continue=None)
        for m in model.modules():
            if isinstance(m, ProjectionHead):
                m.register_forward_hook(_hook)
        trainer = pl.Trainer(accelerator="cpu", devices=1, logger=TensorBoardLogger(str(root / "tb")),
                             callbacks=[_Grab()], enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0, precision="32",
                             max_epochs=1, limit_train_batches=n_steps, limit_val_batches=1)
        trainer.fit(model, datamodule=H.datamodule(items[:-2], items[-2:]))
    shutil.rmtree(root / "tb", ignore_errors=True)
    return dict(model=model, cfg=cfg, grads=grads, projections=projections)
