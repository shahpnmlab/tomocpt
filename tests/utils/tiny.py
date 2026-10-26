"""Tiny, CPU-sized model/config/data helpers for tests (written at Gate 1, frozen afterwards).

* :func:`tiny_swin_kwargs` - constructor kwargs for a tiny (``feature_size=12``) SwinUNETR.
* :func:`tiny_main_config` - a fresh ``MainConfig`` with tiny network settings, CPU, fp32,
  ``batch_size=2``, ``n_cpus_for_dataloading=0`` and every path under ``tmp_path``.
* :func:`random_chunk_datamodule` - a LightningDataModule of random 64^3 chunks with the
  v1 batch keys, or the CLAUDE.md Section 5.1 keys when ``v2=True``.
* :func:`applied_main_config` - context manager: copy a config into the global
  ``tomocpt.mainConfig.mainConfig`` (legacy models read it) and restore the global config,
  and the ``tomocpt.constants`` module, on exit.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["tiny_swin_kwargs", "tiny_main_config", "random_chunk_datamodule", "applied_main_config"]

CHUNK = 64


def tiny_swin_kwargs() -> dict:
    return dict(img_size=64, in_channels=1, out_channels=1, feature_size=12, use_v2=True,
                use_checkpoint=False, drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0)


def tiny_main_config(tmp_path):
    """A ``MainConfig`` for CPU tests. Does not touch the global ``mainConfig``."""
    from tomocpt.defaultConfigs.network_config import PrecisionType
    from tomocpt.mainConfig import MainConfig

    tmp_path = Path(tmp_path)
    cfg = MainConfig()
    swin = cfg.train.network.SwinUNETR
    for k, v in tiny_swin_kwargs().items():
        if k != "img_size" and hasattr(swin, k):
            setattr(swin, k, v)
    cfg.train.network.TORCH_FLOAT_PRECISION = PrecisionType.float
    cfg.train.batch_size = 2
    cfg.train.n_epochs = 1
    cfg.train.n_cpus_for_dataloading = 0
    cfg.train.n_cpus_for_preprocessing = 1
    cfg.train.N_CPUS_IF_NO_GPU = 1
    cfg.train.use_gpus = False
    cfg.train.USE_CUDA_FOR_DATA = False
    cfg.train.model_dir = tmp_path / "model"
    cfg.train.chunks_dir = tmp_path / "chunks"
    cfg.train.training_data_dir = tmp_path / "training_data"
    cfg.prepData.training_data_dir = tmp_path / "training_data"
    cfg.prepData.USE_CUDA_FOR_DATA = False
    cfg.shared.training_data_dir = tmp_path / "training_data"
    cfg.shared.chunks_dir = tmp_path / "chunks"
    cfg.infer.predictions_dir = tmp_path / "predictions"
    cfg.infer.use_cuda = False
    cfg.infer.N_CPUS_IF_NO_GPU = 1
    return cfg


def _assign_inplace(target, source):
    """Copy every field of ``source`` onto ``target`` recursively, keeping nested object
    identity (other modules hold references to the nested sections)."""
    for f in fields(type(source)):
        if not hasattr(source, f.name):
            continue
        sv = getattr(source, f.name)
        tv = getattr(target, f.name, None)
        if is_dataclass(sv) and is_dataclass(tv) and type(sv) is type(tv):
            _assign_inplace(tv, sv)
        else:
            setattr(target, f.name, copy.deepcopy(sv))


@contextlib.contextmanager
def applied_main_config(cfg=None):
    """Apply ``cfg`` (e.g. :func:`tiny_main_config`) to the global ``mainConfig`` for the
    duration of the block; always restore the previous global config and constants."""
    from tomocpt import constants
    from tomocpt.mainConfig import mainConfig

    saved = copy.deepcopy(mainConfig)
    saved_constants = {k: copy.deepcopy(v) for k, v in vars(constants).items() if not k.startswith("__")}
    try:
        if cfg is not None:
            _assign_inplace(mainConfig, cfg)
        yield mainConfig
    finally:
        _assign_inplace(mainConfig, saved)
        for k in [k for k in vars(constants) if not k.startswith("__") and k not in saved_constants]:
            delattr(constants, k)
        for k, v in saved_constants.items():
            setattr(constants, k, v)


def _soft_blobs(g: torch.Generator, n_blobs: int, sigma: float, size: int = CHUNK) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
    out = torch.zeros(size, size, size)
    for _ in range(n_blobs):
        c = torch.randint(8, size - 8, (3,), generator=g).float()
        r2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        out = torch.maximum(out, torch.exp(-r2 / (2 * sigma ** 2)))
    return out


class _RandomChunks(Dataset):
    def __init__(self, n: int, n_classes: int, v2: bool, seed: int):
        self.n, self.n_classes, self.v2, self.seed = n, n_classes, v2, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        x = torch.randn(1, CHUNK, CHUNK, CHUNK, generator=g).clamp_(-3, 3)
        y = torch.stack([_soft_blobs(g, 3, 3.0 + c) for c in range(self.n_classes)])
        item = {"input_data": x, "target_data": y}
        if self.v2:
            item["class_available"] = torch.ones(self.n_classes, dtype=torch.bool)
            item["class_d_vox"] = torch.tensor([20.0 + 4.0 * c for c in range(self.n_classes)], dtype=torch.float32)
            item["apix_work"] = torch.tensor([10.0], dtype=torch.float32)
            item["dataset_id"] = torch.tensor(0, dtype=torch.int64)
        return item


def random_chunk_datamodule(n: int = 2, n_classes: int = 1, v2: bool = False, batch_size: int = 2,
                            seed: int = 0):
    """LightningDataModule with ``n`` random chunks for train and for val (same data).

    v1 keys: ``input_data [B,1,64,64,64]``, ``target_data [B,C,64,64,64]`` (soft blobs in [0,1];
    ``C = n_classes``, 1 for v1). ``v2=True`` adds ``class_available [B,C] bool`` (all True),
    ``class_d_vox [B,C] float32``, ``apix_work [B,1] float32`` (10.0), ``dataset_id [B] int64``.
    """
    import pytorch_lightning as pl

    class _RandomData(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(_RandomChunks(n, n_classes, v2, seed), batch_size=batch_size, num_workers=0)

        def val_dataloader(self):
            return DataLoader(_RandomChunks(n, n_classes, v2, seed), batch_size=batch_size, num_workers=0)

    return _RandomData()
