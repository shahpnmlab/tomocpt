"""Helpers for the Gate 3 tests (no tests here).

New public names (spec ``03_AGENT_MULTICLASS_PICKER.md``, "Public names") are imported inside
the functions, so modules that also hold v1 locks still import on code without them.
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.tiny import applied_main_config, tiny_main_config

TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = TESTS_DIR / "golden"
CHUNK = 64

#: (number, name, diameter in A)
CLASSES_AB = ((1, "A", 150.0), (2, "B", 300.0))
CLASSES_ABC = ((1, "A", 150.0), (2, "B", 300.0), (3, "C", 200.0))


def load_common():
    """``tests/golden/gate3_common.py`` (shared with the golden generator), loaded by path."""
    spec = importlib.util.spec_from_file_location("_gate3_common", GOLDEN_DIR / "gate3_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------------------

def v2_config(tmp_path, classes: Sequence[tuple] = CLASSES_AB, cond: bool = False,
              class_weights: Optional[str] = None):
    """Tiny CPU ``MainConfig`` for pipeline v2: ``multiclass`` with the given classes
    (``(number, name, diameter_ang)``), ``train.network.scale_conditioning = cond``."""
    cfg = tiny_main_config(tmp_path)
    cfg.train.data_pipeline_version = 2
    cfg.prepData.multiclass = True
    cfg.prepData.class_diameters = ",".join(f"{n}:{d}" for n, _, d in classes)
    cfg.prepData.class_names = ",".join(f"{n}:{name}" for n, name, _ in classes)
    cfg.train.network.scale_conditioning = cond
    if class_weights is not None:
        cfg.train.particles.class_weights = class_weights
    return cfg


def n_channels(classes) -> int:
    return len(classes)


# --------------------------------------------------------------------------------------
# batches
# --------------------------------------------------------------------------------------

def render(points, n_classes):
    from tomocpt.labels.helpers import render_targets

    return render_targets(np.asarray(points, dtype=np.float64).reshape(-1, 5), (CHUNK,) * 3, n_classes, "clip")


def v2_item(seed: int, d_vox: Sequence[float], apix: float = 10.0,
            available: Optional[Sequence[bool]] = None, n_per_class: int = 3) -> Dict[str, torch.Tensor]:
    """One pipeline-v2 sample (CLAUDE.md Section 5.1 keys, no batch dimension): noise input,
    ``n_per_class`` particles per class drawn with ``render_targets``."""
    rng = np.random.default_rng(seed)
    C = len(d_vox)
    pts = []
    for c, d in enumerate(d_vox):
        for _ in range(n_per_class):
            z, y, x = rng.uniform(d / 2 + 2, CHUNK - d / 2 - 2, size=3)
            pts.append([z, y, x, c, d])
    x = np.clip(rng.standard_normal((1, CHUNK, CHUNK, CHUNK)), -3, 3).astype(np.float32)
    avail = [True] * C if available is None else list(available)
    return {
        "input_data": torch.from_numpy(x),
        "target_data": torch.from_numpy(render(pts, C)),
        "class_available": torch.tensor(avail, dtype=torch.bool),
        "class_d_vox": torch.tensor(list(d_vox), dtype=torch.float32),
        "apix_work": torch.tensor([apix], dtype=torch.float32),
        "dataset_id": torch.tensor(0, dtype=torch.int64),
    }


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def v2_batch(d_vox, B=2, seed=0, apix=10.0, available=None):
    """``available``: None, a per-class list (same for all samples) or a ``[B][C]`` list."""
    items = []
    for b in range(B):
        av = available[b] if (available is not None and isinstance(available[0], (list, tuple))) else available
        a = apix[b] if isinstance(apix, (list, tuple)) else apix
        items.append(v2_item(seed * 1000 + b, d_vox, apix=a, available=av))
    return collate(items)


class ListData(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def datamodule(items, batch_size=2):
    import pytorch_lightning as pl

    class _D(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(ListData(items), batch_size=batch_size, shuffle=False, num_workers=0)

        def val_dataloader(self):
            return DataLoader(ListData(items), batch_size=batch_size, shuffle=False, num_workers=0)

    return _D()


def cpu_trainer(tmp_dir, callbacks=(), **kw):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    kw.setdefault("max_epochs", 1)
    kw.setdefault("limit_train_batches", 1)
    kw.setdefault("limit_val_batches", 1)
    return pl.Trainer(accelerator="cpu", devices=1, logger=TensorBoardLogger(str(tmp_dir)),
                      callbacks=list(callbacks), enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0, precision="32", **kw)


def apix_tensor(apix: float, B: int = 1) -> torch.Tensor:
    return torch.full((B, 1), float(apix), dtype=torch.float32)


def fixed_input(B=1, seed=11) -> torch.Tensor:
    x = np.random.RandomState(seed).standard_normal((B, 1, CHUNK, CHUNK, CHUNK)).astype(np.float32)
    return torch.from_numpy(x)


# --------------------------------------------------------------------------------------
# v2 checkpoints written by the code under test
# --------------------------------------------------------------------------------------

def build_v2_checkpoint(root: Path, classes, cond: bool) -> dict:
    """Build a tiny pipeline-v2 ``BasePickingModel``, fit one batch on CPU, save it.

    Returns ``dict(path, x, logits={apix or None: logits}, classes)``.
    """
    from tomocpt.networks.pickingModel import BasePickingModel

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = v2_config(root, classes, cond=cond)
    C = len(classes)
    d_vox = [d / 10.0 for _, _, d in classes]
    items = [v2_item(100 + i, d_vox, apix=10.0) for i in range(2)]
    with applied_main_config(cfg):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        trainer = cpu_trainer(root / "tb")
        trainer.fit(model, datamodule=datamodule(items))
        path = root / "v2.ckpt"
        trainer.save_checkpoint(path)
        x = fixed_input(1, seed=21)
        model.eval()
        logits = {}
        with torch.no_grad():
            if cond:
                for a in (10.0, 20.0):
                    logits[a] = model(x, apix=apix_tensor(a)).clone()
            else:
                logits[None] = model(x).clone()
    assert logits[next(iter(logits))].shape[1] == C
    return dict(path=path, x=x, logits=logits, classes=tuple(classes))


def state_dict_of(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def first_step_loss_and_grads(model, items, tmp_dir, batch_size=2):
    """Fit one batch; return ``(loss, {name: grad or None})`` captured right after backward."""
    import pytorch_lightning as pl

    out = {}

    class _Grab(pl.Callback):
        def on_after_backward(self, trainer, pl_module):
            if "grads" not in out:
                out["grads"] = {n: (None if p.grad is None else p.grad.detach().clone())
                                for n, p in pl_module.named_parameters()}

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            v = outputs["loss"] if isinstance(outputs, dict) else outputs
            out.setdefault("loss", float(v.detach().double()))

    trainer = cpu_trainer(tmp_dir, callbacks=[_Grab()])
    trainer.fit(model, datamodule=datamodule(items, batch_size=batch_size))
    return out["loss"], out["grads"], trainer


def param_groups(model):
    """``[(lr_mult, [params])]`` from ``BaseModel._get_model_parameters_for_optimizer``.
    A plain parameter list (training from scratch) is one group with multiplier 1.0."""
    params = list(model._get_model_parameters_for_optimizer())
    if params and isinstance(params[0], dict):
        return [(float(g.get("lr_mult", 1.0)), list(g["params"])) for g in params]
    return [(1.0, params)]
