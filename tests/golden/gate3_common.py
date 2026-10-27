"""Inputs and helpers shared by ``make_goldens_gate3.py`` and ``tests/gate3``.

No ``tomocpt`` import, so the generator running in the ``v2-gate-0-base`` worktree and the
tests running on HEAD build bit-identical inputs and run the same fit. Loaded by file path.

* :func:`v1_batch` - the fixed pipeline-v1 batch of the G3-C1 golden: ``input_data``
  ``[2,1,64,64,64]`` (``RandomState(31)`` normal, clipped to [-3, 3]) and ``target_data``
  ``[2,1,64,64,64]`` (the Gaussian spheres of the G1-C2 golden).
* :func:`reinit_` - overwrite every parameter of a module with values drawn from a seeded
  ``np.random.RandomState`` in sorted-name order, so the weights do not depend on how much
  torch RNG the constructor consumed.
* :func:`first_training_step_loss` - fit one batch with a Lightning trainer (CPU, fp32, a
  throwaway ``TensorBoardLogger``: legacy ``training_step`` writes images to
  ``self.logger.experiment``) and return the value ``training_step`` returned.
"""

import importlib.util
import shutil
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHUNK = 64
REINIT_SEED = 3


def _gate1_common():
    spec = importlib.util.spec_from_file_location("_gate1_common_for_g3", HERE / "gate1_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def v1_batch():
    """``{"input_data", "target_data"}`` float32 numpy arrays, both ``[2,1,64,64,64]``."""
    with np.load(HERE / "gate1_spheres.npz") as z:
        spheres = {int(k[1:]): z[k] for k in z.files}
    _, y_true = _gate1_common().loss_inputs(spheres)
    x = np.clip(np.random.RandomState(31).standard_normal((2, 1, CHUNK, CHUNK, CHUNK)), -3, 3)
    return {"input_data": x.astype(np.float32), "target_data": y_true.astype(np.float32)}


def reinit_(module, seed=REINIT_SEED):
    """Deterministic weights: matrices/kernels ~ N(0, 1/fan_in), 1-D ``*weight`` ~ 1 + 0.05 N,
    other 1-D parameters ~ 0.02 N. Parameters are visited in sorted-name order."""
    import torch

    rs = np.random.RandomState(seed)
    with torch.no_grad():
        for name, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
            shape = tuple(p.shape)
            if p.ndim >= 2:
                fan_in = int(np.prod(shape[1:])) or 1
                v = rs.standard_normal(shape) / np.sqrt(fan_in)
            elif name.endswith("weight"):
                v = 1.0 + 0.05 * rs.standard_normal(shape)
            else:
                v = 0.02 * rs.standard_normal(shape)
            p.copy_(torch.from_numpy(v.astype(np.float32)))
    return module


def first_training_step_loss(model, batch_np):
    """Run ``trainer.fit`` for one training batch (and one validation batch, which the
    ``val_loss_smooth`` scheduler monitor needs) and return the first ``training_step`` loss."""
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader, Dataset

    class _One(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {k: torch.from_numpy(v[i]) for k, v in batch_np.items()}

    class _Data(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(_One(), batch_size=2, shuffle=False, num_workers=0)

        def val_dataloader(self):
            return DataLoader(_One(), batch_size=2, shuffle=False, num_workers=0)

    losses = []

    class _Grab(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            out = outputs["loss"] if isinstance(outputs, dict) else outputs
            losses.append(float(out.detach().double()))

    tb = tempfile.mkdtemp(prefix="g3_tb_")
    try:
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=TensorBoardLogger(tb),
                             limit_train_batches=1, limit_val_batches=1, callbacks=[_Grab()],
                             enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0, precision="32")
        trainer.fit(model, datamodule=_Data())
    finally:
        shutil.rmtree(tb, ignore_errors=True)
    assert len(losses) == 1, losses
    return losses[0]
