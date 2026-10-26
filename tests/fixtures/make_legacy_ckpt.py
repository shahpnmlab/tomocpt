"""Build a tiny legacy checkpoint with the code of ``v2-gate-0-base``.

Run by ``tests/utils/legacy.py`` in a worktree of the tag, with ``PYTHONPATH=<worktree>``::

    python make_legacy_ckpt.py --kind picking|ssl --out X.ckpt --expect-root <worktree>

Writes ``X.ckpt`` (fit for one epoch on random tensors on CPU, so the pickled ``MainConfig``
is the legacy class and the optimizer state is real) and ``X.ref.npz`` with a fixed input
``x`` and the legacy model's eval-mode outputs on it (``logits``, ``hidden4``).
"""

import argparse
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

CHUNK = 64
TINY = dict(in_channels=1, out_channels=1, feature_size=12, use_v2=True, use_checkpoint=False,
            drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["picking", "ssl"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-root", required=True)
    args = ap.parse_args()

    import tomocpt

    root = Path(args.expect_root).resolve()
    got = Path(tomocpt.__file__).resolve()
    if root not in got.parents:
        sys.exit(f"tomocpt imported from {got}, expected under {root}")

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader, Dataset

    from tomocpt.defaultConfigs.network_config import PrecisionType
    from tomocpt.defaultConfigs.train_config import TrainingModes
    from tomocpt.mainConfig import mainConfig

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    for k, v in TINY.items():
        setattr(mainConfig.train.network.SwinUNETR, k, v)
    mainConfig.train.network.TORCH_FLOAT_PRECISION = PrecisionType.float
    mainConfig.train.batch_size = 2
    mainConfig.train.use_gpus = False
    mainConfig.train.n_cpus_for_dataloading = 0
    mainConfig.train.mode = TrainingModes.picking if args.kind == "picking" else TrainingModes.selfSupervised

    class Chunks(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"input_data": torch.randn(1, CHUNK, CHUNK, CHUNK, generator=g).clamp(-3, 3),
                    "target_data": (torch.rand(1, CHUNK, CHUNK, CHUNK, generator=g) > 0.99).float()}

    class Data(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(Chunks(), batch_size=2)

        def val_dataloader(self):
            return DataLoader(Chunks(), batch_size=2)

    if args.kind == "picking":
        from tomocpt.networks.pickingModel import BasePickingModel
        model = BasePickingModel(lr=1e-3, train_continue=None)
    else:
        from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
        model = SelfSupervisedModel(lr=1e-3, train_continue=None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tb = tempfile.mkdtemp(prefix="tb_", dir=out.parent)
    try:
        # A logger is needed: the legacy training_step writes images to logger.experiment.
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=TensorBoardLogger(tb),
                             enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(model, datamodule=Data())
        trainer.save_checkpoint(out)
    finally:
        shutil.rmtree(tb, ignore_errors=True)

    x = np.random.RandomState(7).standard_normal((1, 1, CHUNK, CHUNK, CHUNK)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        logits, hidden4 = model.model(torch.from_numpy(x))
    np.savez(out.with_suffix(".ref.npz"), x=x, logits=logits.numpy(), hidden4=hidden4.numpy(),
             tomocpt_file=str(got))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
