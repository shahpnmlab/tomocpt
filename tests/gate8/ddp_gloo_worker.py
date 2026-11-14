"""Worker for ``test_g8_ddp_gloo.py`` (spec 3.6): run as a script, so Lightning's DDP launcher can
start the second rank by re-running this file.

    python ddp_gloo_worker.py grads <out_dir> <devices> [option=value ...]
    python ddp_gloo_worker.py fit   <out_dir> <devices> [option=value ...]

``grads``: one training step on 4 fixed v2 chunks (2 classes, all available, no dropout); rank 0
saves the gradients after backward (all-reduced under DDP) to ``<out_dir>/grads_<devices>.pt``.
``fit``: two epochs (2 train + 1 val batch each), one checkpoint in ``<out_dir>/ckpt``.
Options are ``section.field=value`` overrides on ``train`` (e.g. ``particles.hessian_block=0``).
"""
import ast
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
sys.path.insert(1, str(TESTS.parent))


def _set(obj, dotted, value):
    *path, last = dotted.split(".")
    for p in path:
        obj = getattr(obj, p)
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    setattr(obj, last, value)


def main(mode, out_dir, devices, options):
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader

    from gate6 import g6_helpers as H6
    from utils.tiny import _RandomChunks, applied_main_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = int(devices)
    cfg = H6.v2_config_with_filaments(out_dir / "cfg", ((1, "a", 120.0), (2, "b", 200.0)), ())
    for o in options:
        k, v = o.split("=", 1)
        _set(cfg.train, k, v)
    n_items, n_batches = (4, 1) if mode == "grads" else (8, 2)
    per_rank = n_items // devices // n_batches

    class _D(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(_RandomChunks(n_items, 2, True, seed=3), batch_size=per_rank, shuffle=False)

        def val_dataloader(self):
            return DataLoader(_RandomChunks(2 * devices, 2, True, seed=4), batch_size=2, shuffle=False)

    grads = {}

    class _Grab(pl.Callback):
        def on_after_backward(self, trainer, pl_module):
            if not grads:
                grads.update({n: p.grad.detach().clone() for n, p in pl_module.named_parameters()
                              if p.grad is not None})
                grads["__none__"] = [n for n, p in pl_module.named_parameters() if p.requires_grad and p.grad is None]

    with applied_main_config(cfg):
        from tomocpt.networks.pickingModel import BasePickingModel

        torch.manual_seed(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        callbacks = [_Grab()]
        kw = dict(max_epochs=1, limit_train_batches=1, limit_val_batches=1, enable_checkpointing=False)
        if mode == "fit":
            from pytorch_lightning.callbacks import ModelCheckpoint

            callbacks.append(ModelCheckpoint(dirpath=str(out_dir / "ckpt"), save_top_k=1, monitor=None))
            kw = dict(max_epochs=2, limit_train_batches=n_batches, limit_val_batches=1, enable_checkpointing=True)
        trainer = pl.Trainer(accelerator="cpu", devices=devices, strategy="ddp" if devices > 1 else "auto",
                             precision="32", logger=False, enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=0, callbacks=callbacks, use_distributed_sampler=True, **kw)
        trainer.fit(model, datamodule=_D())
        if trainer.is_global_zero:
            if mode == "grads":
                torch.save(grads, out_dir / f"grads_{devices}.pt")
            else:
                (out_dir / "fit_done.txt").write_text(f"{trainer.current_epoch} {trainer.global_step}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
