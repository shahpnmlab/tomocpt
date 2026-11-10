"""Worker for ``test_g8c1_ddp_gloo.py`` (G8c-C8): a two-epoch on-the-fly fit on 1 or 2 gloo CPU
ranks from a tomogram store built by the test.

    python ddp_gloo_otf_worker.py <store_dir> <out_dir> <devices>

Each rank writes the crops its sampler yielded per epoch to ``<out_dir>/crops_rank<r>.json``;
rank 0 writes ``fit_done.txt`` (epoch, global step) and one checkpoint in ``<out_dir>/ckpt``.
"""
import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
sys.path.insert(1, str(TESTS.parent))


def main(store_dir, out_dir, devices):
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint

    from gate2 import g2_helpers as G2
    from gate6 import g6_helpers as H6
    from utils.tiny import applied_main_config

    from tomocpt.dataManager import otf
    from tomocpt.defaultConfigs.otf_config import OnTheFlyConfig
    from tomocpt.defaultConfigs.scale_config import ScaleConfig

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = int(devices)
    cfg = H6.v2_config_with_filaments(out_dir / "cfg", ((1, "a", 120.0), (2, "b", 200.0)), ())
    seen = {}
    orig_iter = otf.CropSampler.__iter__

    def _iter(self):
        rows = list(orig_iter(self))
        if self.fixed is None:
            seen.setdefault(str(self.epoch), []).extend([list(r) for r in rows])
        return iter(rows)

    otf.CropSampler.__iter__ = _iter
    with applied_main_config(cfg):
        from tomocpt.networks.pickingModel import BasePickingModel

        torch.manual_seed(0)
        data = otf.OnTheFlyData(str(store_dir), True, batch_size=2, workers_for_data=0,
                                augmentation=G2.no_spatial_aug(), pipeline_version=2, scale_cfg=ScaleConfig(),
                                otf=OnTheFlyConfig(samples_per_epoch=8, val_samples=4))
        data.setup()
        model = BasePickingModel(lr=1e-3, train_continue=None)
        trainer = pl.Trainer(accelerator="cpu", devices=devices, strategy="ddp" if devices > 1 else "auto",
                             precision="32", logger=False, enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=0, max_epochs=2, limit_val_batches=1,
                             callbacks=[ModelCheckpoint(dirpath=str(out_dir / "ckpt"), save_top_k=1, monitor=None)],
                             use_distributed_sampler=True)
        trainer.fit(model, datamodule=data)
        (out_dir / f"crops_rank{trainer.global_rank}.json").write_text(json.dumps(seen))
        if trainer.is_global_zero:
            (out_dir / "fit_done.txt").write_text(f"{trainer.current_epoch} {trainer.global_step}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
