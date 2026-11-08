"""Volume snapshots for the training dashboard (``scripts/train_dashboard.py``).

With ``train.snapshot_chunks = n > 0`` (pipeline v2), the same ``n`` chunks per class and split
(train, val) are predicted at the end of every validation, with the weights validation used
(EMA or SWA when on), and written next to the TensorBoard events:

- ``<log_dir>/snapshots/chunks.npz``: once; ``input`` uint8 ``[K, D, H, W]`` (min-max per chunk),
  ``target`` uint8 ``[K, C, D, H, W]`` (0..1 -> 0..255), ``split``, ``cls`` (the class it was chosen for) and ``name`` per chunk, ``classes``;
- ``<log_dir>/snapshots/epoch_XXXX.npz``: ``pred`` uint8 ``[K, C, D, H, W]``.

The training chunks are read without augmentation. Off by default: nothing is created.
"""
import copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

SNAPSHOT_DIR_NAME = "snapshots"


def pick_chunks(class_sets: List[set], n_classes: int, per_class: int) -> List[Tuple[int, int]]:
    """``(chunk index, class)`` pairs: per class, ``per_class`` chunks spread evenly over the chunks
    that contain it (deterministic), without repeating a chunk."""
    chosen, seen = [], set()
    for c in range(n_classes):
        idx = [i for i, s in enumerate(class_sets) if c in s and i not in seen]
        if not idx:
            continue
        k = min(per_class, len(idx))
        for j in np.linspace(0, len(idx) - 1, k).round().astype(int):
            chosen.append((idx[j], c))
            seen.add(idx[j])
    return chosen


def _to_u8(a: torch.Tensor) -> np.ndarray:
    return (a.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


class VolumeSnapshots(pl.Callback):
    def __init__(self, per_class: int):
        self.per_class = int(per_class)
        self.batch = None
        self.out_dir = None

    def _items(self, dataset, split: str, class_names: List[str]):
        ds = copy.copy(dataset)  # no augmentation; the loaders keep the original
        ds.transform = None
        ds.zoom_aug_max = 1.0
        items, labels = [], []
        for i, c in pick_chunks(ds.chunk_classes(), len(class_names), self.per_class):
            items.append(ds[i])
            vol_path = Path(ds.filepath_tuples[i][0])
            labels.append((split, c, f"{split} {class_names[c]} {vol_path.parent.parent.name}/{vol_path.name}"))
        return items, labels

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        dm = trainer.datamodule
        if dm is None or getattr(dm, "class_table", None) is None:  # setup() runs before on_fit_start
            return
        names = [row.name for row in dm.class_table]
        items, labels = [], []
        for split, ds in (("train", dm.dataset_training), ("val", dm.dataset_val)):
            it, lb = self._items(ds, split, names)
            items += it
            labels += lb
        if not items:
            return
        self.batch = {k: torch.stack([it[k] for it in items]) for k in
                      ("input_data", "target_data", "class_available", "class_d_vox", "apix_work")}
        log_dir = getattr(trainer.logger, "log_dir", None) or trainer.default_root_dir
        self.out_dir = Path(log_dir) / SNAPSHOT_DIR_NAME
        self.out_dir.mkdir(parents=True, exist_ok=True)
        x = self.batch["input_data"][:, 0]
        lo = x.amin(dim=(1, 2, 3), keepdim=True)
        hi = x.amax(dim=(1, 2, 3), keepdim=True)
        np.savez_compressed(
            self.out_dir / "chunks.npz",
            input=_to_u8((x - lo) / (hi - lo).clamp_min(1e-6)),
            target=_to_u8(self.batch["target_data"]),
            split=np.array([s for s, _, _ in labels]), cls=np.array([c for _, c, _ in labels]),
            name=np.array([n for _, _, n in labels]),
            classes=np.array(names), apix=self.batch["apix_work"][:, 0].numpy())

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_rank != 0 or self.batch is None:
            return
        dev = pl_module.device
        batch = {k: v.to(dev) for k, v in self.batch.items()}
        preds = []
        for s in range(0, batch["input_data"].shape[0], 4):
            part = {k: v[s:s + 4] for k, v in batch.items()}
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = pl_module(part["input_data"], apix=pl_module._batch_apix(part))
            y_pred, _ = pl_module._probabilities(logits.float(), part["target_data"], part)
            preds.append(_to_u8(y_pred))
        np.savez_compressed(self.out_dir / f"epoch_{trainer.current_epoch:04d}.npz",
                            pred=np.concatenate(preds))
