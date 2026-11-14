"""Swin gradient checkpointing off when memory allows (Gate 8, human decision 14 Nov 2026).

``network.SwinUNETR.use_checkpoint = True`` (the default) now means "checkpoint the Swin blocks
unless the training GPU has room to keep their activations". The first training step runs with
checkpointing on; :class:`SwinCheckpointAuto` then reads the step's peak memory, adds the
estimated cost of keeping the Swin activations, and turns checkpointing off for the rest of the
fit when the total fits in ``budget`` of the GPU. ``use_checkpoint = False`` is never changed.

Checkpointing does not change the numbers (G8b-C2: logits, loss and gradients bit-identical), so
switching after the first step keeps the run exact. Under DDP every rank takes the decision of the
largest peak (all ranks switch together). No config field, no CLI option.

Cost of keeping the Swin activations, measured at ``feature_size = 36``, 64³ crops, ``bf16-mixed``
(``benchmarks/perf/RESULTS_8b4.md``): CZII B=16 9.9 -> 17.4 GB, filaments B=16 7.4 -> 14.9 GB, so
7.5 GB / 16 = 0.47 GB per sample. It scales with ``feature_size`` and the crop volume, and doubles
at 32-bit precision.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from tomocpt.logger import get_logger

logging = get_logger()

#: GB of Swin activations kept per sample when checkpointing is off (fs 36, 64³, bf16-mixed)
GB_PER_SAMPLE = 0.47
#: fraction of the GPU memory the run may use without checkpointing
BUDGET = 0.85
#: safety factor on the activation estimate
MARGIN = 1.2


def swin_modules(model: torch.nn.Module):
    """Modules of ``model`` with a MONAI ``use_checkpoint`` switch (Swin blocks and layers)."""
    return [m for m in model.modules() if isinstance(getattr(m, "use_checkpoint", None), bool)
            and type(m).__module__.startswith("monai.")]


def set_swin_checkpoint(model: torch.nn.Module, on: bool) -> int:
    """Sets ``use_checkpoint`` on every Swin module; returns how many were set."""
    mods = swin_modules(model)
    for m in mods:
        m.use_checkpoint = bool(on)
    return len(mods)


def extra_gb(batch_size: int, feature_size: int, chunk_size: int, precision: str) -> float:
    """Estimated GB the Swin activations add when checkpointing is off."""
    scale = (feature_size / 36.0) * (chunk_size / 64.0) ** 3
    bytes_factor = 1.0 if "16" in str(precision) else 2.0
    return GB_PER_SAMPLE * batch_size * scale * bytes_factor * MARGIN


def fits_without_checkpoint(peak_gb: float, extra: float, total_gb: float, budget: float = BUDGET) -> bool:
    return peak_gb + extra <= budget * total_gb


class SwinCheckpointAuto(pl.Callback):
    """Turns Swin checkpointing off after the first training step when memory allows."""

    def __init__(self, feature_size: int, chunk_size: int, precision: str, batch_size: int):
        self.batch_size = int(batch_size)
        self.feature_size = int(feature_size)
        self.chunk_size = int(chunk_size)
        self.precision = str(precision)
        self.decided = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.decided:
            return
        self.decided = True
        dev = pl_module.device
        mods = swin_modules(pl_module)
        if dev.type != "cuda" or not any(m.use_checkpoint for m in mods):  # explicit use_checkpoint=False
            return
        x = batch.get("input_data") if isinstance(batch, dict) else None
        bsz = int(x.shape[0]) if torch.is_tensor(x) else self.batch_size
        peak = torch.tensor(torch.cuda.max_memory_reserved(dev) / 1e9, device=dev)
        peak = float(trainer.strategy.reduce(peak, reduce_op="max"))
        total = torch.cuda.get_device_properties(dev).total_memory / 1e9
        extra = extra_gb(bsz, self.feature_size, self.chunk_size, self.precision)
        if fits_without_checkpoint(peak, extra, total):
            n = set_swin_checkpoint(pl_module, False)
            logging.info(f"Swin checkpointing off ({n} modules): peak {peak:.1f} GB + {extra:.1f} GB "
                         f"<= {BUDGET:.0%} of {total:.1f} GB")
        else:
            logging.info(f"Swin checkpointing kept on: peak {peak:.1f} GB + {extra:.1f} GB > "
                         f"{BUDGET:.0%} of {total:.1f} GB")
