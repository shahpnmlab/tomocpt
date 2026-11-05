"""Gate 4f test helpers (no tests). Interface fixed by the TA (spec 04f):

* ``tomocpt.training.averaging.WeightAveraging(mode, ema_decay=0.999, plateau_patience=5,
  plateau_rel_delta=1e-3)``: Lightning callback; attributes ``averaged`` (``{name: tensor}`` over the
  LightningModule's ``state_dict`` floating entries, or None before averaging starts), ``n_updates``,
  ``active`` (bool); ``state_dict()`` / ``load_state_dict()``.
* ``tomocpt.training.averaging.ema_decay_at(t, max_decay) = min(max_decay, (1 + t) / (10 + t))``.
* ``tomocpt.training.averaging.PlateauTrigger(patience, rel_delta)``: ``step(value) -> bool`` (True
  from the validation where averaging starts on).
* ``tomocpt.training.offsets.offset_targets(y, class_d_vox, available, particle_channels=None) ->
  (target [B, 3, D, H, W], mask [B, 1, D, H, W] bool)``.
* ``BasePickingModel.offset_head`` (None when off), ``predict_step_with_offsets(x, apix=None,
  classes=None) -> (probs, offsets)``, ``has_offset_head()``; ``EnsembleModel`` has the same two.
* ``predict_class_maps(..., offsets_out=None)``; ``peaks_from_class_maps(..., offsets=None)``.
"""

from pathlib import Path

import numpy as np
import torch

from gate3 import g3_helpers as G3

CLASSES = ((1, "A", 150.0), (2, "B", 300.0))


def v2_cfg(tmp_path, **train):
    cfg = G3.v2_config(tmp_path, CLASSES, cond=False)
    for k, v in train.items():
        if k.startswith("particles."):
            setattr(cfg.train.particles, k.split(".", 1)[1], v)
        else:
            setattr(cfg.train, k, v)
    return cfg


def items(n=4, seed0=400):
    return [G3.v2_item(seed0 + i, [15.0, 30.0], apix=10.0) for i in range(n)]


def float_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items() if v.is_floating_point()}


def gaussian_map(shape_zyx, centre_zyx, sigma):
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=float) for s in shape_zyx], indexing="ij")
    c = np.asarray(centre_zyx, float)
    return np.exp(-((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / (2 * sigma ** 2)).astype(np.float32)
