"""Inputs and helpers shared by ``make_golden_g5_c1.py`` and ``test_g5_v1_lock.py``.

No ``tomocpt`` import at module level, so the generator running in the ``v2-gate-0-base``
worktree and the test running on HEAD build bit-identical inputs. Loaded by file path.

* :func:`ssl_batch_input` - the fixed SSL v1 input ``[2,1,64,64,64]`` (``RandomState(51)``
  normal, clipped to [-3, 3]).
* :func:`reinit_by_name_` - overwrite every parameter with values from a ``RandomState``
  seeded by the parameter name (crc32), so the weights depend neither on the torch RNG
  consumed by the constructors nor on the presence of other parameters.
* :func:`configure_tiny_ssl` - set the tiny SSL v1 settings on a (global) ``MainConfig``.
* :func:`ssl_v1_step` - seeded ``SelfSupervisedModel._step``: ``[loss, rot, contrastive, recon]``.
* :func:`ssl_v1_forward_sums` - sums of the three eval-mode outputs (weights cross-check).
"""

import zlib

import numpy as np

CHUNK = 64
STEP_SEED = 5
TINY = dict(in_channels=1, out_channels=1, feature_size=12, use_v2=True, use_checkpoint=False,
            drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0)


def ssl_batch_input():
    x = np.clip(np.random.RandomState(51).standard_normal((2, 1, CHUNK, CHUNK, CHUNK)), -3, 3)
    return x.astype(np.float32)


def configure_tiny_ssl(cfg):
    """Tiny SSL v1 settings (CPU, fp32, batch 2) on ``cfg`` (a ``MainConfig``)."""
    from tomocpt.defaultConfigs.network_config import PrecisionType
    from tomocpt.defaultConfigs.train_config import TrainingModes

    for k, v in TINY.items():
        setattr(cfg.train.network.SwinUNETR, k, v)
    cfg.train.network.TORCH_FLOAT_PRECISION = PrecisionType.float
    cfg.train.batch_size = 2
    cfg.train.use_gpus = False
    cfg.train.n_cpus_for_dataloading = 0
    cfg.train.mode = TrainingModes.selfSupervised
    return cfg


def reinit_by_name_(module, offset=7):
    """Matrices/kernels ~ N(0, 1/fan_in), 1-D ``*weight`` ~ 1 + 0.05 N, other 1-D ~ 0.02 N."""
    import torch

    with torch.no_grad():
        for name, p in module.named_parameters():
            rs = np.random.RandomState((zlib.crc32(name.encode()) + offset) % (2 ** 32))
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


def ssl_v1_step(model, x_np, seed=STEP_SEED):
    """``[total, rot, contrastive, recon]`` of ``model._step`` with numpy and torch seeded."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    model.train()
    with torch.no_grad():
        loss, tasks = model._step({"input_data": torch.from_numpy(x_np.copy())}, 0)[:2]
    return [float(loss.double())] + [float(t.double()) for t in tasks]


def ssl_v1_forward_sums(model, x_np):
    import torch

    model.eval()
    with torch.no_grad():
        outs = model(torch.from_numpy(x_np.copy()))
    model.train()
    return [float(o.double().sum()) for o in outs]
