"""Helpers for the Gate 5 tests (no tests here).

New public names (spec ``05_AGENT_SSL.md``, "Public names") are imported inside the functions,
so the v1 lock still imports on code that does not have them yet.

Interface assumptions (see ``README.md``): arrays are ``(Z, Y, X)``; functions of
``tomocpt.networks.ssl.views`` accept numpy float32 volumes and may return numpy arrays or torch
tensors (with or without leading singleton axes); ``rng`` is a ``numpy.random.Generator``.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.tiny import applied_main_config, tiny_main_config
from utils.wedge import apply_missing_wedge, wedge_power_fraction

HERE = Path(__file__).resolve().parent
CHUNK = 64
APIX_REF = 10.0


def load_common():
    """``g5_common.py`` (shared with the golden generator), loaded by path."""
    spec = importlib.util.spec_from_file_location("_gate5_common", HERE / "g5_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
# array helpers
# --------------------------------------------------------------------------------------

def to_np(x) -> np.ndarray:
    """numpy float64 copy of a volume, leading singleton axes removed (3-D result)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    return x


def to_scalar(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().double().reshape(-1)[0])
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


def points_np(p) -> np.ndarray:
    """``[n, 3]`` float64 points."""
    if isinstance(p, torch.Tensor):
        p = p.detach().double().cpu().numpy()
    p = np.asarray(p, dtype=np.float64)
    return p.reshape(-1, 3)


def wedged_noise(shape, seed=0, max_angle_deg=60.0, std=1.0) -> np.ndarray:
    """Periodic white noise with a missing wedge (tilt axis Y), scaled to ``std``. float32."""
    v = apply_missing_wedge(np.random.default_rng(seed).standard_normal(shape), max_angle_deg)
    return (v * (std / v.std())).astype(np.float32)


def view_wedge_fraction(view) -> float:
    """Missing-wedge power fraction of a **crop** (a view cut out of a larger wedged volume).

    A crop is not periodic and its spectrum is the source spectrum convolved with the crop
    window, so ``utils.wedge.wedge_power_fraction`` of a plain 64^3 crop of perfectly wedged
    noise is already ~0.02 (measured: 0.016-0.025) whatever the view code does. This helper
    removes that measurement floor: demean, Tukey(0.5) window, ``kmin = 0.1``, and count only
    power beyond 70 deg (a 10 deg margin around the 60 deg wedge). Measured on correct views of
    wedged noise: ~0.0015; after a 90 deg rotation about Z: ~0.16.
    """
    from scipy.signal.windows import tukey

    v = to_np(view)
    w = [tukey(n, 0.5, sym=False) for n in v.shape]
    win = w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]
    return wedge_power_fraction((v - v.mean()) * win, max_angle_deg=70.0, kmin=0.1)


def shell_noise_std(v, lo=0.8, hi=0.9) -> float:
    """Noise std estimated from the Fourier shell ``[lo, hi]`` x Nyquist (Parseval, white noise)."""
    v = to_np(v)
    F = np.fft.fftn(v - v.mean())
    k = [np.fft.fftfreq(n) for n in v.shape]
    r = np.sqrt(k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2) / 0.5
    shell = (r >= lo) & (r <= hi)
    return float(np.sqrt(np.mean(np.abs(F[shell]) ** 2) / v.size))


def gaussian_blob_volume(shape, centre_zyx, sigma, amplitude=1.0, noise=0.01, seed=0) -> np.ndarray:
    """A Gaussian blob on a faint white-noise background (the background keeps robust
    normalisations well defined). float32."""
    zz, yy, xx = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    c = centre_zyx
    r2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
    v = amplitude * np.exp(-r2 / (2.0 * sigma ** 2))
    v = v + noise * np.random.default_rng(seed).standard_normal(shape)
    return v.astype(np.float32)


def blob_centroid(v, radius=4) -> np.ndarray:
    """Sub-voxel centre ``(z, y, x)`` of the brightest blob: half-maximum thresholded centroid
    in a cube of ``radius`` around the maximum."""
    v = to_np(v)
    m = np.array(np.unravel_index(np.argmax(v), v.shape))
    lo = np.maximum(m - radius, 0)
    hi = np.minimum(m + radius + 1, v.shape)
    sub = v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    w = np.clip(sub - 0.5 * sub.max(), 0, None)
    grids = np.meshgrid(*[np.arange(l, h, dtype=np.float64) for l, h in zip(lo, hi)], indexing="ij")
    return np.array([(g * w).sum() / w.sum() for g in grids])


def flip_axes_of(out, inp):
    """The tuple of axes ``a`` with ``out == np.flip(inp, a)`` exactly, or ``None``."""
    out, inp = to_np(out), to_np(inp)
    for mask in range(8):
        axes = tuple(a for a in range(3) if mask >> a & 1)
        ref = np.flip(inp, axes) if axes else inp
        if out.shape == ref.shape and np.array_equal(out, ref):
            return axes
    return None


# --------------------------------------------------------------------------------------
# SSL config
# --------------------------------------------------------------------------------------

def ssl_cfg(**kw):
    from tomocpt.defaultConfigs.ssl_config import SSLConfig

    return SSLConfig(**kw)


def resolved(cfg):
    """The SSL config with its ``None`` defaults resolved: ``cfg.resolve()`` (or ``resolved()``)
    if the class has one (returning a config, or ``None`` after resolving in place); otherwise
    ``cfg`` itself (resolved at construction)."""
    for name in ("resolve", "resolved"):
        fn = getattr(cfg, name, None)
        if callable(fn):
            out = fn()
            return cfg if out is None else out
    return cfg


def ssl_v2_main_config(tmp_path, **ssl_kw):
    """Tiny CPU ``MainConfig`` for SSL v2: ``train.mode = selfSupervised``, ``train.ssl.version = 2``."""
    from tomocpt.defaultConfigs.train_config import TrainingModes

    cfg = tiny_main_config(tmp_path)
    cfg.train.mode = TrainingModes.selfSupervised
    cfg.train.ssl = ssl_cfg(version=2, **ssl_kw)
    return cfg


def make_source(seed, size=128, n_blobs=6) -> np.ndarray:
    """A 128^3 SSL source chunk at ``APIX_REF``: wedged noise plus dark blobs of a few sizes,
    robust-normalised. float32."""
    rng = np.random.default_rng(seed)
    shape = (size,) * 3
    zz, yy, xx = np.meshgrid(*[np.arange(size, dtype=np.float32)] * 3, indexing="ij")
    v = np.zeros(shape, np.float32)
    for _ in range(n_blobs):
        c = rng.uniform(16, size - 16, 3)
        r = rng.uniform(3, 12)
        d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
        v -= np.clip(r + 0.5 - d, 0, 1)
    v = apply_missing_wedge(v, 60.0) + 0.5 * wedged_noise(shape, seed=seed + 1000)
    return robust_normalise(v)


def robust_normalise(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    med = np.median(v)
    p5, p95 = np.percentile(v, [5, 95])
    return np.clip((v - med) / (p95 - p5), -3, 3).astype(np.float32)


def training_pair(source, cfg, seed):
    """``make_training_pair`` with ``numpy.random`` and ``rng`` both seeded by ``seed``."""
    from tomocpt.networks.ssl.views import make_training_pair

    np.random.seed(seed)
    return make_training_pair(source=source, apix_ref=APIX_REF, ssl_cfg=cfg,
                              rng=np.random.default_rng(seed))


PAIR_KEYS = ("input_a", "target_a", "mask_a", "input_b", "target_b", "mask_b", "target_log2",
             "pts_a", "pts_b", "apix_a", "apix_b")


# --------------------------------------------------------------------------------------
# one SSL v2 training step (CPU)
# --------------------------------------------------------------------------------------

class ListData(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def datamodule(train_items, val_items, batch_size=2, num_workers=0):
    import pytorch_lightning as pl

    class _D(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(ListData(train_items), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

        def val_dataloader(self):
            return DataLoader(ListData(val_items), batch_size=batch_size, shuffle=False, num_workers=0)

    return _D()


def head_classes():
    from tomocpt.networks.ssl.heads import DenseProjectionHead, ProjectionHead, RelativeScaleHead

    return ProjectionHead, DenseProjectionHead, RelativeScaleHead


def head_state_keys(model) -> set:
    """State-dict keys (of ``model``) that belong to an SSL v2 head module."""
    heads = head_classes()
    keys = set()
    for name, mod in model.named_modules():
        if isinstance(mod, heads):
            keys.update(f"{name}.{k}" if name else k for k in mod.state_dict())
    return keys


def fit_ssl_v2_one_step(root: Path, **ssl_kw) -> dict:
    """Build a tiny SSL v2 ``SelfSupervisedModel`` (global config = tiny CPU config with
    ``train.ssl = SSLConfig(version=2, **ssl_kw)``), fit one training batch and one validation
    batch of ``make_training_pair`` items (``default_collate``), save a checkpoint.

    Returns ``dict(path, grads={name: grad or None}, trainable=[names], metrics=set of logged
    names, model, head_keys, cfg)``.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = ssl_v2_main_config(root, **ssl_kw)
    grads = {}

    class _Grab(pl.Callback):
        def on_after_backward(self, trainer, pl_module):
            if not grads:
                for n, p in pl_module.named_parameters():
                    if p.requires_grad:
                        grads[n] = None if p.grad is None else p.grad.detach().clone()

    with applied_main_config(cfg) as g:
        ssl = g.train.ssl
        items = [training_pair(make_source(200 + i), ssl, 300 + i) for i in range(4)]
        torch.manual_seed(0)
        model = SelfSupervisedModel(lr=1e-3, train_continue=None)
        trainer = pl.Trainer(accelerator="cpu", devices=1, logger=TensorBoardLogger(str(root / "tb")),
                             callbacks=[_Grab()], enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0, precision="32",
                             max_epochs=1, limit_train_batches=1, limit_val_batches=1)
        trainer.fit(model, datamodule=datamodule(items[:2], items[2:]))
        path = root / "ssl_v2.ckpt"
        trainer.save_checkpoint(path)
        metrics = set(trainer.callback_metrics.keys()) | set(trainer.logged_metrics.keys())
        head_keys = head_state_keys(model)
    shutil.rmtree(root / "tb", ignore_errors=True)
    return dict(path=path, grads=grads, metrics=metrics, model=model, head_keys=head_keys, cfg=cfg)


# --------------------------------------------------------------------------------------
# network features (for the learning tests)
# --------------------------------------------------------------------------------------

def encoder_pooled(net, x) -> torch.Tensor:
    """``p = concat(avgpool(hidden[3]), avgpool(hidden[4]))`` of the Swin encoder (unconditioned)."""
    hs = net.swinViT(x, net.normalize)
    return torch.cat([hs[3].mean(dim=(2, 3, 4)), hs[4].mean(dim=(2, 3, 4))], dim=1)


def decoder2_features(net, x, apix) -> torch.Tensor:
    """The (FiLM-conditioned) ``decoder2`` output of a ``ScaleConditionedSwinUNETR`` at half the
    input resolution, computed with its public submodules as in its ``forward``."""
    film = net.cond.film
    e = net.cond.scale_embed(apix)
    hs = net.swinViT(x, net.normalize)
    enc1 = net.encoder2(hs[0])
    enc2 = net.encoder3(hs[1])
    enc3 = net.encoder4(hs[2])
    dec4 = film[0](net.encoder10(hs[4]), e)
    dec3 = film[1](net.decoder5(dec4, hs[3]), e)
    dec2 = film[2](net.decoder4(dec3, enc3), e)
    dec1 = film[3](net.decoder3(dec2, enc2), e)
    return film[4](net.decoder2(dec1, enc1), e)


def log_uniform(rng, lo, hi) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
