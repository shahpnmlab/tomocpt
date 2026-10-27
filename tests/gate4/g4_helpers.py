"""Helpers for the Gate 4 tests (no tests here).

New public names (spec ``04_AGENT_INFER_BENCHMARK.md``, "Public names") are imported inside
the test functions, so the v1 locks still run on code that does not have them yet.

* :func:`run_predict` - ``tomocpt.predict.run.predict()`` with the global ``mainConfig.infer``
  set from a dict, on the CPU, with the Dask cluster forced in-process (one worker, one
  thread), optionally with an oracle network swapped into every loaded ``InferenceModel``.
* Oracles (CPU, no parameters) that replace ``InferenceModel.model``:
  :class:`NegInputOracle` (spec G4-C4: ``logit(clip(-x_norm, eps, 1 - eps))`` per channel) and
  :class:`LevelTagOracle` (constant output that encodes the ``apix`` it was called with).
* :func:`blob_tomogram` - noise-free phantom of target-shaped dark blobs plus a smooth bright
  slab that sets the robust-normalisation scale (see the README).
* Readers for the outputs (STAR, maps).
"""

from __future__ import annotations

import contextlib
import math
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
from torch import nn

from utils.tiny import applied_main_config, tiny_main_config

TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = TESTS_DIR / "golden"
CHUNK = 64
EPS = 1e-4


# --------------------------------------------------------------------------------------
# phantoms
# --------------------------------------------------------------------------------------

def target_profile(r: np.ndarray, radius: float) -> np.ndarray:
    """CLAUDE.md Section 1.3 target shape: ``0.5 exp(-r^2/2 ss^2) + 0.5 exp(-r^2/2 sl^2)``,
    ``ss = 0.33 R / 2``, ``sl = 0.66 R / 2``; max 1 at ``r = 0``; 0 for ``r > R``."""
    ss, sl = 0.33 * radius / 2.0, 0.66 * radius / 2.0
    v = 0.5 * np.exp(-r ** 2 / (2 * ss ** 2)) + 0.5 * np.exp(-r ** 2 / (2 * sl ** 2))
    v[r > radius] = 0.0
    return v


def blob_tomogram(shape: Sequence[int], apix: float, particles: Iterable[Tuple[int, float, Sequence[float]]],
                  bright: float = 2.0, slab_vox: Optional[int] = None) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Noise-free ``(Z, Y, X)`` volume: one dark target-shaped blob (``-profile``) per particle
    ``(class_number, diameter_ang, (z, y, x) centre in voxels)``, plus a bright slab (value
    ``bright``, smooth erf edge) at low X. The slab covers > 5 % of the voxels, so the robust
    normalisation gives ``p95 - p5 = bright`` and ``-x_norm = profile / bright`` (peak 0.5 for
    ``bright = 2``, no saturation). Returns ``(volume, {class_number: (N, 3) xyz})``.
    """
    from scipy.special import erf

    shape = tuple(int(s) for s in shape)
    vol = np.zeros(shape, dtype=np.float32)
    coords: Dict[int, List] = {}
    for cls, diam, (cz, cy, cx) in particles:
        R = diam / 2.0 / apix
        lo = [max(0, int(math.floor(c - R - 1))) for c in (cz, cy, cx)]
        hi = [min(s, int(math.ceil(c + R + 2))) for c, s in zip((cz, cy, cx), shape)]
        zz, yy, xx = np.meshgrid(*[np.arange(a, b, dtype=np.float64) for a, b in zip(lo, hi)], indexing="ij")
        r = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.minimum(sub, -target_profile(r, R).astype(np.float32), out=sub)
        coords.setdefault(int(cls), []).append((cx, cy, cz))
    if slab_vox is None:
        slab_vox = int(math.ceil(0.08 * shape[2]))
    x = np.arange(shape[2], dtype=np.float64)
    slab = 0.5 * (1.0 - erf((x - slab_vox) / 3.0))
    vol += (bright * slab).astype(np.float32)[None, None, :]
    return vol, {c: np.asarray(v, dtype=np.float64) for c, v in coords.items()}


def write_mrc(path, vol, apix):
    from utils.phantoms import write_mrc as _w

    return _w(path, vol, apix)


def noise_tomogram(shape=(64, 96, 96), seed=0) -> np.ndarray:
    return np.random.RandomState(seed).standard_normal(shape).astype(np.float32)


# --------------------------------------------------------------------------------------
# oracle networks (swapped into InferenceModel.model)
# --------------------------------------------------------------------------------------

def _apix_value(apix) -> Optional[float]:
    if apix is None:
        return None
    if torch.is_tensor(apix):
        return float(apix.reshape(-1)[0].item())
    return float(apix)


class _OracleBase(nn.Module):
    """Mimics the network contract: ``forward(x, apix=None) -> (logits [B,C,...], hidden)``.
    ``conditioned=True`` adds a ``cond`` attribute, so ``BasePickingModel`` treats it as a
    scale-conditioned network and passes ``apix`` through. ``out.conv.conv`` exists so code
    that reads the channel count from the output conv finds ``C``."""

    def __init__(self, n_channels: int, conditioned: bool = False):
        super().__init__()
        self.C = int(n_channels)
        self.cond = nn.Identity() if conditioned else None
        self.out = nn.Module()
        self.out.conv = nn.Module()
        self.out.conv.conv = nn.Conv3d(1, self.C, 1)
        self.calls: List[Optional[float]] = []

    def _probs(self, x: torch.Tensor, apix: Optional[float]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x, apix=None, **kwargs):
        if isinstance(x, (tuple, list)):
            x = x[0]
        a = _apix_value(apix)
        self.calls.append(a)
        p = self._probs(x.float(), a).clamp(EPS, 1.0 - EPS)
        logits = torch.log(p) - torch.log1p(-p)
        hidden = torch.zeros(x.shape[0], 1, 2, 2, 2, dtype=logits.dtype, device=logits.device)
        return logits, hidden


class NegInputOracle(_OracleBase):
    """Spec G4-C4: every channel is ``sigmoid(logit(clip(-x_norm, eps, 1 - eps)))``."""

    def _probs(self, x, apix):
        return (-x).expand(-1, self.C, -1, -1, -1)


class LevelTagOracle(_OracleBase):
    """Constant output ``tag(apix)`` in every channel and voxel: a class map then shows the
    working voxel size at which that class was read."""

    def __init__(self, n_channels: int, tag: Callable[[Optional[float]], float], conditioned: bool = True):
        super().__init__(n_channels, conditioned=conditioned)
        self.tag = tag

    def _probs(self, x, apix):
        v = float(self.tag(apix))
        return torch.full((x.shape[0], self.C) + tuple(x.shape[2:]), v, dtype=x.dtype, device=x.device)


def tag_of_apix(apix: Optional[float]) -> float:
    """Distinct, smooth tag per working voxel size (0.1 for ``None``)."""
    if apix is None:
        return 0.1
    return 0.15 + 0.02 * float(apix)


# --------------------------------------------------------------------------------------
# predict() harness
# --------------------------------------------------------------------------------------

@contextlib.contextmanager
def inprocess_dask(_monkeypatch=None):
    """Force every Dask cluster that ``predict`` starts to run in this process (one worker,
    one thread, no dashboard), so monkeypatches and the global config reach the worker.
    The patches are undone when the block exits."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield from _inprocess_dask(monkeypatch)


def _inprocess_dask(monkeypatch):
    import dask.distributed as dd

    real = dd.LocalCluster

    def _local(*args, **kwargs):
        return real(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None)

    monkeypatch.setattr(dd, "LocalCluster", _local)
    import tomocpt.predict.run as run_mod

    for name in ("LocalCluster", "LocalCUDACluster"):
        if hasattr(run_mod, name):
            monkeypatch.setattr(run_mod, name, _local)
    try:
        import dask_cuda

        monkeypatch.setattr(dask_cuda, "LocalCUDACluster", _local, raising=False)
    except Exception:  # pragma: no cover
        pass
    yield


@contextlib.contextmanager
def oracle_models(_monkeypatch, factory: Callable[[object], nn.Module]):
    """Every ``InferenceModel.load_from_checkpoint`` returns the real loaded model with its
    network replaced by ``factory(model)``. Yields the list of oracles created. The patch is
    undone when the block exits."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield from _oracle_models(monkeypatch, factory)


def _oracle_models(monkeypatch, factory):
    from tomocpt.networks.inference import InferenceModel

    orig = InferenceModel.load_from_checkpoint
    made: List[nn.Module] = []

    def _load(*args, **kwargs):
        model = orig(*args, **kwargs)
        oracle = factory(model)
        model.model = oracle
        made.append(oracle)
        return model

    monkeypatch.setattr(InferenceModel, "load_from_checkpoint", _load)
    yield made


def n_classes_of(model) -> int:
    from tomocpt.labels.multiclass import class_table_from_config

    try:
        v2 = int(getattr(model.hparams.config.train, "data_pipeline_version", 1) or 1) == 2
    except Exception:
        v2 = False
    return len(class_table_from_config(model.hparams.config.prepData)) if v2 else 1


def neg_oracle_factory(conditioned: bool = False):
    return lambda model: NegInputOracle(n_classes_of(model), conditioned=conditioned)


def tag_oracle_factory(conditioned: bool = True):
    return lambda model: LevelTagOracle(n_classes_of(model), tag_of_apix, conditioned=conditioned)


def run_predict(tmp_path, monkeypatch, infer: dict, oracle: Optional[Callable] = None):
    """Run ``predict()`` with ``mainConfig.infer`` = tiny defaults updated by ``infer``
    (CPU, in-process Dask). Returns ``(predictions_dir, oracles)``."""
    from tomocpt.predict.run import predict

    cfg = tiny_main_config(tmp_path)
    cfg.infer.use_cuda = False
    cfg.infer.predictions_dir = Path(tmp_path) / "predictions"
    cfg.infer.cpus_per_worker = 1
    for k, v in infer.items():
        setattr(cfg.infer, k, v)
    made: List[nn.Module] = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(applied_main_config(cfg))
        stack.enter_context(inprocess_dask(monkeypatch))
        if oracle is not None:
            made = stack.enter_context(oracle_models(monkeypatch, oracle))
        predict(config=None)
    return Path(cfg.infer.predictions_dir), made


# --------------------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------------------

def read_star_particles(path):
    import starfile

    d = starfile.read(str(path))
    if isinstance(d, dict):
        return d["particles"]
    return d


def picks_by_class(star_path) -> Dict[int, np.ndarray]:
    """``{rlnClassNumber: (N, 4) [x, y, z, score]}`` from a RELION STAR file."""
    df = read_star_particles(star_path)
    out: Dict[int, np.ndarray] = {}
    cls = df["rlnClassNumber"].astype(int).to_numpy() if "rlnClassNumber" in df else np.ones(len(df), int)
    xyzs = df[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAutopickFigureOfMerit"]].to_numpy(float)
    for c in np.unique(cls):
        out[int(c)] = xyzs[cls == c]
    return out


def read_map(path) -> Tuple[np.ndarray, float]:
    import mrcfile

    with mrcfile.open(str(path), permissive=True) as m:
        return m.data.copy(), float(m.voxel_size.x)


def nearest_dist(picks_xyz: np.ndarray, true_xyz: np.ndarray) -> np.ndarray:
    """Distance from every true point to its nearest pick (inf if there are no picks)."""
    picks_xyz = np.asarray(picks_xyz, float).reshape(-1, 3)
    true_xyz = np.asarray(true_xyz, float).reshape(-1, 3)
    if len(picks_xyz) == 0:
        return np.full(len(true_xyz), np.inf)
    d = np.linalg.norm(true_xyz[:, None, :] - picks_xyz[None, :, :], axis=-1)
    return d.min(axis=1)


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------

def load_g3_helpers():
    from gate3 import g3_helpers

    return g3_helpers


def build_v2_ckpt(root: Path, classes, cond: bool) -> dict:
    """Tiny pipeline-v2 checkpoint written by the code under test (Gate 3 helper):
    ``classes`` = ``((number, name, diameter_ang), ...)``."""
    return load_g3_helpers().build_v2_checkpoint(Path(root), tuple(classes), cond=cond)


def load_gate4_common():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_gate4_common", GOLDEN_DIR / "gate4_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def copy_ckpt(src, dst) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def deep_equal(a, b) -> bool:
    """Structural equality of checkpoint contents (tensors compared exactly)."""
    import pickle

    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape \
            and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    try:
        return pickle.dumps(a) == pickle.dumps(b)
    except Exception:
        return a == b
