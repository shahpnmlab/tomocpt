"""Shared code for the Gate 2 v1-equivalence golden (G2-C1).

Used both by ``make_goldens_gate2.py`` (run in a worktree of ``v2-gate-0-base``) and by
``tests/gate2/test_g2_v1_equivalence.py`` (run on the code under test). It must not import
``tomocpt`` at module level: the caller decides which ``tomocpt`` is on the import path.

The legacy chunker (``do_chunking``) runs its tasks on a Dask cluster whose workers pick the
GPU when one is visible, and drops empty chunks with an unseeded ``random.random()``. To make
the output reproducible bit for bit, :func:`inprocess_chunking` runs the cluster in-process
(``processes=False``, one worker, one thread, so tasks run in submission order), hides CUDA
(``torch.cuda.is_available() -> False``, so every resize runs on the CPU in fp32), and gives
the chunking module its own ``random.Random(seed)``. The code path of ``do_chunking`` itself
is not changed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
from pathlib import Path

import numpy as np

#: make_dataset(n_tomos=2) with the default species (class 1, 120 A, 6 copies), 64x128x128 at 10 A/px.
DATASET_KWARGS = dict(n_tomos=2, seed=0)
#: Seed of the random.Random that replaces the chunking module's ``random``.
DROP_SEED = 0
#: Variants: "default_alpha" keeps ALPHA_FOR_DROPPING_EMPTY_CUBES at its default (empty chunks
#: dropped through the seeded RNG); "keep_all" sets it so high that no chunk is dropped.
VARIANTS = {"default_alpha": None, "keep_all": 1.0e9}


@contextlib.contextmanager
def inprocess_chunking(seed: int = DROP_SEED, alpha=None):
    """Patch the execution environment of ``do_chunking`` (see module docstring)."""
    import dask.distributed
    import torch

    from tomocpt.dataManager import chunking
    from tomocpt.mainConfig import mainConfig

    orig_cluster = dask.distributed.LocalCluster

    def _cluster(*args, **kwargs):
        kwargs.update(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None)
        return orig_cluster(**kwargs)

    saved = {
        "cluster": dask.distributed.LocalCluster,
        "chunk_cluster": getattr(chunking, "LocalCluster", None),
        "is_available": torch.cuda.is_available,
        "random": chunking.random,
        "alpha": mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES,
    }
    dask.distributed.LocalCluster = _cluster
    if saved["chunk_cluster"] is not None:
        chunking.LocalCluster = _cluster
    torch.cuda.is_available = lambda: False
    chunking.random = random.Random(seed)
    if alpha is not None:
        mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES = alpha
    try:
        yield
    finally:
        dask.distributed.LocalCluster = saved["cluster"]
        if saved["chunk_cluster"] is not None:
            chunking.LocalCluster = saved["chunk_cluster"]
        torch.cuda.is_available = saved["is_available"]
        chunking.random = saved["random"]
        mainConfig.prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES = saved["alpha"]


def prepare_v1_dataset(root: Path):
    """make_dataset + legacy prepare_data. Returns (SyntheticDataset, training_data_dir)."""
    from utils.phantoms import make_dataset

    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.labels.run import prepare_data

    root = Path(root)
    ds = make_dataset(root / "data", **DATASET_KWARGS)
    training_data_dir = root / "training_data"
    prep = PrepdataConfig(**ds.legacy_prepdata_fields(training_data_dir))
    prepare_data(prep)
    return ds, training_data_dir


def run_v1_chunking(training_data_dir: Path, chunks_dir: Path, variant: str):
    """Legacy do_chunking with every new argument at its default."""
    from tomocpt.dataManager.chunking import do_chunking
    from tomocpt.utils import read_particles_csvs

    tomos_df = read_particles_csvs(str(training_data_dir))
    tomos_df = tomos_df.sort_values("tomogram_path").reset_index(drop=True)
    with inprocess_chunking(alpha=VARIANTS[variant]):
        do_chunking(tomos_df, chunkedDataDir=str(chunks_dir), desired_particle_pixel_size=20,
                    n_cpus=1, require_labels=True, use_gpus=False)


def _array_digest(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "sum": float(arr.astype(np.float64).sum()),
        "max": float(arr.max()) if arr.size else 0.0,
    }


def summarize_chunks(chunks_dir: Path) -> dict:
    """{relative path: digest} for every file under ``chunks_dir`` except the MONAI cache.

    ``.nii.gz`` files are decoded (gzip bytes are not compared); ``.json`` files are parsed;
    other files are stored as text.
    """
    import torchio as tio

    chunks_dir = Path(chunks_dir)
    out = {}
    for p in sorted(chunks_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(chunks_dir).as_posix()
        if rel.startswith(".monai_cache"):
            continue
        if rel.endswith(".nii.gz"):
            out[rel] = _array_digest(tio.ScalarImage(p).data.numpy())
        elif rel.endswith(".json"):
            out[rel] = {"json": json.loads(p.read_text())}
        else:
            out[rel] = {"text": p.read_text()}
    return out
