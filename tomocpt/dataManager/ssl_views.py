"""SSL v2 data (Gate 5, T5.1): source chunks at a reference voxel size and the scale-pair dataset.

No particle size is needed: each tomogram is resampled once (Fourier, anti-aliased) to
``apix_ref = clip(apix_native, apix_ref_min, apix_ref_max)``, robust-normalised and cut into
``source_chunk``^3 chunks with stride ``source_chunk // 2``. Items are built on the fly by
``tomocpt.networks.ssl.views.make_training_pair`` (two views at random scales per item).

Layout under ``chunks_dir``::

    chunk_layout.json                   {"layout": 2, "kind": "ssl", "ssl_version": 2, ...,
                                          "tomograms": [{"path", "dir", "split", "apix_native",
                                                         "apix_ref", "shape", "n_chunks"}]}
    {train,val}/<dir>/ssl_sources/<stem>_<z>_<y>_<x>.npy     float16 chunks
    done_labels_False.txt

Library code: every setting is an argument (no ``mainConfig`` reads).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from tomocpt.logger import get_logger

logging = get_logger()

SSL_KIND = "ssl"
SSL_SOURCES_DIRNAME = "ssl_sources"
MRC_SUFFIXES = (".mrc", ".rec", ".map")

__all__ = ["make_ssl_source_chunks", "ScalePairDataset", "SSLPairData", "ssl_reference_apix",
           "list_ssl_tomograms", "is_ssl_layout", "SSL_KIND"]


def ssl_reference_apix(apix_native: float, apix_ref_min: float, apix_ref_max: float) -> float:
    """``clip(apix_native, apix_ref_min, apix_ref_max)``."""
    return float(min(max(float(apix_native), float(apix_ref_min)), float(apix_ref_max)))


def is_ssl_layout(layout: Optional[dict]) -> bool:
    return bool(layout) and layout.get("kind") == SSL_KIND


def list_ssl_tomograms(training_data_dir: Union[str, Path]) -> List[str]:
    """Tomograms for SSL: the ``tomogram_path`` column of the prepare_data CSVs
    (``<dir>/*/*.csv``), or, without CSVs, the MRC files directly in ``training_data_dir``."""
    training_data_dir = str(training_data_dir)
    csvs = glob.glob(os.path.join(training_data_dir, "*", "*.csv"))
    if csvs:
        from tomocpt.utils import read_particles_csvs

        paths = read_particles_csvs(training_data_dir)["tomogram_path"].astype(str).tolist()
    else:
        paths = sorted(p for p in glob.glob(os.path.join(training_data_dir, "*"))
                       if p.lower().endswith(MRC_SUFFIXES))
    out = list(dict.fromkeys(paths))  # unique, in order
    if not out:
        raise FileNotFoundError(f"No tomograms found for self-supervised training in {training_data_dir}")
    return out


def _chunk_starts(n: int, size: int, stride: int) -> List[int]:
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)
    return starts


def _dir_names(paths: Sequence[str]) -> List[str]:
    stems = [Path(p).stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{i:03d}__{s}" for i, s in enumerate(stems)]


def _split(paths: Sequence[str], seed: int, val_fraction: float) -> List[str]:
    """``"train"`` / ``"val"`` per tomogram (by tomogram; one tomogram -> ``"both"``)."""
    if len(paths) < 2:
        return ["both"] * len(paths)
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(paths))
    _, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed)
    val = set(int(i) for i in np.atleast_1d(val_idx))
    return ["val" if i in val else "train" for i in idx]


def _prepare_volume(path: str, apix_override: Optional[float], apix_ref_min: float, apix_ref_max: float,
                    device: torch.device) -> Tuple[np.ndarray, float, float]:
    from tomocpt.dataManager.preprocessing import load_mrc, robust_normalization
    from tomocpt.dataManager.scaling import read_apix, resample_volume, working_shape

    apix_native = read_apix(path, apix_override)
    apix_ref = ssl_reference_apix(apix_native, apix_ref_min, apix_ref_max)
    vol = load_mrc(path, normalize=False)
    shape = working_shape(vol.shape, apix_native, apix_ref)
    if tuple(shape) != tuple(vol.shape):
        t = torch.from_numpy(vol).to(device)
        vol = resample_volume(t, shape, "fourier").cpu().numpy()
    vol = robust_normalization(np.asarray(vol, dtype=np.float32)).astype(np.float32)
    return vol, apix_native, apix_ref


def make_ssl_source_chunks(tomogram_paths: Sequence[Union[str, Path]], chunks_dir: Union[str, Path],
                           apix_ref_min: float = 8.0, apix_ref_max: float = 16.0, source_chunk: int = 128,
                           stride: Optional[int] = None, apix_override: Optional[float] = None,
                           seed: int = 42, val_fraction: float = 0.2, use_gpus: bool = False) -> dict:
    """Write SSL v2 source chunks for ``tomogram_paths`` under ``chunks_dir``; returns the manifest.

    Each tomogram: Fourier resampling to ``apix_ref``, robust normalisation, zero padding to at
    least ``source_chunk`` per axis, chunks of ``source_chunk``^3 with ``stride`` (default
    ``source_chunk // 2``). Train/val split by tomogram (``val_fraction``, ``seed``); a single
    tomogram gives every fifth chunk to validation."""
    from tomocpt.dataManager.chunking import get_chunk_layout_path, get_chunking_name_done
    from tomocpt.dataManager.metadata import write_chunk_layout

    paths = [str(p) for p in dict.fromkeys(str(p) for p in tomogram_paths)]
    if not paths:
        raise ValueError("make_ssl_source_chunks needs at least one tomogram")
    size = int(source_chunk)
    stride = int(stride or size // 2)
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if use_gpus and torch.cuda.is_available() else "cpu")
    names = _dir_names(paths)
    splits = _split(paths, seed, val_fraction)
    tomos = []
    for path, name, split in zip(paths, names, splits):
        vol, apix_native, apix_ref = _prepare_volume(path, apix_override, apix_ref_min, apix_ref_max, device)
        pad = [(0, max(0, size - n)) for n in vol.shape]
        if any(p[1] for p in pad):
            vol = np.pad(vol, pad, mode="constant")
        starts = [_chunk_starts(n, size, stride) for n in vol.shape]
        k = 0
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    chunk_split = split if split != "both" else ("val" if k % 5 == 4 else "train")
                    out_dir = chunks_dir / chunk_split / name / SSL_SOURCES_DIRNAME
                    out_dir.mkdir(parents=True, exist_ok=True)
                    chunk = vol[z:z + size, y:y + size, x:x + size].astype(np.float16)
                    np.save(out_dir / f"{Path(path).stem}_{z}_{y}_{x}.npy", chunk)
                    k += 1
        logging.info(f"SSL v2 chunks: {path} at {apix_native:.3f} A -> apix_ref {apix_ref:.3f} A, "
                     f"shape {tuple(vol.shape)}, {k} chunk(s) ({split})")
        tomos.append({"path": path, "dir": name, "split": split, "apix_native": apix_native,
                      "apix_ref": apix_ref, "shape": [int(n) for n in vol.shape], "n_chunks": k})
    layout = {"layout": 2, "kind": SSL_KIND, "ssl_version": 2, "source_chunk": size, "stride": stride,
              "apix_ref_min": float(apix_ref_min), "apix_ref_max": float(apix_ref_max), "tomograms": tomos}
    write_chunk_layout(chunks_dir, layout)
    assert Path(get_chunk_layout_path(str(chunks_dir))).is_file()
    Path(get_chunking_name_done(str(chunks_dir), require_labels=False)).write_text("done\n")
    return layout


class ScalePairDataset(Dataset):
    """SSL v2 items: ``make_training_pair`` on a source chunk (one item per chunk and epoch).

    ``files``: ``[(chunk path, apix_ref)]``. Training items draw fresh views every call
    (generator seeded from ``torch.initial_seed()``, which differs per worker and epoch, the
    index and a call counter); validation items are fixed (seeded by ``seed`` and the index)."""

    def __init__(self, files: Sequence[Tuple[str, float]], ssl_cfg, is_training: bool = True, seed: int = 0):
        self.files = [(str(p), float(a)) for p, a in files]
        self.ssl_cfg = ssl_cfg.resolved() if callable(getattr(ssl_cfg, "resolved", None)) else ssl_cfg
        self.is_training = bool(is_training)
        self.seed = int(seed)
        self._calls = 0

    def __len__(self) -> int:
        return len(self.files)

    def _rng(self, index: int) -> np.random.Generator:
        if not self.is_training:
            return np.random.default_rng([self.seed, index])
        self._calls += 1
        return np.random.default_rng([int(torch.initial_seed()) % (2 ** 63), index, self._calls])

    def __getitem__(self, index: int) -> dict:
        from tomocpt.networks.ssl.views import make_training_pair

        path, apix_ref = self.files[index]
        source = np.load(path).astype(np.float32)
        return make_training_pair(source=source, apix_ref=apix_ref, ssl_cfg=self.ssl_cfg, rng=self._rng(index))

    @classmethod
    def from_chunks_dir(cls, chunks_dir: Union[str, Path], split: str, ssl_cfg, is_training: bool,
                        seed: int = 0) -> "ScalePairDataset":
        from tomocpt.dataManager.metadata import read_chunk_layout

        layout = read_chunk_layout(chunks_dir)
        if not is_ssl_layout(layout):
            raise ValueError(f"{chunks_dir} holds no SSL v2 chunks (chunk_layout.json kind != 'ssl')")
        files = []
        for t in layout["tomograms"]:
            d = Path(chunks_dir) / split / t["dir"] / SSL_SOURCES_DIRNAME
            files += [(str(p), float(t["apix_ref"])) for p in sorted(d.glob("*.npy"))]
        if not files:
            raise ValueError(f"No SSL v2 chunks in split '{split}' of {chunks_dir}")
        return cls(files, ssl_cfg, is_training=is_training, seed=seed)


class SSLPairData(pl.LightningDataModule):
    """Data module of SSL v2 (``train.ssl.version = 2``): ``ScalePairDataset`` train / val."""

    def __init__(self, chunks_dir, batch_size: int, workers_for_data: int, ssl_cfg,
                 persistent_workers: bool = True, seed: int = 0):
        super().__init__()
        self.chunks_dir = chunks_dir
        self.batch_size = int(batch_size)
        self.workers = int(workers_for_data)
        self.ssl_cfg = ssl_cfg
        self.persistent = bool(persistent_workers) and self.workers > 0
        self.seed = int(seed)

    def setup(self, stage: Optional[str] = None):
        self.dataset_training = ScalePairDataset.from_chunks_dir(self.chunks_dir, "train", self.ssl_cfg, True, self.seed)
        self.dataset_val = ScalePairDataset.from_chunks_dir(self.chunks_dir, "val", self.ssl_cfg, False, self.seed)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.workers,
                          persistent_workers=self.persistent, drop_last=shuffle and len(ds) > self.batch_size)

    def train_dataloader(self):
        return self._loader(self.dataset_training, True)

    def val_dataloader(self):
        return self._loader(self.dataset_val, False)
