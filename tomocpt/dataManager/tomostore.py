"""Tomogram store of on-the-fly crop sampling (Gate 8c.1).

Per tomogram, level and scale index, ``<split>/<dir name>__L{li}s{k}/`` holds what pipeline v2
chunking would cut into chunks (same plan of levels, resample, padding and coordinate map; the
directory names of chunking):

- ``volume.npy``: the padded working-size volume (``volume_dtype``, float16 by default);
- ``points.npy``: float32 ``[N, 5]`` ``z, y, x, class_index, d_vox`` in padded-volume voxels
  (after the border policy);
- ``filaments.npy`` (filament classes only): float32 ``[M, 8]`` segments, same frame;
- ``negative_mask.npy`` (negative masks only): the padded mask, bits packed along x
  (``np.packbits(mask, axis=-1)``);
- ``chunk_meta.json`` (unchanged :class:`~tomocpt.dataManager.metadata.ChunkMeta`) and
  ``store_meta.json`` (shape, chunk grid, empty-chunk drop probability, dtype).

The manifest ``chunk_layout.json`` gains ``"storage": "tomogram"`` (absent = chunk files).
Written by :func:`build_tomogram_store` (``chunking.do_chunking(..., storage="tomogram")``).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from tomocpt.dataManager.metadata import ChunkMeta, read_chunk_meta, write_chunk_meta

STORE_META_FNAME = "store_meta.json"
VOLUME_FNAME = "volume.npy"
POINTS_FNAME = "points.npy"
FILAMENTS_FNAME = "filaments.npy"
NEGATIVE_MASK_FNAME = "negative_mask.npy"


def write_store_entry(out_dir: Union[str, Path], padded: np.ndarray, points: np.ndarray, segments: np.ndarray,
                      negative_mask: Optional[np.ndarray], meta: ChunkMeta, drop_p: float, chunk_size: int,
                      chunk_stride: int, volume_dtype: str = "float16") -> Path:
    """One store directory (called by the pipeline v2 chunking task)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / VOLUME_FNAME, np.ascontiguousarray(padded, dtype=np.dtype(volume_dtype)))
    np.save(out / POINTS_FNAME, np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 5))
    if segments is not None and len(segments):
        np.save(out / FILAMENTS_FNAME, np.ascontiguousarray(segments, dtype=np.float32).reshape(-1, 8))
    if negative_mask is not None:
        np.save(out / NEGATIVE_MASK_FNAME, np.packbits(np.ascontiguousarray(negative_mask, dtype=bool), axis=-1))
    write_chunk_meta(out, meta)
    info = {"shape": [int(s) for s in padded.shape], "drop_p": float(drop_p), "chunk_size": int(chunk_size),
            "chunk_stride": int(chunk_stride), "volume_dtype": str(volume_dtype)}
    with open(out / STORE_META_FNAME, "w") as f:
        json.dump(info, f, indent=1)
    return out


@dataclass
class TomogramStore:
    """One store directory. Arrays are memory-mapped lazily (also after pickling into workers)."""
    path: str
    meta: ChunkMeta
    shape: tuple
    drop_p: float
    chunk_size: int
    chunk_stride: int
    volume_dtype: str
    points: np.ndarray
    segments: Optional[np.ndarray]
    has_negative_mask: bool
    _vol: Optional[np.ndarray] = field(default=None, repr=False)
    _neg: Optional[np.ndarray] = field(default=None, repr=False)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_vol"] = None
        state["_neg"] = None
        return state

    @property
    def volume(self) -> np.ndarray:
        if self._vol is None:
            self._vol = np.load(Path(self.path) / VOLUME_FNAME, mmap_mode="r")
        return self._vol

    def crop(self, corner, size: Optional[int] = None) -> np.ndarray:
        """float32 ``[size]^3`` crop at ``corner`` (z, y, x) of the padded volume."""
        n = int(size or self.chunk_size)
        z, y, x = (int(c) for c in corner)
        return np.asarray(self.volume[z:z + n, y:y + n, x:x + n], dtype=np.float32)

    def negative_crop(self, corner, size: Optional[int] = None) -> Optional[np.ndarray]:
        """bool ``[size]^3`` negative-mask crop, or None when the tomogram has no mask."""
        if not self.has_negative_mask:
            return None
        if self._neg is None:
            self._neg = np.load(Path(self.path) / NEGATIVE_MASK_FNAME, mmap_mode="r")
        n = int(size or self.chunk_size)
        z, y, x = (int(c) for c in corner)
        b0, b1 = x // 8, (x + n + 7) // 8
        bits = np.unpackbits(np.asarray(self._neg[z:z + n, y:y + n, b0:b1]), axis=-1)
        o = x - 8 * b0
        return bits[..., o:o + n].astype(bool)

    def points_in_box(self, corner, size: Optional[int] = None) -> np.ndarray:
        """Points of the crop at ``corner`` in crop voxels (the chunking rule: centres within the
        box grown by the radius), float32 ``[n, 5]``."""
        n = int(size or self.chunk_size)
        pts = self.points
        if not len(pts):
            return np.zeros((0, 5), np.float32)
        local = pts[:, :3] - np.asarray(corner, dtype=np.float32)
        r = (pts[:, 4].astype(np.float64) / 2.0)[:, None]
        sel = np.all((local >= -r) & (local < n + r), axis=1)
        return np.concatenate([local[sel], pts[sel, 3:5]], axis=1).astype(np.float32)

    def segments_in_box(self, corner, size: Optional[int] = None) -> Optional[np.ndarray]:
        """Filament segments near the crop at ``corner``, in crop voxels (None when there are none)."""
        if self.segments is None or not len(self.segments):
            return None
        from tomocpt.labels.filaments import segments_near_box

        n = int(size or self.chunk_size)
        sel = segments_near_box(self.segments, [int(c) for c in corner], n)
        if not sel.any():
            return None
        out = self.segments[sel].copy()
        shift = np.asarray(corner, dtype=np.float32)
        out[:, 0:3] -= shift
        out[:, 3:6] -= shift
        return out


def is_store_dir(d: Union[str, Path]) -> bool:
    return (Path(d) / STORE_META_FNAME).is_file() and (Path(d) / VOLUME_FNAME).is_file()


def open_store_dir(d: Union[str, Path]) -> TomogramStore:
    d = Path(d)
    with open(d / STORE_META_FNAME) as f:
        info = json.load(f)
    seg_path = d / FILAMENTS_FNAME
    segs = np.load(seg_path).astype(np.float32).reshape(-1, 8) if seg_path.is_file() else None
    return TomogramStore(
        path=str(d), meta=read_chunk_meta(d), shape=tuple(int(s) for s in info["shape"]),
        drop_p=float(info["drop_p"]), chunk_size=int(info["chunk_size"]), chunk_stride=int(info["chunk_stride"]),
        volume_dtype=str(info["volume_dtype"]),
        points=np.load(d / POINTS_FNAME).astype(np.float32).reshape(-1, 5),
        segments=segs if segs is not None and len(segs) else None,
        has_negative_mask=(d / NEGATIVE_MASK_FNAME).is_file())


def read_tomogram_store(split_dir: Union[str, Path]) -> List[TomogramStore]:
    """Every store directory of one split (``<chunks_dir>/train`` or ``/val``), sorted by name
    (symlinked fold directories included)."""
    root = Path(split_dir)
    if not root.is_dir():
        return []
    return [open_store_dir(d) for d in sorted(root.iterdir(), key=lambda p: p.name) if is_store_dir(d)]


def build_tomogram_store(tomosDf: pd.DataFrame, chunkedDataDir: str, **kwargs) -> None:
    """Pipeline v2 ``do_chunking`` that writes the tomogram store instead of chunk files (same
    keyword arguments; ``volume_dtype`` optional)."""
    from tomocpt.dataManager.chunking import do_chunking

    kwargs = dict(kwargs)
    kwargs.setdefault("desired_particle_pixel_size", 20)
    kwargs["pipeline_version"] = 2
    kwargs["storage"] = "tomogram"
    do_chunking(tomosDf, chunkedDataDir=chunkedDataDir, **kwargs)
