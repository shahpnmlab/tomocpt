"""Chunk metadata of data pipeline v2: per-directory ``chunk_meta.json`` and the
``chunks_dir/chunk_layout.json`` manifest (one manifest for v1 and v2, invariant 4)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Union

CHUNK_META_FNAME = "chunk_meta.json"
CHUNK_LAYOUT_FNAME = "chunk_layout.json"


@dataclass
class ChunkMeta:
    tomo_name: str
    dataset_id: int
    level: int
    scale_index: int
    apix_native: float
    apix_work: float
    class_available: List[bool] = field(default_factory=list)
    class_d_vox: List[float] = field(default_factory=list)
    pipeline_version: int = 2
    resample: str = "fourier"


def _atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def write_chunk_meta(dir: Union[str, Path], meta: ChunkMeta) -> Path:
    """Write ``<dir>/chunk_meta.json``."""
    d = asdict(meta)
    d["class_available"] = [bool(x) for x in d["class_available"]]
    d["class_d_vox"] = [float(x) for x in d["class_d_vox"]]
    path = Path(dir) / CHUNK_META_FNAME
    _atomic_write_json(path, d)
    return path


def read_chunk_meta(dir: Union[str, Path]) -> ChunkMeta:
    """Read ``<dir>/chunk_meta.json`` (unknown keys are ignored)."""
    with open(Path(dir) / CHUNK_META_FNAME) as f:
        d = json.load(f)
    names = {f.name for f in fields(ChunkMeta)}
    return ChunkMeta(**{k: v for k, v in d.items() if k in names})


def write_chunk_layout(chunks_dir: Union[str, Path], info: dict) -> Path:
    """Write ``<chunks_dir>/chunk_layout.json`` (the v1 code writes the same file)."""
    path = Path(chunks_dir) / CHUNK_LAYOUT_FNAME
    _atomic_write_json(path, info)
    return path


def read_chunk_layout(chunks_dir: Union[str, Path]) -> Optional[dict]:
    """The manifest as a dict, or None if ``chunks_dir`` has none."""
    path = Path(chunks_dir) / CHUNK_LAYOUT_FNAME
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def layout_pipeline_version(layout: Optional[dict]) -> int:
    """``pipeline_version`` of a manifest; missing (or no manifest) means 1 (invariant 4)."""
    if not layout:
        return 1
    return int(layout.get("pipeline_version", 1))
