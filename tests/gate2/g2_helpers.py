"""Helpers for the Gate 2 tests (no tests here).

New public names (spec ``02_AGENT_SCALE_DATA.md``, "Public names") are imported inside the
functions, so a module that also holds legacy-equivalence tests still imports on code that
does not have them yet.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

TESTS_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = TESTS_DIR / "golden"
CHUNK = 64
STRIDE = 32
BAND = (10.0, 32.0)


def load_common():
    """``tests/golden/gate2_common.py`` (shared with the golden generator), loaded by path."""
    spec = importlib.util.spec_from_file_location("_gate2_common", GOLDEN_DIR / "gate2_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def grid_starts(size: int, patch: int = CHUNK, stride: int = STRIDE) -> List[int]:
    """Start indices of TorchIO GridSampler patches along one axis (patch 64, overlap 32)."""
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def v1_pad_front(size: int, min_size: int = CHUNK) -> int:
    """Front padding of ``preprocessing.symmetrize_and_pad`` along one axis."""
    return max(0, min_size - size) // 2


def profile(r, radius):
    """CLAUDE.md Section 1.3 profile (max 1 at r = 0, zero for r > R)."""
    r = np.asarray(r, dtype=np.float64)
    ss, sl = 0.33 * radius / 2.0, 0.66 * radius / 2.0
    v = 0.5 * np.exp(-r ** 2 / (2 * ss ** 2)) + 0.5 * np.exp(-r ** 2 / (2 * sl ** 2))
    return np.where(r > radius, 0.0, v)


def sphere_volume(shape, centres_zyx, radii, value=-1.0, noise=0.3, seed=0):
    """Gaussian noise plus dark solid spheres; content only matters for normalisation."""
    rng = np.random.default_rng(seed)
    vol = rng.normal(0.0, noise, size=shape).astype(np.float32)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    for (cz, cy, cx), r in zip(centres_zyx, radii):
        vol[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] += value
    return vol


# --------------------------------------------------------------------------------------
# pipeline v2 inputs (tracking table + coordinate CSVs, as the spec defines them)
# --------------------------------------------------------------------------------------

def write_v2_inputs(root: Path, tomos: Sequence[dict], class_diameters: Dict[int, float],
                    split_rows: bool = False, dataset_dir_name: str = "dataset_1_class_all") -> pd.DataFrame:
    """Write tomograms, ``coords/<tomo>.csv`` files and tracking CSV(s); return the table.

    ``tomos``: dicts with ``name``, ``volume`` (Z, Y, X), ``apix`` and ``coords``
    (``{class_number: (N, 3) xyz}`` in native voxels).

    The table has the legacy columns (``tomogram_path``, ``label_path``,
    ``particle_diameter_angst``, ``num_particles``) plus the two the spec adds
    (``coords_path``, ``apix_native``). Coordinate CSVs have the spec's columns
    ``x, y, z, class_number, diameter_ang``.

    ``split_rows=True`` writes one dataset per class (``dataset_<k>_class_<c>``), each with a
    row for every tomogram and a coordinate CSV holding only that class: several rows then
    refer to the same tomogram file (the layout of legacy multi-species training data).
    """
    from utils.phantoms import write_mrc

    root = Path(root)
    tomo_dir = root / "tomograms"
    tdd = root / "training_data"
    rows = []
    groups = ([(f"dataset_{k + 1}_class_{c}", [c]) for k, c in enumerate(sorted(class_diameters))]
              if split_rows else [(dataset_dir_name, sorted(class_diameters))])
    for t in tomos:
        tomo_path = tomo_dir / f"{t['name']}.mrc"
        if not tomo_path.is_file():
            write_mrc(tomo_path, t["volume"], t["apix"])
    for ds_name, classes in groups:
        ds_dir = tdd / ds_name
        ds_rows = []
        for t in tomos:
            recs = []
            for c in classes:
                xyz = np.asarray(t["coords"].get(c, np.zeros((0, 3))), dtype=float).reshape(-1, 3)
                for x, y, z in xyz:
                    recs.append(dict(x=x, y=y, z=z, class_number=int(c), diameter_ang=float(class_diameters[c])))
            coords_df = pd.DataFrame(recs, columns=["x", "y", "z", "class_number", "diameter_ang"])
            coords_path = ds_dir / "coords" / f"{t['name']}.csv"
            coords_path.parent.mkdir(parents=True, exist_ok=True)
            coords_df.to_csv(coords_path, index=False)
            label_cls = "all" if len(classes) > 1 else str(classes[0])
            ds_rows.append({
                "tomogram_path": str(tomo_dir / f"{t['name']}.mrc"),
                "label_path": str(ds_dir / f"class_{label_cls}" / "labels" / f"{t['name']}.labels.mrc"),
                "particle_diameter_angst": float(max(class_diameters[c] for c in classes)),
                "num_particles": int(len(coords_df)),
                "coords_path": str(coords_path),
                "apix_native": float(t["apix"]),
            })
        pd.DataFrame(ds_rows).to_csv(ds_dir / "tracking.csv", index=False)
        rows.extend(ds_rows)
    return pd.DataFrame(rows).reset_index(drop=True)


def phantom_tomos(names, shape, apix, species, seed0=0, snr=1.0, wedge_deg=60.0):
    from utils.phantoms import make_phantom

    out = []
    for i, name in enumerate(names):
        ph = make_phantom(shape=shape, apix=apix, species=species, wedge_deg=wedge_deg, snr=snr, seed=seed0 + i)
        out.append(dict(name=name, volume=ph.volume, apix=apix, coords=ph.coords, shape=tuple(shape)))
    return out


def run_v2_chunking(df, chunks_dir, class_table, scale_cfg=None, seed=42, **scale_overrides):
    """``do_chunking`` with the Gate 2 keyword arguments, run in-process on the CPU."""
    from tomocpt.dataManager.chunking import do_chunking
    from tomocpt.defaultConfigs.scale_config import ScaleConfig

    if scale_cfg is None:
        scale_cfg = ScaleConfig(**scale_overrides)
    common = load_common()
    with common.inprocess_chunking():
        do_chunking(df, chunkedDataDir=str(chunks_dir), desired_particle_pixel_size=20, n_cpus=1,
                    require_labels=True, use_gpus=False, pipeline_version=2, scale_cfg=scale_cfg,
                    class_table=class_table, seed=seed)
    return Path(chunks_dir)


# --------------------------------------------------------------------------------------
# reading chunk output
# --------------------------------------------------------------------------------------

def chunk_dirs(chunks_dir: Path, splits=("train", "val")) -> List[Path]:
    out = []
    for split in splits:
        d = Path(chunks_dir) / split
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir() if p.is_dir()))
    return out


def points_files(chunk_dir: Path) -> List[Path]:
    return sorted((Path(chunk_dir) / "points").glob("*.npy"))


def volume_files(chunk_dir: Path) -> List[Path]:
    return sorted((Path(chunk_dir) / "volumes").glob("*.nii.gz"))


def metas(chunks_dir: Path, splits=("train", "val")):
    from tomocpt.dataManager.metadata import read_chunk_meta

    return {d: read_chunk_meta(d) for d in chunk_dirs(chunks_dir, splits)}


def table_from_manifest(entries):
    """Class table rows from ``chunk_layout.json`` -> ``ClassSpec``.

    The spec does not fix the JSON form of a row; accept ``dataclasses.asdict(ClassSpec)``
    dicts (keys ``number``, ``name``, ``diameter_ang``, ``kind``) or lists in field order.
    """
    from tomocpt.labels.multiclass import ClassSpec

    rows = []
    for e in entries:
        if isinstance(e, dict):
            rows.append(ClassSpec(number=int(e["number"]), name=str(e["name"]),
                                  diameter_ang=float(e["diameter_ang"]), kind=str(e.get("kind", "particle"))))
        else:
            e = list(e)
            rows.append(ClassSpec(int(e[0]), str(e[1]), float(e[2]), str(e[3]) if len(e) > 3 else "particle"))
    return rows


def same_table(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (int(x.number), str(x.name), str(getattr(x.kind, "value", x.kind))) != \
                (int(y.number), str(y.name), str(getattr(y.kind, "value", y.kind))):
            return False
        if not math.isclose(float(x.diameter_ang), float(y.diameter_ang), rel_tol=1e-9):
            return False
    return True


# --------------------------------------------------------------------------------------
# data module
# --------------------------------------------------------------------------------------

def make_data(chunks_dir, augmentation=None, scale_cfg=None, chunk_sampling=None, threshold=0.1,
              batch_size=2, pipeline_version=2, setup=True):
    from tomocpt.dataManager.dataloading import Data
    from tomocpt.defaultConfigs.scale_config import ScaleConfig
    from tomocpt.defaultConfigs.train_config import AugmentationConfig, ChunkSampling

    data = Data(data_dir=str(chunks_dir), return_labels=True, batch_size=batch_size, workers_for_data=0,
                augmentation=augmentation if augmentation is not None else AugmentationConfig(),
                pipeline_version=pipeline_version,
                scale_cfg=scale_cfg if scale_cfg is not None else ScaleConfig(),
                chunk_sampling=chunk_sampling if chunk_sampling is not None else ChunkSampling.uniform,
                repeat_factor_threshold=threshold)
    if setup:
        data.setup()
    return data


def no_spatial_aug(**kw):
    """AugmentationConfig without spatial transforms (affine off, elastic/blur stage off)."""
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    base = dict(affine_p=0.0, elastic_blur_p=0.0)
    base.update(kw)
    return AugmentationConfig(**base)


def all_intensity_aug():
    """Spatial transforms off; every intensity and Fourier augmentation at p = 1.

    ``tio.RandomAnisotropy`` is left off: it resamples the whole subject (a spatial operation).
    The elastic/blur stage is kept at p = 1 with all weight on the blur (an intensity transform).
    """
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    return AugmentationConfig(
        affine_p=0.0, elastic_blur_p=1.0, elastic_weight=0.0, blur_weight=1.0,
        use_mw_aug=True, use_fourier_aug=True, missing_wedge_prob=1.0, amplitude_prob=1.0,
        sample_kernel_prob=1.0,
        brightness_gradient_p=1.0, local_gamma_p=1.0, brightness_p=1.0, contrast_p=1.0,
        noise_p=1.0, gamma_p=1.0, bias_field_p=1.0,
    )


def iter_transforms(t):
    """Every transform inside a TorchIO Compose / OneOf tree (depth first)."""
    import torchio as tio

    yield t
    if isinstance(t, tio.Compose):
        for s in t.transforms:
            yield from iter_transforms(s)
    elif isinstance(t, tio.OneOf):
        for s in t.transforms_dict:
            yield from iter_transforms(s)


def read_json(p: Path):
    return json.loads(Path(p).read_text())
