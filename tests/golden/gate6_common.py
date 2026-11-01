"""Inputs and helpers shared by ``make_goldens_gate6.py`` and ``tests/gate6`` (G6-C1).

G6-C1: with no filament class, chunks, batches, model outputs and predictions equal Gate 4.
The reference is the ``v2-dev`` code before Gate 6 (Gates 2-5 behaviour; nothing
filament-related exists). The generator runs this module on that code; the test runs the same
functions on the code under test and compares.

Loaded by file path. ``tomocpt`` is imported inside the functions only.

Pipeline (all CPU, fp32, in-process, seeded):

* :func:`prepare_and_chunk` - ``utils.phantoms.make_dataset`` (two tomograms, two particle
  species, no tubes) -> multi-class ``prepare_data`` -> ``do_chunking(pipeline_version=2)``
  (in-process Dask, CUDA hidden, as ``gate2_common.inprocess_chunking``).
* :func:`chunk_arrays` - every ``points/*.npy`` in full, every chunk volume as a digest
  (float64 sum, sum of squares, seeded projection) plus a ``[::4, ::4, ::4]`` subsample.
* :func:`batch_arrays` - the first validation batch and the first training batch (default
  ``AugmentationConfig``, seeds 0) of ``Data`` on those chunks: every key (CLAUDE.md 5.1);
  dense tensors as digest + subsample.
* :func:`forward_arrays` - a tiny v2 ``BasePickingModel`` (two classes, voxel-size conditioning
  on) with ``gate3_common.reinit_`` weights: logits at 10 and 15 A/px (digest + subsample).
* :func:`predict_arrays` - a tiny v2 checkpoint (Gate 3 helper, then ``reinit_`` weights) run
  through ``tomocpt.predict.run.predict()`` (in-process, CPU, one thread) on a blob tomogram,
  with the real network (per-class maps and STAR coordinates) and with the Gate 4
  ``NegInputOracle`` (STAR coordinates).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

CHUNK = 64
APIX = 10.0
SHAPE = (64, 96, 96)
#: (class_number, diameter_ang, count, hollow): 12 and 20 voxels at 10 A/px, inside the band.
SPECIES = ((1, 120.0, 5, False), (2, 200.0, 3, False))
CLASSES = ((1, "small", 120.0), (2, "big", 200.0))
DATASET_SEED = 6
SUB = 4  # subsample stride of dense arrays
PROJ_SEED = 1234
PREDICT_THREADS = 1


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gate2_common():
    return _load("_gate2_common_for_g6", HERE / "gate2_common.py")


def gate3_common():
    return _load("_gate3_common_for_g6", HERE / "gate3_common.py")


def gate4_common():
    return _load("_gate4_common_for_g6", HERE / "gate4_common.py")


def seed_all(seed: int = 0):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def digest(prefix: str, arr, out: dict):
    """Store ``<prefix>__sum``, ``__sumsq``, ``__proj``, ``__shape`` and ``__sub`` (stride SUB on
    the last three axes) of a dense array in ``out``."""
    a = np.asarray(arr)
    a64 = a.astype(np.float64)
    w = np.random.RandomState(PROJ_SEED).standard_normal(a.size)
    out[f"{prefix}__shape"] = np.asarray(a.shape, dtype=np.int64)
    out[f"{prefix}__sum"] = np.asarray(a64.sum())
    out[f"{prefix}__sumsq"] = np.asarray((a64 ** 2).sum())
    out[f"{prefix}__proj"] = np.asarray(float(np.dot(a64.reshape(-1), w)))
    if a.ndim >= 3:
        out[f"{prefix}__sub"] = np.ascontiguousarray(a[..., ::SUB, ::SUB, ::SUB]).astype(np.float32)
    else:
        out[f"{prefix}__sub"] = a.astype(np.float32) if a.dtype.kind == "f" else a


# --------------------------------------------------------------------------------------
# chunks
# --------------------------------------------------------------------------------------

def prepare_and_chunk(root: Path):
    """Returns ``(chunks_dir, tracking df, class table)``."""
    from utils.phantoms import make_dataset

    from tomocpt.dataManager.chunking import do_chunking
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
    from tomocpt.defaultConfigs.scale_config import ScaleConfig
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    root = Path(root)
    seed_all(0)
    ds = make_dataset(root / "data", n_tomos=2, shape=SHAPE, apix=APIX, species=SPECIES, seed=DATASET_SEED)
    tdd = root / "training_data"
    prep = PrepdataConfig(raw_data_dir=str(ds.tomo_dir), training_data_dir=tdd, particle_length_ang="200",
                          coordinate_files=str(ds.star_path), class_id="all", multiclass=True,
                          class_diameters=",".join(f"{n}:{d}" for n, _, d in CLASSES),
                          class_names=",".join(f"{n}:{name}" for n, name, _ in CLASSES))
    prepare_data(prep)
    df = read_particles_csvs(str(tdd)).sort_values("tomogram_path").reset_index(drop=True)
    table = class_table_from_config(prep)
    chunks = root / "chunks"
    with gate2_common().inprocess_chunking():
        do_chunking(df, chunkedDataDir=str(chunks), desired_particle_pixel_size=20, n_cpus=1,
                    require_labels=True, use_gpus=False, pipeline_version=2, scale_cfg=ScaleConfig(),
                    class_table=table, seed=42)
    return chunks, df, table


def chunk_arrays(chunks_dir: Path):
    """``(arrays, info)``: arrays keyed by relative path; info = sorted file lists and the
    manifest (``chunk_layout.json``) with paths made relative."""
    import torchio as tio

    chunks_dir = Path(chunks_dir)
    arrays, files = {}, []
    for p in sorted(chunks_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(chunks_dir).as_posix()
        parts = rel.split("/")
        if rel.startswith(".monai_cache") or len(parts) < 4:
            continue
        kind = parts[2]
        if kind == "points" and rel.endswith(".npy"):
            arrays[f"pts:{rel}"] = np.load(p)
            files.append(rel)
        elif kind == "volumes" and rel.endswith(".nii.gz"):
            digest(f"vol:{rel}", tio.ScalarImage(p).data.numpy(), arrays)
            files.append(rel)
    layout = json.loads((chunks_dir / "chunk_layout.json").read_text())
    return arrays, {"files": files, "layout": _relative(layout, chunks_dir)}


def _relative(obj, root: Path):
    root_s = str(Path(root).resolve())
    parent_s = str(Path(root).resolve().parent)
    if isinstance(obj, dict):
        return {k: _relative(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relative(v, root) for v in obj]
    if isinstance(obj, str):
        return obj.replace(root_s, "<chunks>").replace(parent_s, "<root>")
    return obj


# --------------------------------------------------------------------------------------
# batches
# --------------------------------------------------------------------------------------

def make_data(chunks_dir):
    from tomocpt.dataManager.dataloading import Data
    from tomocpt.defaultConfigs.scale_config import ScaleConfig
    from tomocpt.defaultConfigs.train_config import AugmentationConfig, ChunkSampling

    data = Data(data_dir=str(chunks_dir), return_labels=True, batch_size=2, workers_for_data=0,
                augmentation=AugmentationConfig(), pipeline_version=2, scale_cfg=ScaleConfig(),
                chunk_sampling=ChunkSampling.uniform, repeat_factor_threshold=0.1)
    data.setup()
    return data


def batch_arrays(chunks_dir):
    import torch

    out = {}
    seed_all(0)
    data = make_data(chunks_dir)
    for tag, loader_fn in (("val", data.val_dataloader), ("train", data.train_dataloader)):
        seed_all(0)
        batch = next(iter(loader_fn()))
        for k, v in sorted(batch.items()):
            if not torch.is_tensor(v):
                continue
            a = v.detach().cpu().numpy()
            if a.ndim >= 4:
                digest(f"batch:{tag}:{k}", a, out)
            else:
                out[f"batch:{tag}:{k}"] = a
    return out


# --------------------------------------------------------------------------------------
# model forward
# --------------------------------------------------------------------------------------

def v2_config(tmp_path, classes=CLASSES, cond=True):
    from utils.tiny import tiny_main_config

    cfg = tiny_main_config(tmp_path)
    cfg.train.data_pipeline_version = 2
    cfg.prepData.multiclass = True
    cfg.prepData.class_diameters = ",".join(f"{n}:{d}" for n, _, d in classes)
    cfg.prepData.class_names = ",".join(f"{n}:{name}" for n, name, _ in classes)
    cfg.train.network.scale_conditioning = cond
    return cfg


def forward_arrays(tmp_path):
    import torch

    from utils.tiny import applied_main_config

    from tomocpt.networks.pickingModel import BasePickingModel

    out = {}
    seed_all(0)
    with applied_main_config(v2_config(tmp_path)):
        model = BasePickingModel(lr=1e-3, train_continue=None)
        gate3_common().reinit_(model.model)
        model.eval()
        x = torch.from_numpy(np.random.RandomState(61).standard_normal((1, 1, CHUNK, CHUNK, CHUNK))
                             .astype(np.float32))
        with torch.no_grad():
            for a in (10.0, 15.0):
                logits = model(x, apix=torch.full((1, 1), a))
                digest(f"forward:apix{a:g}", logits.numpy(), out)
    return out


# --------------------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------------------

#: (class_number, diameter_ang, (z, y, x)) blobs of the predict tomogram
BLOBS = ((1, 120.0, (32.2, 30.3, 40.6)), (1, 120.0, (31.7, 70.8, 70.4)), (1, 120.0, (30.6, 25.1, 75.2)),
         (2, 200.0, (33.3, 66.4, 30.7)))


def predict_tomogram():
    from gate4 import g4_helpers as G4

    vol, coords = G4.blob_tomogram(SHAPE, APIX, BLOBS)
    return vol, coords


def build_reinit_v2_ckpt(root: Path) -> Path:
    from gate3 import g3_helpers as G3

    from utils.tiny import applied_main_config

    root = Path(root)
    seed_all(0)
    with applied_main_config(None):
        info = G3.build_v2_checkpoint(root / "fit", CLASSES, cond=False)
    out = root / "v2_reinit.ckpt"
    with applied_main_config(None):
        gate4_common().make_reinit_ckpt(info["path"], out)
    return out


@contextlib.contextmanager
def _threads(n):
    import torch

    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def _read_star_rows(path: Path) -> np.ndarray:
    import starfile

    d = starfile.read(str(path))
    df = d["particles"] if isinstance(d, dict) else d
    cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAutopickFigureOfMerit"]
    a = df[cols].to_numpy(np.float64)
    order = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
    return a[order]


def predict_arrays(root: Path):
    import mrcfile
    import pytest

    from gate4 import g4_helpers as G4

    root = Path(root)
    out = {}
    ckpt = build_reinit_v2_ckpt(root / "ckpt")
    vol, _ = predict_tomogram()
    tomo = G4.write_mrc(root / "tomograms" / "TS_c1.mrc", vol, APIX)
    runs = {
        "net": dict(oracle=None),
        "oracle": dict(oracle=G4.neg_oracle_factory()),
    }
    for tag, spec in runs.items():
        seed_all(0)
        with _threads(PREDICT_THREADS), pytest.MonkeyPatch.context() as mp:
            pd, _ = G4.run_predict(root / f"run_{tag}", mp, dict(
                weights=ckpt, tomogram_file=tomo, save_prediction_confidence_map=(tag == "net"),
                cross_class_nms=(tag == "net"),
                # Gate 4 spacing (one diameter), pinned since the v2 default became 0.5 x d
                # (human decision 1 Nov 2026); 1.0 x d reproduces the golden exactly
                peak_spacing_fraction=1.0,
                predictions_coord_format="relion_31", predictions_coord_filename="picks.star"),
                oracle=spec["oracle"])
        for _, name, _ in CLASSES:
            star = pd / f"picks_{name}.star"
            out[f"pred:{tag}:{name}:coords"] = _read_star_rows(star) if star.is_file() else np.zeros((0, 4))
            if tag == "net":
                with mrcfile.open(str(pd / f"TS_c1_{name}.mrc"), permissive=True) as m:
                    a = m.data.copy()
                out[f"pred:{tag}:{name}:map_dtype"] = np.asarray(str(a.dtype))
                digest(f"pred:{tag}:{name}:map", a.astype(np.float32), out)
    return out


def all_arrays(root: Path):
    """Every golden array (dict) and the json info."""
    root = Path(root)
    chunks, _, _ = prepare_and_chunk(root / "c")
    arrays, info = chunk_arrays(chunks)
    arrays.update(batch_arrays(chunks))
    arrays.update(forward_arrays(root / "f"))
    arrays.update(predict_arrays(root / "p"))
    return arrays, info
