"""G7-C3 (data path): ``prepData.negative_masks`` -> ``prepare_data`` -> pipeline-v2 chunking ->
``Data`` batches carry ``negative_mask [B, 1, 64, 64, 64]`` (bool) aligned with ``input_data``;
without ``negative_masks`` the batch has no such key (Gate 6 batches unchanged).

Inputs: two ``make_dataset`` tomograms (64 x 128 x 128, 10 A/px, class 1 "ribo" 120 A, inside the
band: no resampling). The tomograms are replaced by white noise plus a 2-voxel slab at value 100
(3 % of the voxels); the mask MRC marks exactly the slab. After the robust normalisation the slab
is clipped to +-3 (contrast inversion may flip the sign) while the noise stays below 2, so
``negative_mask == (|input| > 2.5)`` checks the alignment voxel by voxel. Spatial augmentation is
off (``affine_p = elastic_blur_p = 0``).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from gate2 import g2_helpers as G2
from gate6 import g6_helpers as H6
from utils.phantoms import make_dataset, write_mrc
from utils.tiny import applied_main_config

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
SHAPE = (64, 128, 128)
APIX = 10.0
PART = ((1, "ribo", 120.0),)
SLAB = (30, 32)


def _gate2_common():
    spec = importlib.util.spec_from_file_location("_gate2_common_for_g7", GOLDEN / "gate2_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("g7_negmask")
    ds = make_dataset(root / "data", n_tomos=2, shape=SHAPE, apix=APIX, seed=0)
    mask_dir = root / "membrane_masks"
    mask_dir.mkdir()
    rng = np.random.default_rng(0)
    for path, name in zip(ds.tomo_paths, ds.tomo_names):
        vol = rng.standard_normal(SHAPE).astype(np.float32)
        mask = np.zeros(SHAPE, np.uint8)
        mask[SLAB[0]:SLAB[1]] = 1
        vol[mask > 0] = 100.0
        write_mrc(path, vol, APIX)
        write_mrc(mask_dir / f"{name}.mrc", mask, APIX)
    return ds, mask_dir, root


def _chunks(dataset, tag, **extra):
    from tomocpt.dataManager.chunking import do_chunking
    from tomocpt.defaultConfigs.scale_config import ScaleConfig
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.labels.run import prepare_data
    from tomocpt.utils import read_particles_csvs

    ds, _, root = dataset
    tdd = root / f"td_{tag}"
    prep = H6.prep_config(ds, tdd, PART, (), **extra)
    with applied_main_config(None):
        prepare_data(prep)
        df = read_particles_csvs(str(tdd)).sort_values("tomogram_path").reset_index(drop=True)
        chunks = root / f"chunks_{tag}"
        with _gate2_common().inprocess_chunking():
            do_chunking(df, chunkedDataDir=str(chunks), desired_particle_pixel_size=20, n_cpus=1,
                        require_labels=True, use_gpus=False, pipeline_version=2, scale_cfg=ScaleConfig(),
                        class_table=class_table_from_config(prep), seed=42)
    return chunks


def test_G7_C3_negative_mask_reaches_the_batch_aligned(dataset):
    _, mask_dir, _ = dataset
    chunks = _chunks(dataset, "mask", negative_masks=f"membrane:{mask_dir}")
    data = G2.make_data(chunks, augmentation=G2.no_spatial_aug())
    n_batches, n_mask_vox = 0, 0
    for loader in (data.train_dataloader(), data.val_dataloader()):
        for batch in loader:
            assert "negative_mask" in batch, sorted(batch)
            m = batch["negative_mask"]
            x = batch["input_data"]
            assert m.dtype == torch.bool, m.dtype
            assert tuple(m.shape) == (x.shape[0], 1, 64, 64, 64), tuple(m.shape)
            expected = x.abs() > 2.5
            assert torch.equal(m, expected), \
                f"mask and slab disagree at {int((m != expected).sum())} voxels"
            n_batches += 1
            n_mask_vox += int(m.sum())
    assert n_batches > 0 and n_mask_vox > 0


def test_G7_C3_no_negative_masks_no_batch_key(dataset):
    chunks = _chunks(dataset, "nomask")
    data = G2.make_data(chunks, augmentation=G2.no_spatial_aug())
    batch = next(iter(data.train_dataloader()))
    assert "negative_mask" not in batch, sorted(batch)
