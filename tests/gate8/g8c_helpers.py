"""Shared fixtures of the Gate 8c tests: phantom tomograms, the chunk baseline and the tomogram
store built from the same inputs (in-process chunking on the CPU)."""
from pathlib import Path

import numpy as np

from gate2 import g2_helpers as G2

DIAMS = {1: 120.0, 2: 200.0}
NAMES = {1: "a", 2: "b"}
SPECIES = ((1, 120.0, 6, False), (2, 200.0, 2, False))


def write_inputs(root: Path):
    """Four tomograms: three at 10 A/px (no resample) and one at 6 A/px (class b is 33 voxels there,
    above the band: resampled to 6.25 A/px and padded on z)."""
    tomos = G2.phantom_tomos(["TS_a", "TS_b", "TS_c"], (64, 128, 128), 10.0, SPECIES, seed0=80)
    tomos += G2.phantom_tomos(["TS_d"], (64, 128, 128), 6.0, SPECIES, seed0=90)
    return G2.write_v2_inputs(root, tomos, DIAMS)


def write_sparse_inputs(root: Path):
    """Four larger, sparse tomograms (empty grid cells and a rare class: the sampler statistics)."""
    sp = ((1, 120.0, 4, False), (2, 200.0, 1, False))
    tomos = G2.phantom_tomos(["SP_a", "SP_b", "SP_c", "SP_d"], (64, 256, 256), 10.0, sp, seed0=70)
    return G2.write_v2_inputs(root, tomos, DIAMS)


def class_table():
    from tomocpt.labels.multiclass import class_table_from_config
    from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig

    prep = PrepdataConfig()
    prep.multiclass = True
    prep.class_diameters = ",".join(f"{k}:{v}" for k, v in DIAMS.items())
    prep.class_names = ",".join(f"{k}:{v}" for k, v in NAMES.items())
    return class_table_from_config(prep)


def build(df, out: Path, storage: str = "chunks", volume_dtype: str = "float16", seed: int = 42, **kw) -> Path:
    from tomocpt.dataManager.chunking import do_chunking
    from tomocpt.defaultConfigs.scale_config import ScaleConfig

    with G2.load_common().inprocess_chunking():
        do_chunking(df, chunkedDataDir=str(out), desired_particle_pixel_size=20, n_cpus=1, require_labels=True,
                    use_gpus=False, pipeline_version=2, scale_cfg=ScaleConfig(), class_table=class_table(),
                    seed=seed, storage=storage, volume_dtype=volume_dtype, **kw)
    return Path(out)


def chunk_locs(chunk_dir: Path):
    """``{(z, y, x): (volume path, points path)}`` of one chunk directory (names ``volumes_{i}_{z}_{y}_{x}``)."""
    out = {}
    for vp in G2.volume_files(chunk_dir):
        stem = vp.name.split(".")[0]
        i, z, y, x = (int(v) for v in stem.split("_")[-4:])
        pp = chunk_dir / "points" / f"points_{i}_{z}_{y}_{x}.npy"
        out[(z, y, x)] = (vp, pp)
    return out


def otf_data(chunks, augmentation=None, chunk_sampling="uniform", batch_size=2, threshold=0.1, **otf):
    from tomocpt.dataManager.otf import OnTheFlyData
    from tomocpt.defaultConfigs.otf_config import OnTheFlyConfig
    from tomocpt.defaultConfigs.scale_config import ScaleConfig
    from tomocpt.defaultConfigs.train_config import ChunkSampling

    data = OnTheFlyData(data_dir=str(chunks), return_labels=True, batch_size=batch_size, workers_for_data=0,
                        augmentation=augmentation if augmentation is not None else G2.no_spatial_aug(),
                        pipeline_version=2, scale_cfg=ScaleConfig(), chunk_sampling=ChunkSampling(chunk_sampling),
                        repeat_factor_threshold=threshold,
                        otf=OnTheFlyConfig(**otf))
    data.setup()
    return data


def as_np(t):
    return np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t)
