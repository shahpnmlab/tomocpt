"""Chunk directory naming, train/val split by tomogram, and the stale-chunk guard.

Covers the fix for chunk collisions and train/val leakage in multi-species
chunking, described in HANDOFF_chunk_collisions.md.
"""

from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pandas as pd
import pytest

from tomocpt.dataManager.chunking import (
    assign_chunk_dir_names,
    get_chunk_layout_path,
    get_chunking_name_done,
    process_single_tomo_task,
    split_rows_by_tomogram,
)
from tomocpt.defaultConfigs.train_config import TrainingModes
from tomocpt.training.helpers import _prepare_data_if_needed


def _row(tomogram_path, label_path, diameter=100.0, num_particles=10):
    return {
        "tomogram_path": tomogram_path,
        "label_path": label_path,
        "particle_diameter_angst": diameter,
        "num_particles": num_particles,
    }


# --- Gate D, item 1 -----------------------------------------------------

def test_assign_chunk_dir_names_unique_stems_is_legacy():
    df = pd.DataFrame(
        [
            _row("/data/tomoA.mrc", "/out/dataset_1_class_all/class_all/labels/tomoA.labels.mrc"),
            _row("/data/tomoB.mrc", "/out/dataset_2_class_all/class_all/labels/tomoB.labels.mrc"),
        ]
    )
    names = assign_chunk_dir_names(df)
    assert list(names) == ["tomoA", "tomoB"]


# --- Gate D, item 2 -----------------------------------------------------

def test_assign_chunk_dir_names_prefixes_on_collision():
    rows = []
    for dataset_idx in (1, 2):
        for stem in ("tomoA", "tomoB", "tomoC"):
            rows.append(
                _row(
                    f"/data/{stem}.mrc",
                    f"/out/dataset_{dataset_idx}_class_all/class_all/labels/{stem}.labels.mrc",
                )
            )
    df = pd.DataFrame(rows)
    names = assign_chunk_dir_names(df)

    assert names.is_unique
    assert len(names) == 6
    for dataset_idx in (1, 2):
        for stem in ("tomoA", "tomoB", "tomoC"):
            assert f"dataset_{dataset_idx}_class_all__{stem}" in set(names)


# --- Gate D, item 3 -----------------------------------------------------

def test_assign_chunk_dir_names_unresolvable_collision_raises():
    df = pd.DataFrame(
        [
            _row("/data/tomoA.mrc", "/out/dataset_1_class_all/class_all/labels/tomoA.labels.mrc"),
            _row("/data/tomoA.mrc", "/out/dataset_1_class_all/class_all/labels/tomoA.labels.mrc"),
        ]
    )
    with pytest.raises(ValueError):
        assign_chunk_dir_names(df)


# --- Gate D, item 4 -----------------------------------------------------

def test_split_rows_by_tomogram_unique_stems_matches_train_test_split():
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame(
        [
            _row(f"/data/tomo{i}.mrc", f"/out/dataset_1_class_all/class_all/labels/tomo{i}.labels.mrc")
            for i in range(10)
        ]
    )
    expected_train, expected_val = train_test_split(df, test_size=0.2, random_state=42)
    got_train, got_val = split_rows_by_tomogram(df, test_size=0.2, seed=42)

    assert set(got_train.index) == set(expected_train.index)
    assert set(got_val.index) == set(expected_val.index)


# --- Gate D, item 5 -----------------------------------------------------

def test_split_rows_by_tomogram_groups_multi_species_rows():
    rows = []
    for dataset_idx in range(6):
        for tomo_idx in range(7):
            stem = f"tomo{tomo_idx}"
            rows.append(
                _row(
                    f"/data/{stem}.mrc",
                    f"/out/dataset_{dataset_idx}_class_all/class_all/labels/{stem}.labels.mrc",
                )
            )
    df = pd.DataFrame(rows).reset_index(drop=True)

    train_df, val_df = split_rows_by_tomogram(df, test_size=0.2, seed=42)

    train_stems = set(train_df["tomogram_path"].map(lambda p: Path(p).stem))
    val_stems = set(val_df["tomogram_path"].map(lambda p: Path(p).stem))

    assert train_stems.isdisjoint(val_stems)
    assert len(train_df) + len(val_df) == len(df)
    assert len(val_stems) >= 1


# --- Gate D, item 6 -----------------------------------------------------

def test_split_rows_by_tomogram_single_tomogram_raises():
    df = pd.DataFrame(
        [
            _row("/data/tomoA.mrc", "/out/dataset_1_class_all/class_all/labels/tomoA.labels.mrc"),
            _row("/data/tomoA.mrc", "/out/dataset_2_class_all/class_all/labels/tomoA.labels.mrc"),
        ]
    )
    with pytest.raises(ValueError, match="distinct tomogram"):
        split_rows_by_tomogram(df)


# --- Gate D, item 7 -------------------------------------------------------

def _write_tracking_csv(dataset_dir: Path, stem: str, tomo_path: Path):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    label_path = dataset_dir / "class_all" / "labels" / f"{stem}.labels.mrc"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([_row(str(tomo_path), str(label_path))])
    df.to_csv(dataset_dir / "tracking.csv", index=False)


def test_stale_chunk_guard_raises_on_repeated_stems(tmp_path):
    training_data_dir = tmp_path / "training_data"
    tomo_path = tmp_path / "tomoA.mrc"
    tomo_path.write_text("fake")
    _write_tracking_csv(training_data_dir / "dataset_1_class_all", "tomoA", tomo_path)
    _write_tracking_csv(training_data_dir / "dataset_2_class_all", "tomoA", tomo_path)

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    Path(get_chunking_name_done(str(chunks_dir), require_labels=True)).write_text("done")

    config = SimpleNamespace(
        mode=TrainingModes.picking,
        chunks_dir=str(chunks_dir),
        training_data_dir=str(training_data_dir),
    )

    with pytest.raises(RuntimeError, match="chunk-collision bug"):
        _prepare_data_if_needed(config)


def test_stale_chunk_guard_allows_unique_stems(tmp_path):
    training_data_dir = tmp_path / "training_data"
    tomo_path_a = tmp_path / "tomoA.mrc"
    tomo_path_a.write_text("fake")
    tomo_path_b = tmp_path / "tomoB.mrc"
    tomo_path_b.write_text("fake")
    _write_tracking_csv(training_data_dir / "dataset_1_class_all", "tomoA", tomo_path_a)
    _write_tracking_csv(training_data_dir / "dataset_2_class_all", "tomoB", tomo_path_b)

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    Path(get_chunking_name_done(str(chunks_dir), require_labels=True)).write_text("done")

    config = SimpleNamespace(
        mode=TrainingModes.picking,
        chunks_dir=str(chunks_dir),
        training_data_dir=str(training_data_dir),
    )

    _prepare_data_if_needed(config)  # must not raise
    assert not Path(get_chunk_layout_path(str(chunks_dir))).is_file()


# --- Gate D, item 8 (integration, CPU only) -------------------------------

def _write_synthetic_tomogram_and_labels(tmp_path: Path, voxel_size=10.0, shape=96):
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(shape, shape, shape)).astype(np.float32)

    centers = [(24, 24, 24), (72, 72, 72), (24, 72, 48)]

    def _sphere_mask(radius_px):
        zz, yy, xx = np.meshgrid(
            np.arange(shape), np.arange(shape), np.arange(shape), indexing="ij"
        )
        mask = np.zeros((shape, shape, shape), dtype=np.float32)
        for cz, cy, cx in centers:
            dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
            mask[dist <= radius_px] = 1.0
        return mask

    for cz, cy, cx in centers:
        volume[cz - 2 : cz + 2, cy - 2 : cy + 2, cx - 2 : cx + 2] += 5.0

    tomo_path = tmp_path / "synthetic.mrc"
    with mrcfile.new(str(tomo_path), overwrite=True) as mrc:
        mrc.set_data(volume)
        mrc.voxel_size = voxel_size

    diameters_angst = {"small": 100.0, "large": 250.0}
    label_paths = {}
    for dataset_idx, (species, diameter) in enumerate(diameters_angst.items(), 1):
        radius_px = (diameter / voxel_size) / 2.0
        mask = _sphere_mask(radius_px)
        label_dir = tmp_path / f"dataset_{dataset_idx}_class_{species}" / f"class_{species}" / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / "synthetic.labels.mrc"
        with mrcfile.new(str(label_path), overwrite=True) as mrc:
            mrc.set_data(mask)
            mrc.voxel_size = voxel_size
        label_paths[species] = label_path

    return tomo_path, label_paths, diameters_angst


def test_process_single_tomo_task_two_species_no_collision(tmp_path):
    outpath = tmp_path / "train"
    outpath.mkdir()
    cache_dir = tmp_path / ".monai_cache"

    tomo_path, label_paths, diameters_angst = _write_synthetic_tomogram_and_labels(tmp_path)

    rows = []
    for species, diameter in diameters_angst.items():
        rows.append(
            pd.Series(
                _row(str(tomo_path), str(label_paths[species]), diameter=diameter)
            )
        )
    df = pd.DataFrame(rows).reset_index(drop=True)
    chunk_dir_names = assign_chunk_dir_names(df)
    assert chunk_dir_names.is_unique

    n_cubes_by_dir = {}
    for idx, row in df.iterrows():
        task = {
            "tomo_info": row,
            "outpath": str(outpath),
            "cache_dir": str(cache_dir),
            "require_labels": True,
            "desired_particle_pixel_size": 20,
            "chunk_dir_name": chunk_dir_names.loc[idx],
        }
        tomo_path_out, n_cubes = process_single_tomo_task(task)
        assert tomo_path_out is not None, "worker task failed, see logs"
        n_cubes_by_dir[chunk_dir_names.loc[idx]] = n_cubes

    out_dirs = sorted(d for d in outpath.iterdir() if d.is_dir())
    assert len(out_dirs) == 2

    counts = []
    for d in out_dirs:
        vol_files = sorted((d / "volumes").glob("volumes_*.nii.gz"))
        label_files = sorted((d / "labels_supervised").glob("labels_supervised_*.nii.gz"))
        assert len(vol_files) == len(label_files)
        assert len(vol_files) > 0
        vol_suffixes = {f.name[len("volumes_"):] for f in vol_files}
        label_suffixes = {f.name[len("labels_supervised_"):] for f in label_files}
        assert vol_suffixes == label_suffixes
        counts.append(len(vol_files))

    assert counts[0] != counts[1]
