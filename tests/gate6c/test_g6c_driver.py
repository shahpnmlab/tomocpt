"""Gate 6c: fixed validation split, filament driver recipe, pictures (spec 06c_AGENT_FILAMENT_DRIVER.md)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _df():
    return pd.DataFrame({"tomogram_path": [f"/d/{s}.mrc" for s in ("a", "b", "b", "c")], "v": range(4)})


def test_G6c_C1_split_by_stems():
    from tomocpt.dataManager.chunking import split_rows_by_stems

    tr, va = split_rows_by_stems(_df(), "b")
    assert list(va["v"]) == [1, 2] and list(tr["v"]) == [0, 3]
    tr, va = split_rows_by_stems(_df(), " a, c ")
    assert list(va["v"]) == [0, 3] and list(tr["v"]) == [1, 2]
    with pytest.raises(ValueError, match="zzz"):
        split_rows_by_stems(_df(), "a,zzz")
    with pytest.raises(ValueError):
        split_rows_by_stems(_df(), "a,b,c")
    with pytest.raises(ValueError):
        split_rows_by_stems(_df(), " , ")


def _layout_cfg(tmp_path, val_stems, cv_folds=0):
    from types import SimpleNamespace

    done = tmp_path / "done_labels_True.txt"
    done.write_text("x")
    return SimpleNamespace(chunks_dir=str(tmp_path), data_pipeline_version=2, cv_folds=cv_folds, cv_fold=0,
                           val_stems=val_stems, mode="picking")


def test_G6c_C2_default_and_refusal(tmp_path, monkeypatch):
    from tomocpt.defaultConfigs.train_config import TrainConfig
    from tomocpt.training import helpers as H

    assert TrainConfig().val_stems is None
    assert H._requested_val_stems(_layout_cfg(tmp_path, None)) is None
    assert H._requested_val_stems(_layout_cfg(tmp_path, "b, a")) == ["a", "b"]
    with pytest.raises(ValueError, match="cv_folds"):
        H._requested_val_stems(_layout_cfg(tmp_path, "a", cv_folds=3))
    with pytest.raises(ValueError, match="pipeline"):
        cfg = _layout_cfg(tmp_path, "a")
        cfg.data_pipeline_version = 1
        H._requested_val_stems(cfg)
    # recorded != requested
    H._check_layout_val_stems({"val_stems": ["a"]}, ["a"], "x")
    H._check_layout_val_stems({}, None, "x")
    with pytest.raises(RuntimeError, match="val_stems"):
        H._check_layout_val_stems({"val_stems": ["a"]}, ["b"], "x")
    with pytest.raises(RuntimeError, match="val_stems"):
        H._check_layout_val_stems({}, ["b"], "x")
    with pytest.raises(RuntimeError, match="val_stems"):
        H._check_layout_val_stems({"val_stems": ["a"]}, None, "x")


def test_G6c_C3_recipe():
    from benchmarks.filaments import apply_recipe
    from tomocpt.defaultConfigs.train_config import TrainConfig

    train = apply_recipe({"augmentation": {"label_interpolation": "linear"}, "particles": {"class_head": "softmax"}})
    assert train["chunk_sampling"] == "repeat_factor"
    assert train["augmentation"] == {"label_interpolation": "linear", "rotation_mode": "flips", "flip_p": 0.5}
    assert train["particles"] == {"class_head": "softmax", "cross_class_neg_weight": 0.0}
    assert train["early_stop_rel_delta"] == 0.007 and train["early_stop_patience"] == 10
    assert train["lr_plateau_match_early_stop"] is True and train["lr_plateau_patience"] == 5
    assert train["weight_averaging"] == "ema"
    cfg = TrainConfig()
    for k in ("chunk_sampling", "early_stop_rel_delta", "early_stop_patience", "lr_plateau_match_early_stop",
              "lr_plateau_patience", "weight_averaging"):
        assert hasattr(cfg, k)


def _data_root(tmp_path, runs=("s1", "s2", "s3"), objects=("mt",)):
    root = tmp_path / "data"
    (root / "9" / "tomograms").mkdir(parents=True)
    for r in runs:
        (root / "9" / "tomograms" / f"{r}.mrc").write_text("t")
        for o in objects:
            (root / "9" / "masks" / o).mkdir(parents=True, exist_ok=True)
            (root / "9" / "masks" / o / f"{r}.mrc").write_text("m")
    return root


def test_G6c_C4_stage_fixed(tmp_path):
    from benchmarks.filaments import _stage_fixed

    root = _data_root(tmp_path)
    out = tmp_path / "stage"
    staged = _stage_fixed(root, "9", ["mt"], out, ["s1", "s2"], ["s3"])
    assert [e["name"] for e in staged["train"]] == ["9_s1", "9_s2"]
    assert [e["name"] for e in staged["test"]] == ["9_s3"]
    assert sorted(p.name for p in (out / "train" / "tomograms").iterdir()) == ["9_s1.mrc", "9_s2.mrc", "9_s3.mrc"]
    assert (out / "train" / "masks" / "mt" / "9_s3.mrc").is_file()
    with pytest.raises(ValueError):
        _stage_fixed(root, "9", ["mt"], tmp_path / "o2", ["s1"], ["s1"])
    with pytest.raises(FileNotFoundError):
        _stage_fixed(root, "9", ["mt"], tmp_path / "o3", ["s1"], ["nope"])


def test_G6c_C5_classify_and_target():
    from benchmarks.filament_views import classify_traces, target_map

    true = [np.array([[0.0, 10, 5], [40.0, 10, 5]]), np.array([[0.0, 30, 5], [40.0, 30, 5]])]
    pred = [np.array([[0.0, 11, 5], [40.0, 11, 5]]), np.array([[0.0, 50, 5], [40.0, 50, 5]])]
    matched, false, missed = classify_traces(pred, true, tol_vox=3.0)
    length = lambda ps: sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in ps)
    assert length(matched) == pytest.approx(40.0)
    assert length(false) == pytest.approx(40.0)
    assert length(missed) == pytest.approx(40.0)
    assert np.all(np.abs(np.concatenate(missed)[:, 1] - 30) < 1e-9)
    # a trace that leaves a true centreline half way is split
    bent = [np.array([[0.0, 10, 5], [20.0, 10, 5], [20.0, 45, 5]])]
    m, f, _ = classify_traces(bent, true[:1], tol_vox=3.0)
    assert length(m) > 20 and length(f) > 25

    t = target_map(true[:1], (10, 40, 45), 60.0, 10.0)
    assert t.shape == (10, 40, 45)
    assert t[5, 10, 20] == pytest.approx(1.0, abs=1e-5)
    assert t[5, 30, 20] == 0.0


def test_G6c_C5_pictures_are_written(tmp_path):
    from benchmarks.filament_views import save_filament_view, save_threshold_curve, target_map

    rng = np.random.default_rng(0)
    shape = (20, 40, 60)
    true = [np.array([[5.0, 20, 10], [55.0, 20, 10]])]
    tgt = target_map(true, shape, 60.0, 10.0)
    p = save_filament_view(tmp_path / "v" / "view.png", rng.normal(size=shape), tgt, 0.8 * tgt,
                           [true[0] + 1, np.array([[5.0, 35, 10], [40.0, 35, 10]])], true, 3.0, "title")
    assert p.is_file() and p.read_bytes()[:4] == b"\x89PNG"
    # nothing traced, nothing true: still a picture
    p = save_filament_view(tmp_path / "v" / "empty.png", rng.normal(size=shape), 0 * tgt, 0 * tgt, [], [], 3.0, "t")
    assert p.is_file()
    entry = {"threshold": 0.5, "curve": [[0.1, 0.2, 0.9, 0.3], [0.5, 0.8, 0.8, 0.8]]}
    assert save_threshold_curve(tmp_path / "v" / "c.png", entry, "t").is_file()
    assert save_threshold_curve(tmp_path / "v" / "c0.png", {"threshold": None, "curve": []}, "t").is_file()


def test_G6c_C6_dump_training_images(tmp_path):
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from benchmarks.filament_views import dump_training_images

    w = SummaryWriter(str(tmp_path / "logs" / "v0"))
    for step in range(3):
        w.add_image("input_val", torch.rand(1, 8, 8), global_step=step)
        w.add_image("a/b", torch.rand(1, 8, 8), global_step=step)
    w.add_image("once", torch.rand(1, 8, 8), global_step=0)
    w.close()
    paths = dump_training_images(tmp_path / "logs", tmp_path / "out")
    names = sorted(p.name for p in paths)
    assert names == ["a_b_epoch000.png", "a_b_epoch002.png", "input_val_epoch000.png", "input_val_epoch002.png",
                     "once_epoch000.png"]
    assert all(p.read_bytes()[:4] == b"\x89PNG" for p in paths)
    assert dump_training_images(tmp_path / "nothing", tmp_path / "out2") == []
