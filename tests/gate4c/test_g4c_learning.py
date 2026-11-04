"""G4c-C8 (slow): ``tomocpt train`` with ``cv_all_folds`` end to end (real fold processes, GPU)."""

import json
from pathlib import Path

import pytest
import torch

from gate2 import g2_helpers as G2
from gate4c.test_g4c_picks import _Spy
from utils.tiny import applied_main_config, tiny_main_config

pytestmark = pytest.mark.slow

DIAMS = {1: 120.0, 2: 300.0}
NAMES = {1: "small", 2: "large"}
SPECIES = ((1, 120.0, 12, False), (2, 300.0, 3, False))
K = 3


def _cfg(tmp, tdd):
    cfg = tiny_main_config(tmp)
    for sec in (cfg.train, cfg.shared):
        sec.chunks_dir = tmp / "chunks"
        sec.training_data_dir = tdd
    cfg.prepData.training_data_dir = tdd
    cfg.prepData.multiclass = True
    cfg.prepData.class_diameters = ",".join(f"{c}:{d}" for c, d in DIAMS.items())
    cfg.prepData.class_names = ",".join(f"{c}:{n}" for c, n in NAMES.items())
    cfg.train.data_pipeline_version = 2
    cfg.train.model_dir = tmp / "models"
    cfg.train.experiment_name = "cv"
    cfg.train.cv_folds = K
    cfg.train.cv_all_folds = True
    cfg.train.cv_class_weights = "small:1,large:2"
    cfg.train.use_gpus = True
    cfg.train.n_epochs = 1
    cfg.train.batch_size = 4
    cfg.train.n_cpus_for_dataloading = 2
    cfg.infer.use_cuda = True
    cfg.infer.calibration_min_particles = 5
    return cfg


def test_G4c_C8_cv_mode_end_to_end(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from tomocpt.predict.ensemble import load_ensemble
    from tomocpt.training.run import train

    names = ["TS_a", "TS_b", "TS_c", "TS_d", "TS_e", "TS_f"]
    G2.write_v2_inputs(tmp_path / "data", G2.phantom_tomos(names, (64, 128, 128), 10.0, SPECIES, seed0=70), DIAMS)
    cfg = _cfg(tmp_path, tmp_path / "data" / "training_data")
    cv_dir = tmp_path / "models" / "cv"

    spy = _Spy(monkeypatch)
    with applied_main_config(cfg) as mc:
        train(mc.train)
    assert spy.calls == 0, "the ensemble predicted validation tomograms again"
    for k in range(K):
        assert (cv_dir / "folds" / f"fold{k}" / "done").is_file()
        assert (cv_dir / "folds" / f"fold{k}" / "checkpoints" / "weights.val_picks.npz").is_file()
    assert (tmp_path / "chunks" / "pool").is_dir()
    for f in ("ensemble.json", "oof_picks.npz", "cv_summary.md", "cv_summary.json", "peak_profiles.png"):
        assert (cv_dir / f).is_file(), f
    ens = json.loads((cv_dir / "ensemble.json").read_text())
    assert len(ens["checkpoints"]) == K
    assert sorted(Path(p).stem for p in ens["calibration"]["oof_tomograms"]) == names
    assert load_ensemble(cv_dir / "ensemble.json") is not None
    summary = json.loads((cv_dir / "cv_summary.json").read_text())
    assert set(summary["classes"]) == set(NAMES.values())
    first = ens["calibration"]["classes"]

    # the same command again trains nothing and rebuilds the same thresholds
    mtimes = {k: (cv_dir / "folds" / f"fold{k}" / "checkpoints" / "weights.ckpt").stat().st_mtime for k in range(K)}
    with applied_main_config(cfg) as mc:
        train(mc.train)
    assert spy.calls == 0
    assert {k: (cv_dir / "folds" / f"fold{k}" / "checkpoints" / "weights.ckpt").stat().st_mtime for k in range(K)} == mtimes
    assert json.loads((cv_dir / "ensemble.json").read_text())["calibration"]["classes"] == first
