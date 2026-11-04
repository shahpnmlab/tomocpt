"""G4c-C1 (defaults), G4c-C6 (errors), G4c-C7 (driver and resume, fold process stubbed)."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gate2 import g2_helpers as G2
from gate4c.test_g4c_chunks import DIAMS, K, SPECIES, _cfg
from utils.tiny import applied_main_config


def test_G4c_C1_defaults():
    from tomocpt.defaultConfigs.train_config import TrainConfig

    t = TrainConfig()
    assert t.cv_all_folds is False
    assert t.cv_class_weights is None
    code = "import sys, tomocpt.training.run; print('tomocpt.training.cv' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("g4c_mode")
    tomos = G2.phantom_tomos(["TS_a", "TS_b", "TS_c", "TS_d"], (64, 96, 96), 10.0, SPECIES, seed0=60)
    G2.write_v2_inputs(root, tomos, DIAMS)
    return root / "training_data"


def _mode_cfg(tmp_path, tdd, **train):
    cfg = _cfg(tmp_path, tdd, tmp_path / "chunks", all_folds=True)
    cfg.train.model_dir = tmp_path / "models"
    cfg.train.experiment_name = "cvrun"
    for k, v in train.items():
        setattr(cfg.train, k, v)
    return cfg


@pytest.mark.parametrize("case", ["folds1", "v1", "ssl", "resume", "finetune", "too_many_folds"])
def test_G4c_C6_invalid_cv_mode_raises_before_work(tmp_path, inputs, case):
    from tomocpt.training.run import train

    kw = {"folds1": dict(cv_folds=1), "v1": dict(data_pipeline_version=1), "ssl": dict(mode="selfSupervised"),
          "resume": dict(resume_from=str(tmp_path / "x.ckpt")), "finetune": dict(fine_tune_from=str(tmp_path / "x.ckpt")),
          "too_many_folds": dict(cv_folds=9)}[case]
    cfg = _mode_cfg(tmp_path, inputs, **kw)
    if case == "ssl":
        from tomocpt.defaultConfigs.train_config import TrainingModes
        cfg.train.mode = TrainingModes.selfSupervised
    with applied_main_config(cfg) as mc, pytest.raises(ValueError):
        train(mc.train)
    assert not (tmp_path / "chunks").exists() or not any((tmp_path / "chunks").iterdir())
    assert not (tmp_path / "models" / "cvrun" / "folds").exists()


class _Driver:
    """Stubs the fold process (writes a checkpoint, or fails) and the ensemble step."""

    def __init__(self, monkeypatch, fail_fold=None):
        from tomocpt.training import cv

        self.cmds, self.ensembles, self.fail_fold = [], [], fail_fold

        def _run(cmd, log_path):
            cfg_path = Path(cmd[cmd.index("--config_file") + 1])
            fold_cfg = yaml.safe_load(cfg_path.read_text())
            self.cmds.append((cmd, fold_cfg, Path(log_path)))
            k = int(fold_cfg["train"]["cv_fold"])
            if k == self.fail_fold:
                return 3
            ck = Path(fold_cfg["train"]["model_dir"]) / fold_cfg["train"]["experiment_name"] / "checkpoints" / "weights.ckpt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            ck.write_bytes(b"stub")
            return 0

        def _ens(config, checkpoints, cv_dir):
            self.ensembles.append([Path(c) for c in checkpoints])

        monkeypatch.setattr(cv, "_run_fold_process", _run)
        monkeypatch.setattr(cv, "_build_ensemble_and_report", _ens)

    def folds_run(self):
        return [int(c[1]["train"]["cv_fold"]) for c in self.cmds]


def _run_cv(cfg):
    from tomocpt.training.run import train

    with applied_main_config(cfg) as mc, G2.load_common().inprocess_chunking():
        train(mc.train)


def test_G4c_C7_all_folds_then_ensemble(tmp_path, inputs, monkeypatch):
    d = _Driver(monkeypatch)
    cfg = _mode_cfg(tmp_path, inputs)
    _run_cv(cfg)
    cv_dir = tmp_path / "models" / "cvrun"
    assert d.folds_run() == list(range(K))
    for k, (cmd, fold_cfg, log) in enumerate(d.cmds):
        assert cmd[:4] == [sys.executable, "-m", "tomocpt.main", "train"]
        assert Path(cmd[cmd.index("--config_file") + 1]) == cv_dir / "folds" / f"fold{k}" / "config.yaml"
        t = fold_cfg["train"]
        assert t["cv_all_folds"] is False and int(t["cv_fold"]) == k and int(t["cv_folds"]) == K
        assert Path(t["model_dir"]) == cv_dir / "folds" and t["experiment_name"] == f"fold{k}"
        assert Path(t["chunks_dir"]) == tmp_path / "chunks" / f"fold{k}"
        assert int(t["data_pipeline_version"]) == 2
        assert fold_cfg["prepData"]["class_diameters"] == cfg.prepData.class_diameters
        assert log == cv_dir / "folds" / f"fold{k}" / "train.log"
        assert (cv_dir / "folds" / f"fold{k}" / "done").is_file()
    assert d.ensembles == [[cv_dir / "folds" / f"fold{k}" / "checkpoints" / "weights.ckpt" for k in range(K)]]
    meta = json.loads((cv_dir / "cv_config.json").read_text())
    assert int(meta["folds"]) == K and int(meta["seed"]) == 42
    stems = sorted(s for k in range(K) for s in meta["val_stems"][str(k)])
    assert stems == ["TS_a", "TS_b", "TS_c", "TS_d"]


def test_G4c_C7_resume_skips_done_and_reruns_unfinished(tmp_path, inputs, monkeypatch):
    cfg = _mode_cfg(tmp_path, inputs)
    with pytest.raises(RuntimeError, match=r"fold 1.*train\.log|train\.log.*fold 1"):
        _Driver(monkeypatch, fail_fold=1)
        _run_cv(cfg)
    cv_dir = tmp_path / "models" / "cvrun"
    assert (cv_dir / "folds" / "fold0" / "done").is_file()
    assert not (cv_dir / "folds" / "fold1" / "done").exists()
    stale = cv_dir / "folds" / "fold1" / "stale.txt"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("left over")

    d = _Driver(monkeypatch)
    _run_cv(cfg)
    assert d.folds_run() == [1, 2]
    assert not stale.exists(), "an unfinished fold must be cleared before it is rerun"
    assert len(d.ensembles) == 1

    d = _Driver(monkeypatch)
    _run_cv(cfg)
    assert d.folds_run() == []
    assert len(d.ensembles) == 1, "the ensemble and report are rebuilt on every run"


@pytest.mark.parametrize("change", ["folds", "seed"])
def test_G4c_C7_other_k_or_seed_in_same_cv_dir_raises(tmp_path, inputs, monkeypatch, change):
    _Driver(monkeypatch)
    _run_cv(_mode_cfg(tmp_path, inputs))
    kw = dict(cv_folds=2) if change == "folds" else dict(SEED_FOR_TRAIN_VAL_SPLIT=7)
    with pytest.raises(ValueError):
        _run_cv(_mode_cfg(tmp_path, inputs, **kw))
