"""Gate 8d: CV and calibration waste (spec 08_AGENT_PERFORMANCE.md, 8d).

* G8d-C1 a filament-only model's calibration runs no particle path (no peak finding, no true
  particle reads, no peak profiles) and writes the same filament calibration as a mixed model
  on the same maps (E0).
* G8d-C2 parallel fold processes (``training.cv.fold_parallelism``, automatic: one per visible
  GPU, pinned with ``CUDA_VISIBLE_DEVICES``) write the same fold configs as the sequential run,
  run concurrently, and give every fold its own GPU.
* OOF picks are never predicted twice when ``val_picks.npz`` exists: covered by
  ``tests/gate4c/test_g4c_picks.py::test_G4c_C3_build_ensemble_reuses_val_picks``.
"""

import threading
from pathlib import Path

import pytest
import yaml

from gate6 import g6_helpers as G6
from gate6b import test_g6b_filament_calibration as B


def test_G8d_C1_filament_only_skips_particle_path(tmp_path, monkeypatch):
    import torch

    from tomocpt.predict import calibration as C
    from tomocpt.predict import helpers as PH
    from tomocpt.predict import multiclass as MC

    entry = B._val_entry(tmp_path / "val", "a")
    mixed = B.G6.build_ckpt(tmp_path / "mix", B.PART, B.FIL)
    B._patch(monkeypatch)
    ref = C.calibrate_checkpoint(mixed, [dict(entry)], B._calib_cfg())["classes"][101]

    fil_only = G6.build_ckpt(tmp_path / "fil", (), B.FIL)
    monkeypatch.setattr(MC, "predict_class_maps", lambda model, tomo, cfg, device, classes=None, offsets_out=None:
                        ({0: torch.from_numpy(B._two_tube_map())}, 10.0))

    def _forbidden(*a, **k):
        raise AssertionError("particle path ran for a filament-only model")

    monkeypatch.setattr(PH, "_extract_peaks_from_mask", _forbidden)
    monkeypatch.setattr(C, "_true_coords", _forbidden)
    monkeypatch.setattr(C, "peak_profile_accumulate", _forbidden)
    e = dict(entry, class_available=[True])
    record = C.calibrate_checkpoint(fil_only, [e], B._calib_cfg())
    assert set(record["classes"]) == {101}
    assert record["classes"][101] == ref


class _Driver:
    def __init__(self, monkeypatch, barrier=None):
        from tomocpt.training import cv

        self.runs, self.lock, self.started = [], threading.Lock(), 0

        def _run(cmd, log_path, env=None):
            cfg_path = Path(cmd[cmd.index("--config_file") + 1])
            text = cfg_path.read_text()
            fold_cfg = yaml.safe_load(text)
            with self.lock:
                self.started += 1
                first_two = self.started <= 2
            if barrier is not None and first_two:
                barrier.wait(timeout=30)  # two folds in flight at once, or BrokenBarrierError
            with self.lock:
                self.runs.append((int(fold_cfg["train"]["cv_fold"]), text,
                                  None if env is None else env.get("CUDA_VISIBLE_DEVICES")))
            ck = Path(fold_cfg["train"]["model_dir"]) / fold_cfg["train"]["experiment_name"] / "checkpoints" / "weights.ckpt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            ck.write_bytes(b"stub")
            return 0

        monkeypatch.setattr(cv, "_run_fold_process", _run)
        monkeypatch.setattr(cv, "_build_ensemble_and_report", lambda config, checkpoints, cv_dir: None)


def test_G8d_C2_parallel_folds_match_sequential(tmp_path, monkeypatch):
    from tomocpt.training import cv

    real_run_cv = cv.run_cv

    def _run(slots, barrier=None):
        d = _Driver(monkeypatch, barrier)
        monkeypatch.setattr(cv, "_validate", lambda config: None)
        monkeypatch.setattr(cv, "prepare_cv_chunks", lambda config: [tmp_path / f"chunks{k}" for k in range(3)])
        cfg = _train_section(tmp_path / ("par" if len(slots) > 1 else "seq"))
        real_run_cv(cfg, slots=slots)
        return d

    seq = _run([None])  # one slot: sequential, environment unchanged
    par = _run(["0", "1"], barrier=threading.Barrier(2))
    by_fold = lambda d: {k: t for k, t, _ in d.runs}  # noqa: E731
    norm = lambda t: t.replace(str(tmp_path / "par"), "R").replace(str(tmp_path / "seq"), "R")  # noqa: E731
    assert sorted(by_fold(par)) == [0, 1, 2]
    assert {k: norm(t) for k, t in by_fold(par).items()} == {k: norm(t) for k, t in by_fold(seq).items()}
    assert all(g is None for _, _, g in seq.runs)
    assert {g for _, _, g in par.runs} <= {"0", "1"}


def _train_section(root):
    from tomocpt.mainConfig import MainConfig

    cfg = MainConfig()
    t = cfg.train
    t.cv_folds, t.cv_all_folds, t.use_gpus = 3, True, False
    t.chunks_dir, t.model_dir, t.experiment_name = str(root / "chunks"), str(root / "model"), "cv"
    t.data_pipeline_version = 2
    return t


def test_G8d_C2_fold_parallelism_is_automatic(monkeypatch):
    import torch

    from tomocpt.training.cv import fold_parallelism

    class _C:
        use_gpus = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert fold_parallelism(_C) == [None]  # one GPU: as before
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert fold_parallelism(_C) == ["0", "1"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert fold_parallelism(_C) == ["3", "5"]
    _C.use_gpus = False
    assert fold_parallelism(_C) == [None]
