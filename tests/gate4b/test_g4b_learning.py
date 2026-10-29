"""G4b-C7 (slow, CUDA): K = 3 folds of a synthetic two-class dataset, tiny model
(feature_size 12) trained through ``tomocpt.training.run.train`` with ``train.cv_folds = 3``,
``train.cv_fold = k``; ``build_ensemble`` on the three best checkpoints; ``predict`` on a
held-out tomogram with every fold model (its own calibration) and with the ensemble file.

Budget: three trainings of N_EPOCHS short epochs plus four GPU predictions; target
< 10 min on an RTX A6000.
Margin: the ensemble mean F1 may be at most F1_MARGIN = 0.02 below the worst fold model
(averaging fold models is expected to beat the worst member; the margin absorbs one
borderline pick of the ~25 held-out particles).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gate2 import g2_helpers as G2
from gate4 import g4_helpers as H4
from gate4b import g4b_helpers as H
from utils.tiny import applied_main_config, tiny_main_config

pytestmark = pytest.mark.slow

K = 3
APIX = 10.0
SHAPE = (96, 160, 160)
DIAMS = {1: 120.0, 2: 300.0}
NAMES = {1: "small", 2: "large"}
SPECIES = ((1, 120.0, 20, False), (2, 300.0, 5, False))
N_EPOCHS = 12
MIN_PARTICLES = 10
F1_MARGIN = 0.02


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")


def _fold_cfg(tmp, tdd, k):
    cfg = tiny_main_config(tmp / f"fold{k}")
    for sec in (cfg.train, cfg.shared):
        sec.chunks_dir = tmp / f"chunks_f{k}"
        sec.training_data_dir = tdd
    cfg.prepData.training_data_dir = tdd
    cfg.train.model_dir = tmp / f"model_f{k}"
    cfg.train.data_pipeline_version = 2
    cfg.prepData.multiclass = True
    cfg.prepData.class_diameters = ",".join(f"{c}:{d}" for c, d in DIAMS.items())
    cfg.prepData.class_names = ",".join(f"{c}:{n}" for c, n in NAMES.items())
    cfg.train.cv_folds = K
    cfg.train.cv_fold = k
    cfg.train.use_gpus = True
    cfg.train.n_epochs = N_EPOCHS
    cfg.train.batch_size = 4
    cfg.train.n_cpus_for_dataloading = 4
    cfg.train.optimizer.lr = 1e-3
    return cfg


def _best_ckpt(model_dir: Path) -> Path:
    from tomocpt.predict.calibration import read_calibration

    ckpts = sorted(Path(model_dir).rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    assert ckpts, f"no checkpoint under {model_dir}"
    calibrated = [p for p in ckpts if read_calibration(p) is not None]
    return (calibrated or ckpts)[-1]


def _f1(pred_dir, coords):
    from tomocpt.metrics import match_points, prf_beta

    out = {}
    for c, name in NAMES.items():
        picks = H.class_picks(pred_dir, name)
        radius = 0.5 * DIAMS[c] / (2.0 * APIX)
        pairs, fp, fn = match_points(picks[:, :3], np.asarray(coords[c], float), radius)
        out[name] = prf_beta(len(pairs), len(fp), len(fn), beta=1.0)["fbeta"]
    return out


def test_G4b_C7_cv_folds_and_ensemble(tmp_path, monkeypatch):
    _need_cuda()
    from tomocpt.defaultConfigs.infer_config import InferConfig
    from tomocpt.predict.ensemble import build_ensemble
    from tomocpt.training.run import train
    from utils.phantoms import make_phantom

    names = ["TS_a", "TS_b", "TS_c", "TS_d", "TS_e", "TS_f"]
    tomos = G2.phantom_tomos(names, SHAPE, APIX, SPECIES, seed0=50)
    G2.write_v2_inputs(tmp_path / "data", tomos, DIAMS)
    tdd = tmp_path / "data" / "training_data"
    held = make_phantom(shape=SHAPE, apix=APIX, species=SPECIES, wedge_deg=60.0, snr=1.0, seed=99)
    held_path = H4.write_mrc(tmp_path / "held" / "TS_held.mrc", held.volume, APIX)

    ckpts = []
    for k in range(K):
        cfg = _fold_cfg(tmp_path, tdd, k)
        with applied_main_config(cfg) as mc:
            train(mc.train)
        layout = json.loads((tmp_path / f"chunks_f{k}" / "chunk_layout.json").read_text())
        assert layout["cv"] == {"folds": K, "fold": k}
        ckpts.append(_best_ckpt(tmp_path / f"model_f{k}"))

    infer_cfg = InferConfig()
    infer_cfg.use_cuda = True
    infer_cfg.predictions_dir = tmp_path / "calib_pred"
    infer_cfg.calibration_min_particles = MIN_PARTICLES
    ens_path = tmp_path / "ensemble.json"
    with applied_main_config(None):
        build_ensemble([str(c) for c in ckpts], ens_path, infer_cfg)
    ens = json.loads(ens_path.read_text())
    assert len(ens["checkpoints"]) == K
    cal = ens["calibration"]["classes"]
    assert sorted(H.stem(p) for p in ens["calibration"]["oof_tomograms"]) == sorted(names)
    for c in DIAMS:
        e = H.entry_of(cal, c)
        if int(e["n_true"]) >= MIN_PARTICLES:
            assert e["threshold"] is not None, (c, e)
    assert any(int(H.entry_of(cal, c)["n_true"]) >= MIN_PARTICLES for c in DIAMS)

    base = dict(tomogram_file=held_path, use_cuda=True)
    scores = {}
    for k, ck in enumerate(ckpts):
        d, _ = H4.run_predict(tmp_path / f"pred_f{k}", monkeypatch, dict(base, weights=ck))
        scores[f"fold{k}"] = _f1(d, held.coords)
    d, _ = H4.run_predict(tmp_path / "pred_ens", monkeypatch, dict(base, weights=ens_path))
    scores["ensemble"] = _f1(d, held.coords)
    mean = {k: float(np.mean(list(v.values()))) for k, v in scores.items()}
    print("\nG4B_RESULT " + json.dumps({"f1": scores, "mean": mean}, sort_keys=True))
    worst = min(v for k, v in mean.items() if k != "ensemble")
    assert mean["ensemble"] >= worst - F1_MARGIN, mean
