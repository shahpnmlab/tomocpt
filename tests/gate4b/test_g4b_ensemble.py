"""G4b-C4 (``build_ensemble``), G4b-C5 (``EnsembleModel.predict_step``), G4b-C6 (``predict``
with an ensemble file).

C4 and C6 use the Gate 4 oracle (``NegInputOracle`` swapped into every model returned by
``InferenceModel.load_from_checkpoint``), as the Gate 4 calibration tests do: fold members
must be loaded through that public method (README).
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gate4 import g4_helpers as H4
from gate4b import g4b_helpers as H
from gate4b.conftest import ABC_CLASSES, COND_CLASSES


def _infer_cfg(tmp_path, **kw):
    from tomocpt.defaultConfigs.infer_config import InferConfig

    cfg = InferConfig()
    cfg.predictions_dir = Path(tmp_path) / "calib_pred"
    cfg.use_cuda = False
    cfg.cross_class_nms = False
    cfg.calibration_min_particles = 1
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _two_folds(tmp_path, src_ckpt):
    """Two fold checkpoints (copies of ``src_ckpt``), each with its own chunks_dir whose
    chunk_layout.json lists one validation tomogram."""
    vals, ckpts = [], []
    for k in range(2):
        v = H.blob_val_tomogram(tmp_path / "tomos", f"TS_fold{k}", COND_CLASSES, shift=3.0 * k)
        chunks = tmp_path / f"chunks_f{k}"
        H.write_fold_layout(chunks, COND_CLASSES, [v["entry"]], folds=2, fold=k)
        ckpts.append(H.fold_ckpt(src_ckpt, tmp_path / "ck" / f"fold{k}.ckpt", chunks))
        vals.append(v)
    return ckpts, vals


def test_G4b_C4_build_ensemble_pools_oof_picks(tmp_path, monkeypatch, cond_ckpt):
    from tomocpt.predict.calibration import calibrate_checkpoint
    from tomocpt.predict.ensemble import build_ensemble, is_ensemble_file, load_ensemble

    ckpts, vals = _two_folds(tmp_path, cond_ckpt["path"])
    out = tmp_path / "ens" / "ensemble.json"
    cfg = _infer_cfg(tmp_path)
    with H4.inprocess_dask(monkeypatch), H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=True)):
        build_ensemble([str(c) for c in ckpts], out, cfg)
        # reference: the Gate 4 calibration path on all OOF tomograms pooled (the fold
        # models are copies of one checkpoint, so per-fold picks equal these picks)
        ref_ck = H4.copy_ckpt(cond_ckpt["path"], tmp_path / "ck" / "ref.ckpt")
        ref = calibrate_checkpoint(ref_ck, [v["entry"] for v in vals], _infer_cfg(tmp_path / "r"))

    assert out.is_file()
    assert is_ensemble_file(out)
    assert not is_ensemble_file(ckpts[0])
    obj = json.loads(out.read_text())
    assert obj["kind"] == "tomocpt_ensemble" and int(obj["version"]) == 1
    assert [Path(p).resolve() for p in obj["checkpoints"]] == [Path(c).resolve() for c in ckpts]
    assert all(Path(p).is_absolute() for p in obj["checkpoints"])
    cal = obj["calibration"]
    assert float(cal["beta"]) == pytest.approx(4.0)
    assert {int(c) for c in cal["classes"]} == {1, 2}
    assert sorted(H.stem(p) for p in cal["oof_tomograms"]) == ["TS_fold0", "TS_fold1"]
    n_thr = 0
    for c in (1, 2):
        got, exp = H.entry_of(cal["classes"], c), H.entry_of(ref["classes"], c)
        assert int(got["n_true"]) == int(exp["n_true"]) == sum(len(v["coords"][c]) for v in vals)
        if exp["threshold"] is None:
            assert got["threshold"] is None, c
        else:
            n_thr += 1
            assert got["threshold"] == pytest.approx(exp["threshold"], abs=1e-9), c
    assert n_thr >= 1, "reference calibration found no threshold (test is vacuous)"
    assert load_ensemble(out) is not None


def test_G4b_C4_mismatched_class_tables_raise(tmp_path, monkeypatch, cond_ckpt):
    from tomocpt.predict.ensemble import build_ensemble

    ckpts, _ = _two_folds(tmp_path, cond_ckpt["path"])

    def _other_table(cfg):
        cfg.prepData.class_diameters = "1:100.0,2:600.0"
    bad = H.edit_ckpt_config(ckpts[1], tmp_path / "ck" / "fold1_bad.ckpt", _other_table)
    with H4.inprocess_dask(monkeypatch), H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=True)):
        with pytest.raises(ValueError):
            build_ensemble([str(ckpts[0]), str(bad)], tmp_path / "bad.json", _infer_cfg(tmp_path))


def test_G4b_C4_overlapping_validation_tomograms_raise(tmp_path, monkeypatch, cond_ckpt):
    from tomocpt.predict.ensemble import build_ensemble

    ckpts, vals = _two_folds(tmp_path, cond_ckpt["path"])
    # fold 1 also validates on fold 0's tomogram
    H.write_fold_layout(tmp_path / "chunks_f1", COND_CLASSES, [vals[1]["entry"], vals[0]["entry"]],
                        folds=2, fold=1)
    with H4.inprocess_dask(monkeypatch), H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=True)):
        with pytest.raises(ValueError):
            build_ensemble([str(c) for c in ckpts], tmp_path / "bad.json", _infer_cfg(tmp_path))


# ------------------------------------------------------------------------------------------
# G4b-C5
# ------------------------------------------------------------------------------------------

def _load(path):
    from tomocpt.networks.inference import InferenceModel

    return InferenceModel.load_from_checkpoint(str(path), map_location="cpu", strict=True).eval()


def _ensemble(tmp_path, paths):
    """The EnsembleModel of ``paths``: from ``load_ensemble`` on an ensemble file if it returns
    one, else ``EnsembleModel(list of loaded members)`` (README: constructor is open)."""
    from tomocpt.predict.ensemble import EnsembleModel, load_ensemble

    f = H.write_ensemble(Path(tmp_path) / f"ens_{len(paths)}.json", paths)
    obj = load_ensemble(f)
    if isinstance(obj, EnsembleModel):
        ens = obj
    elif isinstance(obj, (tuple, list)) and any(isinstance(o, EnsembleModel) for o in obj):
        ens = next(o for o in obj if isinstance(o, EnsembleModel))
    elif isinstance(obj, dict) and any(isinstance(o, EnsembleModel) for o in obj.values()):
        ens = next(o for o in obj.values() if isinstance(o, EnsembleModel))
    else:
        ens = EnsembleModel([_load(p) for p in paths])
    if hasattr(ens, "cpu"):
        ens = ens.cpu()
    if hasattr(ens, "eval"):
        ens.eval()
    return ens


def _softmax_copy(src, dst):
    def _edit(cfg):
        cfg.train.particles.class_head = "softmax"
    return H.edit_ckpt_config(src, dst, _edit)


@pytest.mark.parametrize("head", ["sigmoid", "softmax"])
def test_G4b_C5_ensemble_is_the_mean_of_members(tmp_path, abc_ckpts, head):
    paths = [i["path"] for i in abc_ckpts]
    if head == "softmax":
        paths = [_softmax_copy(p, tmp_path / f"sm{i}.ckpt") for i, p in enumerate(paths)]
    members = [_load(p) for p in paths]
    ens = _ensemble(tmp_path, paths)
    x = abc_ckpts[0]["x"]
    with torch.no_grad():
        for classes in (None, [0, 2]):
            outs = [m.predict_step(x, apix=None, classes=classes).float() for m in members]
            assert float((outs[0] - outs[1]).abs().max()) > 1e-4, "members are identical (vacuous)"
            got = ens.predict_step(x, apix=None, classes=classes).float()
            assert got.shape == outs[0].shape
            np.testing.assert_allclose(got.numpy(), torch.stack(outs).mean(0).numpy(), rtol=0, atol=1e-6)


@pytest.mark.parametrize("head", ["sigmoid", "softmax"])
def test_G4b_C5_one_member_equals_the_model(tmp_path, abc_ckpts, cond_ckpt, head):
    p = abc_ckpts[1]["path"]
    if head == "softmax":
        p = _softmax_copy(p, tmp_path / "sm.ckpt")
    single, ens = _load(p), _ensemble(tmp_path, [p])
    x = abc_ckpts[1]["x"]
    with torch.no_grad():
        np.testing.assert_allclose(ens.predict_step(x).float().numpy(), single.predict_step(x).float().numpy(),
                                   rtol=0, atol=1e-6)
    # a scale-conditioned member gets apix
    single, ens = _load(cond_ckpt["path"]), _ensemble(tmp_path / "c", [cond_ckpt["path"]])
    a = torch.full((1, 1), 20.0)
    with torch.no_grad():
        np.testing.assert_allclose(ens.predict_step(cond_ckpt["x"], apix=a).float().numpy(),
                                   single.predict_step(cond_ckpt["x"], apix=a).float().numpy(), rtol=0, atol=1e-6)


# ------------------------------------------------------------------------------------------
# G4b-C6
# ------------------------------------------------------------------------------------------

def test_G4b_C6_predict_with_ensemble_file(tmp_path, monkeypatch, cond_ckpt):
    ckpt = H4.copy_ckpt(cond_ckpt["path"], tmp_path / "ck" / "m.ckpt")
    v = H.blob_val_tomogram(tmp_path / "tomos", "TS_test", COND_CLASSES)
    tomo, coords = v["path"], v["coords"]
    base = dict(tomogram_file=tomo, cross_class_nms=False, confidence_threshold=0.6)
    oracle = H4.neg_oracle_factory(True)

    # single checkpoint with explicit thresholds
    d_ck, _ = H4.run_predict(tmp_path / "ck_run", monkeypatch,
                             dict(base, weights=ckpt, class_thresholds="s100:0.3,l500:0.3"), oracle=oracle)
    # one-member ensemble, the same thresholds from its calibration
    ens = H.write_ensemble(tmp_path / "e1" / "ensemble.json", [ckpt], {1: 0.3, 2: 0.3})
    d_en, _ = H4.run_predict(tmp_path / "en_run", monkeypatch, dict(base, weights=ens), oracle=oracle)
    for name, c in (("s100", 1), ("l500", 2)):
        assert (d_en / f"tomocpt_coords_{name}.star").is_file(), name
        a, b = H.sort_rows(H.class_picks(d_ck, name)), H.sort_rows(H.class_picks(d_en, name))
        assert len(a) >= len(coords[c]), name
        assert a.shape == b.shape, name
        np.testing.assert_allclose(b[:, :3], a[:, :3], rtol=0, atol=1e-4)
        np.testing.assert_allclose(b[:, 3], a[:, 3], rtol=0, atol=1e-5)
    # the same thresholds through class_thresholds on the ensemble
    d_en2, _ = H4.run_predict(tmp_path / "en_run2", monkeypatch,
                              dict(base, weights=ens, class_thresholds="s100:0.3,l500:0.3"), oracle=oracle)
    for name in ("s100", "l500"):
        np.testing.assert_allclose(H.sort_rows(H.class_picks(d_en2, name)), H.sort_rows(H.class_picks(d_ck, name)),
                                   rtol=0, atol=1e-4)


def test_G4b_C6_class_thresholds_override_ensemble_calibration(tmp_path, monkeypatch, cond_ckpt):
    ckpt = H4.copy_ckpt(cond_ckpt["path"], tmp_path / "ck" / "m.ckpt")
    v = H.blob_val_tomogram(tmp_path / "tomos", "TS_test", COND_CLASSES)
    ens = H.write_ensemble(tmp_path / "e" / "ensemble.json", [ckpt, ckpt], {1: 0.95, 2: 0.05})
    base = dict(weights=ens, tomogram_file=v["path"], cross_class_nms=False, confidence_threshold=0.3)
    oracle = H4.neg_oracle_factory(True)
    d1, _ = H4.run_predict(tmp_path / "r1", monkeypatch, base, oracle=oracle)
    assert len(H.class_picks(d1, "s100")) == 0       # ensemble calibration 0.95 (oracle peaks 0.5)
    assert len(H.class_picks(d1, "l500")) >= 1
    d2, _ = H4.run_predict(tmp_path / "r2", monkeypatch, dict(base, class_thresholds="s100:0.3"), oracle=oracle)
    assert len(H.class_picks(d2, "s100")) >= len(v["coords"][1])
