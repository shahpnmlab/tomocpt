"""G4c-C3: validation picks saved by the calibration, reused by ``build_ensemble``; ``oof_picks.npz``.

Uses the Gate 4 ``NegInputOracle`` and the Gate 4b fold set-up (copies of one tiny checkpoint,
each with a hand-written ``chunk_layout.json`` listing one validation tomogram).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gate4 import g4_helpers as H4
from gate4b import g4b_helpers as HB
from gate4c import g4c_helpers as H
from gate4c.conftest import COND_CLASSES


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
    vals, ckpts = [], []
    for k in range(2):
        v = HB.blob_val_tomogram(tmp_path / "tomos", f"TS_fold{k}", COND_CLASSES, shift=3.0 * k)
        chunks = tmp_path / f"chunks_f{k}"
        HB.write_fold_layout(chunks, COND_CLASSES, [v["entry"]], folds=2, fold=k)
        ckpts.append(HB.fold_ckpt(src_ckpt, tmp_path / "ck" / f"fold{k}" / "weights.ckpt", chunks))
        vals.append(v)
    return ckpts, vals


class _Spy:
    """Counts ``collect_validation_picks`` calls (in calibration and in ensemble) and keeps the
    returned values and the ``profiles`` filled."""

    def __init__(self, monkeypatch):
        from tomocpt.predict import calibration

        self.calls, self.results = 0, []
        orig = calibration.collect_validation_picks

        def _wrapped(model, val, cfg, device, profiles=None):
            self.calls += 1
            out = orig(model, val, cfg, device, profiles=profiles)
            self.results.append((out, profiles))
            return out

        monkeypatch.setattr(calibration, "collect_validation_picks", _wrapped)


def _calibrate_folds(tmp_path, ckpts, vals, cfg):
    from tomocpt.predict.calibration import calibrate_checkpoint

    for ck, v in zip(ckpts, vals):
        calibrate_checkpoint(ck, [v["entry"]], cfg)


def _same_picks(a, b):
    assert set(a) == set(b)
    for c in a:
        assert len(a[c]) == len(b[c])
        for (xa, sa), (xb, sb) in zip(a[c], b[c]):
            np.testing.assert_array_equal(np.asarray(xa, float).reshape(-1, 3), np.asarray(xb, float).reshape(-1, 3))
            np.testing.assert_array_equal(np.asarray(sa, float).reshape(-1), np.asarray(sb, float).reshape(-1))


def _same_truths(a, b):
    assert set(a) == set(b)
    for c in a:
        for ta, tb in zip(a[c], b[c]):
            assert (ta is None) == (tb is None)
            if ta is not None:
                np.testing.assert_array_equal(np.asarray(ta, float).reshape(-1, 3), np.asarray(tb, float).reshape(-1, 3))


def test_G4c_C3_val_picks_round_trip(tmp_path, monkeypatch, cond_ckpt):
    from tomocpt.predict.calibration import read_val_picks, val_picks_path

    ckpts, vals = _two_folds(tmp_path, cond_ckpt["path"])
    cfg = _infer_cfg(tmp_path)
    with H4.inprocess_dask(monkeypatch), H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=True)):
        spy = _Spy(monkeypatch)
        _calibrate_folds(tmp_path, ckpts, vals, cfg)
    assert spy.calls == 2
    for ck, v, ((picks, truths, radii, grid), profiles) in zip(ckpts, vals, spy.results):
        p = val_picks_path(ck)
        assert p == Path(ck).with_name(Path(ck).stem + ".val_picks.npz")
        assert p.is_file()
        got = read_val_picks(p)
        _same_picks(got["picks"], picks)
        _same_truths(got["truths"], truths)
        assert {int(c): float(r) for c, r in got["radii"].items()} == pytest.approx({int(c): float(r) for c, r in radii.items()})
        assert list(got["grid"]) == pytest.approx(list(grid))
        assert [Path(t).name for t in got["tomograms"]] == [Path(v["entry"]["tomogram_path"]).name]
        assert np.asarray(got["available"]).shape == (1, len(COND_CLASSES))
        assert isinstance(got["fingerprint"], str) and got["fingerprint"]
        assert set(int(c) for c in got["profiles"]) == {1, 2}
        for c, acc in got["profiles"].items():
            assert int(acc["n"]) == len(v["coords"][int(c)])


def _record_core(path):
    rec = json.loads(Path(path).read_text())
    cal = rec["calibration"]
    return {"classes": cal["classes"], "oof": sorted(Path(p).name for p in cal["oof_tomograms"]),
            "grid": cal["grid"], "beta": cal["beta"]}


def test_G4c_C3_build_ensemble_reuses_val_picks(tmp_path, monkeypatch, cond_ckpt):
    from tomocpt.predict.calibration import val_picks_path
    from tomocpt.predict.ensemble import build_ensemble, read_oof_picks

    ckpts, vals = _two_folds(tmp_path, cond_ckpt["path"])
    cfg = _infer_cfg(tmp_path)
    with H4.inprocess_dask(monkeypatch), H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=True)):
        _calibrate_folds(tmp_path, ckpts, vals, cfg)
        spy = _Spy(monkeypatch)
        reused = build_ensemble([str(c) for c in ckpts], tmp_path / "a" / "ensemble.json", cfg)
        assert spy.calls == 0, "build_ensemble predicted again although the validation picks exist"

        for ck in ckpts:
            shutil.move(str(val_picks_path(ck)), str(val_picks_path(ck)) + ".bak")
        fresh = build_ensemble([str(c) for c in ckpts], tmp_path / "b" / "ensemble.json", cfg)
        assert spy.calls == 2
    assert _record_core(tmp_path / "a" / "ensemble.json") == _record_core(tmp_path / "b" / "ensemble.json")
    assert reused["calibration"]["classes"] == fresh["calibration"]["classes"]

    rec = json.loads((tmp_path / "a" / "ensemble.json").read_text())
    assert rec["oof_picks"] == "oof_picks.npz"
    assert (tmp_path / "a" / "oof_picks.npz").is_file()
    oof = read_oof_picks(tmp_path / "a" / "ensemble.json")
    assert [Path(t).stem for t in oof["tomograms"]] == ["TS_fold0", "TS_fold1"]
    assert list(oof["tomogram_fold"]) == [0, 1]
    assert {int(k): v for k, v in oof["class_names"].items()} == {n: name for n, name, _ in COND_CLASSES}
    raw = np.load(tmp_path / "a" / "oof_picks.npz")
    assert raw["picks"].shape[1] == 7 and raw["truths"].shape[1] == 6
    for k, v in enumerate(vals):
        for c in (1, 2):
            rows = raw["truths"][(raw["truths"][:, 0] == k) & (raw["truths"][:, 2] == c)]
            assert len(rows) == len(v["coords"][c])
            assert set(rows[:, 1].astype(int)) <= {k}


@pytest.mark.parametrize("change", ["fingerprint", "tomograms"])
def test_G4c_C3_stale_val_picks_are_not_reused(tmp_path, monkeypatch, cond_ckpt, change):
    from tomocpt.predict.ensemble import build_ensemble

    ckpts, vals = _two_folds(tmp_path, cond_ckpt["path"])
    cfg = _infer_cfg(tmp_path)
    with H4.inprocess_dask(monkeypatch), H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=True)):
        _calibrate_folds(tmp_path, ckpts, vals, cfg)
        if change == "fingerprint":
            cfg = _infer_cfg(tmp_path, peak_spacing_fraction=0.4)
            expect = 2
        else:
            other = HB.blob_val_tomogram(tmp_path / "tomos2", "TS_other", COND_CLASSES, shift=1.0)
            HB.write_fold_layout(tmp_path / "chunks_f1", COND_CLASSES, [other["entry"]], folds=2, fold=1)
            expect = 1
        spy = _Spy(monkeypatch)
        build_ensemble([str(c) for c in ckpts], tmp_path / "ens" / "ensemble.json", cfg)
    assert spy.calls == expect
