"""G4-C10 calibrate_thresholds, G4-C11 calibration round trip.

Formats chosen where the spec is silent (README): ``picks``, ``truths`` and ``radii`` are dicts
keyed by class number; ``picks[c]`` is a list per tomogram of ``(xyz [N, 3], score [N])``;
``truths[c]`` a list per tomogram of ``xyz [M, 3]`` or ``None``; ``radii[c]`` the particle
radius in the unit of the coordinates. Test scores never sit on a grid value, so ``>=`` and
``>`` give the same result.
"""

import numpy as np
import pytest

from gate4 import g4_helpers as H


def _grid_points(n, spacing=40.0, offset=0.0):
    k = int(np.ceil(n ** (1 / 3)))
    pts = [(i, j, l) for i in range(k) for j in range(k) for l in range(k)][:n]
    return np.asarray(pts, dtype=np.float64) * spacing + offset


def _picks(true_xyz, true_score, false_xyz, false_score):
    xyz = np.concatenate([true_xyz, false_xyz]).reshape(-1, 3)
    s = np.concatenate([np.full(len(true_xyz), true_score), np.full(len(false_xyz), false_score)])
    return (xyz, s)


def _cal(picks, truths, radii, **kw):
    from tomocpt.predict.calibration import calibrate_thresholds

    return calibrate_thresholds(picks, truths, radii, **kw)


def _entry(res, c):
    return res[c] if c in res else res[str(c)]


def _two_score_case(n_true=30, s_true=0.7, s_false=0.4):
    t = _grid_points(n_true)
    f = _grid_points(n_true, offset=5000.0)
    return {1: [_picks(t, s_true, f, s_false)]}, {1: [t]}, {1: 6.0}


def test_G4_C10_true_07_false_04_threshold_in_between():
    picks, truths, radii = _two_score_case()
    e = _entry(_cal(picks, truths, radii), 1)
    assert e["threshold"] is not None
    assert 0.4 < e["threshold"] <= 0.7 + 1e-9
    assert e["fbeta"] == pytest.approx(1.0)
    assert int(e["n_true"]) == 30


def test_G4_C10_few_particles_get_none():
    picks, truths, radii = _two_score_case(n_true=10)
    e = _entry(_cal(picks, truths, radii, min_particles=20), 1)
    assert e["threshold"] is None
    assert int(e["n_true"]) == 10


def test_G4_C10_unavailable_tomogram_adds_no_false_positives():
    picks, truths, radii = _two_score_case()
    alone = _entry(_cal(picks, truths, radii), 1)
    extra = _grid_points(20, offset=9000.0)
    picks[1].append((extra, np.full(len(extra), 0.9)))  # high-scoring picks, class unavailable
    truths[1].append(None)
    both = _entry(_cal(picks, truths, radii), 1)
    assert both["fbeta"] == pytest.approx(1.0)
    assert both["threshold"] == pytest.approx(alone["threshold"])
    assert int(both["n_true"]) == 30


def test_G4_C10_ties_pick_the_higher_threshold():
    picks, truths, radii = _two_score_case(s_true=0.87, s_false=0.1)
    e = _entry(_cal(picks, truths, radii), 1)
    assert e["threshold"] == pytest.approx(0.85, abs=1e-6)
    e = _entry(_cal(picks, truths, radii, grid=[0.3, 0.5, 0.6]), 1)
    assert e["threshold"] == pytest.approx(0.6, abs=1e-9)


def test_G4_C10_matches_a_reference_sweep():
    """Two classes, two tomograms, spread scores: threshold and F-beta equal a sweep with
    ``tomocpt.metrics`` (match radius 0.5 * radius, beta 4, ties -> higher threshold)."""
    from tomocpt.metrics import match_points, prf_beta

    rs = np.random.RandomState(5)
    picks, truths, radii = {}, {}, {1: 8.0, 2: 14.0}
    for c in (1, 2):
        picks[c], truths[c] = [], []
        for t in range(2):
            tx = _grid_points(25, spacing=60.0, offset=1000.0 * t + 100.0 * c)
            st = rs.uniform(0.2, 0.95, len(tx)).round(2) + 0.003
            fx = tx[:15] + 200.0 + rs.uniform(0, 1, (15, 3))
            sf = rs.uniform(0.05, 0.6, 15).round(2) + 0.003
            jitter = rs.uniform(-1, 1, tx.shape)
            picks[c].append((np.concatenate([tx + jitter, fx]), np.concatenate([st, sf])))
            truths[c].append(tx)
    res = _cal(picks, truths, radii, beta=4.0, min_particles=20)
    grid = [round(0.05 * k, 2) for k in range(1, 20)]
    for c in (1, 2):
        best = None
        for thr in grid:
            tp = fp = fn = 0
            for (xyz, s), tx in zip(picks[c], truths[c]):
                keep = s >= thr
                pairs, fpi, fni = match_points(xyz[keep], tx, 0.5 * radii[c])
                tp, fp, fn = tp + len(pairs), fp + len(fpi), fn + len(fni)
            f = prf_beta(tp, fp, fn, beta=4.0)["fbeta"]
            if best is None or f >= best[1] - 1e-12:
                best = (thr, f)
        e = _entry(res, c)
        assert e["threshold"] == pytest.approx(best[0], abs=1e-6), c
        assert e["fbeta"] == pytest.approx(best[1], abs=1e-9), c
        assert int(e["n_true"]) == 50


# ------------------------------------------------------------------------------------------
# G4-C11
# ------------------------------------------------------------------------------------------

def _val_tomogram(tmp_path, ckpt_classes):
    """The C3 phantom (two_level classes) written with a Gate 2 style val_tomograms entry."""
    import pandas as pd

    from gate4.test_g4_roundtrip import two_level_phantom

    vol, coords, apix = two_level_phantom()
    tomo = H.write_mrc(tmp_path / "val" / "TS_val.mrc", vol, apix)
    diam = {n: d for n, _, d in ckpt_classes}
    rows = [dict(x=x, y=y, z=z, class_number=c, diameter_ang=diam[c]) for c, xyz in coords.items() for x, y, z in xyz]
    csv = tmp_path / "val" / "TS_val.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    entry = dict(tomogram_path=str(tomo), coords_path=str(csv), dataset_id=0, apix_native=apix,
                 class_available=[True] * len(ckpt_classes))
    return entry, tomo, coords


def _infer_cfg(tmp_path, **kw):
    from tomocpt.defaultConfigs.infer_config import InferConfig

    cfg = InferConfig()
    cfg.predictions_dir = tmp_path / "calib_pred"
    cfg.use_cuda = False
    cfg.cross_class_nms = False
    cfg.calibration_min_particles = 1
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _class_picks(pred_dir, name):
    p = pred_dir / f"tomocpt_coords_{name}.star"
    if not p.is_file():
        return np.zeros((0, 4))
    return np.concatenate(list(H.picks_by_class(p).values()) or [np.zeros((0, 4))])


def test_G4_C11_calibration_round_trip(tmp_path, monkeypatch, v2_ckpts):
    import torch

    from tomocpt.networks.inference import InferenceModel
    from tomocpt.predict.calibration import calibrate_checkpoint, read_calibration

    info = v2_ckpts("two_level_cond")
    classes = info["classes"]
    ckpt = H.copy_ckpt(info["path"], tmp_path / "ck" / "v2.ckpt")
    plain = H.copy_ckpt(info["path"], tmp_path / "ck" / "v2_uncalibrated.ckpt")
    before = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert read_calibration(plain) is None
    entry, tomo, coords = _val_tomogram(tmp_path, classes)

    with H.inprocess_dask(monkeypatch), H.oracle_models(monkeypatch, H.neg_oracle_factory(conditioned=True)):
        res = calibrate_checkpoint(ckpt, [entry], _infer_cfg(tmp_path))

    after = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "tomocpt_calibration" in after
    cal = after["tomocpt_calibration"]
    assert cal["version"] == 1
    assert float(cal["beta"]) == pytest.approx(4.0)
    assert len(cal["grid"]) > 0 and min(cal["grid"]) == pytest.approx(0.05)
    assert {int(k) for k in cal["classes"]} == {1, 2}
    assert "val_tomograms" in cal
    for c in (1, 2):
        e = _entry(cal["classes"], c)
        assert {"threshold", "fbeta", "n_true"} <= set(e)
        assert int(e["n_true"]) == len(coords[c])
    assert set(after) - {"tomocpt_calibration"} == set(before)
    for k in before:
        assert H.deep_equal(before[k], after[k]), f"checkpoint key {k!r} changed"
    rc = read_calibration(ckpt)
    assert rc is not None and H.deep_equal(rc["classes"], cal["classes"])
    assert isinstance(res, dict)

    # still loads strictly and gives the same logits
    g3 = H.load_g3_helpers()
    model = InferenceModel.load_from_checkpoint(str(ckpt), map_location="cpu", strict=True).eval()
    with torch.no_grad():
        for a, ref in info["logits"].items():
            got = model(info["x"], apix=g3.apix_tensor(a)) if a is not None else model(info["x"])
            torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)

    # predict uses the calibrated thresholds: s100 -> 0.95 (no picks), l500 -> 0.05
    for k in list(after["tomocpt_calibration"]["classes"]):
        after["tomocpt_calibration"]["classes"][k]["threshold"] = 0.95 if int(k) == 1 else 0.05
    torch.save(after, ckpt)
    base = dict(weights=ckpt, tomogram_file=tomo, cross_class_nms=False, confidence_threshold=0.3)
    pd1, _ = H.run_predict(tmp_path / "r1", monkeypatch, base, oracle=H.neg_oracle_factory(True))
    assert len(_class_picks(pd1, "s100")) == 0
    assert len(_class_picks(pd1, "l500")) >= 1

    # class_thresholds overrides the calibration
    pd2, _ = H.run_predict(tmp_path / "r2", monkeypatch, dict(base, class_thresholds="s100:0.3"),
                           oracle=H.neg_oracle_factory(True))
    assert len(_class_picks(pd2, "s100")) >= len(coords[1])

    # unknown class name -> ValueError naming it
    with pytest.raises(ValueError, match="notaclass"):
        H.run_predict(tmp_path / "r3", monkeypatch, dict(base, class_thresholds="notaclass:0.3"),
                      oracle=H.neg_oracle_factory(True))

    # no calibration key -> confidence_threshold
    pd4, _ = H.run_predict(tmp_path / "r4", monkeypatch, dict(base, weights=plain, confidence_threshold=0.6),
                           oracle=H.neg_oracle_factory(True))
    assert len(_class_picks(pd4, "s100")) == 0 and len(_class_picks(pd4, "l500")) == 0
    pd5, _ = H.run_predict(tmp_path / "r5", monkeypatch, dict(base, weights=plain),
                           oracle=H.neg_oracle_factory(True))
    assert len(_class_picks(pd5, "s100")) >= len(coords[1]) and len(_class_picks(pd5, "l500")) >= 1
