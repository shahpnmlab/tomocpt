"""Gate 6b: tracing threshold calibration of filament classes (spec 06b_AGENT_FILAMENT_CALIBRATION.md)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from gate6 import g6_helpers as G6


def _tube_map(shape, line_zyx_pairs, R):
    """Soft tubes: ``line_zyx_pairs`` = [(p0_xyz, p1_xyz, height), ...]; a Gaussian across each."""
    grid = np.stack(np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij"), axis=-1)
    out = np.zeros(shape, dtype=np.float32)
    for p0, p1, h in line_zyx_pairs:
        a, b = np.array(p0, float)[::-1], np.array(p1, float)[::-1]
        ab = b - a
        t = np.clip((grid - a) @ ab / (ab @ ab), 0, 1)
        d = np.linalg.norm(grid - (a + t[..., None] * ab), axis=-1)
        out = np.maximum(out, h * np.exp(-d * d / (2 * (0.5 * R) ** 2)).astype(np.float32))
    return out


STRONG = ((5.0, 12.0, 20.0), (75.0, 12.0, 20.0))
WEAK = ((5.0, 40.0, 20.0), (75.0, 40.0, 20.0))
SHAPE = (40, 56, 80)  # z, y, x


def _two_tube_map():
    return _tube_map(SHAPE, [(*STRONG, 0.95), (*WEAK, 0.4)], R=5.0)


def _cfg(**kw):
    from tomocpt.defaultConfigs.infer_config import InferConfig

    cfg = InferConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------- C1
def _brute(pred, true, tol, step=0.25):
    from benchmarks.filaments import _densify

    def cover(src, ref):
        near = tot = 0.0
        for p in src:
            d = _densify(p, step)
            mid, seg = 0.5 * (d[:-1] + d[1:]), np.linalg.norm(np.diff(d, axis=0), axis=1)
            best = np.full(len(mid), np.inf)
            for r in ref:
                for a, b in zip(r[:-1], r[1:]):
                    ab = b - a
                    t = np.clip((mid - a) @ ab / (ab @ ab), 0, 1)
                    best = np.minimum(best, np.linalg.norm(mid - (a + t[:, None] * ab), axis=1))
            tot += seg.sum()
            near += seg[best <= tol].sum()
        return near, tot

    tp_p, len_p = cover(pred, true)
    tp_t, len_t = cover(true, pred)
    return tp_p / len_p, tp_t / len_t


def test_G6b_C1_metric_is_exact_and_shared():
    import benchmarks.filaments as B
    import tomocpt.metrics as M

    assert B.centerline_scores is M.centerline_scores
    rng = np.random.default_rng(0)
    for trial in range(4):
        true = [np.cumsum(rng.normal(0, 3, (40, 3)), axis=0) for _ in range(3)]
        pred = [t + rng.normal(0, 1.5, t.shape) for t in true[:2]] + [np.cumsum(rng.normal(0, 3, (20, 3)), axis=0)]
        for tol in (1.0, 2.5, 5.0):
            s = M.centerline_scores(pred, true, tol)
            p, r = _brute(pred, true, tol)
            assert s["precision"] == pytest.approx(p, abs=1e-12)
            assert s["recall"] == pytest.approx(r, abs=1e-12)


def test_G6b_C1_metric_band_and_empty():
    from tomocpt.metrics import centerline_scores

    # Parallel lines exactly at the tolerance: a dense-only test would miss it (nodes offset).
    true = [np.array([[0.0, 0, 0], [10.0, 0, 0]])]
    pred = [np.array([[0.125, 2.0, 0], [9.875, 2.0, 0]])]
    s = centerline_scores(pred, true, 2.0)
    assert s["precision"] == pytest.approx(1.0)
    assert centerline_scores([], true, 2.0)["recall"] == 0.0
    assert centerline_scores(true, [], 2.0)["precision"] == 0.0


# ---------------------------------------------------------------------------- C2
def test_G6b_C2_scores_and_choice_on_two_tubes():
    from tomocpt.predict.calibration import (FILAMENT_GRID, calibrate_filament_thresholds,
                                             filament_threshold_scores)

    apix = 10.0
    grid = list(FILAMENT_GRID)
    assert grid == pytest.approx([0.1 * k for k in range(1, 10)])
    true = [np.linspace(STRONG[0], STRONG[1], 71)]
    cfg = _cfg(filament_min_length_ang=100.0, filament_smoothing_ang=5.0)
    s = filament_threshold_scores(_two_tube_map(), true, 100.0, apix, cfg, grid)
    assert s.shape == (len(grid), 5)
    assert np.all(s[:, 3] == pytest.approx(700.0, rel=1e-6))  # true length in A
    assert np.all(s[:, 4] == 1)
    at = {round(t, 1): i for i, t in enumerate(grid)}
    prec = lambda t: s[at[t], 0] / s[at[t], 1]
    rec = lambda t: s[at[t], 2] / s[at[t], 3]
    assert prec(0.2) < 0.7  # the weak tube is traced: false length
    assert prec(0.5) > 0.97 and rec(0.5) > 0.9
    entry = calibrate_filament_thresholds({7: [s]}, grid, beta=1.0)[7]
    assert 0.4 <= entry["threshold"] <= 0.9
    assert entry["f1"] > 0.9 and entry["kind"] == "filament" and entry["n_true"] == 1
    assert entry["true_length_ang"] == pytest.approx(700.0, rel=1e-6)
    assert len(entry["curve"]) == len(grid)


# ---------------------------------------------------------------------------- C3
def _row(tp_p, len_p, tp_t, len_t, n=1):
    return [tp_p, len_p, tp_t, len_t, n]


def test_G6b_C3_pooling_by_length_not_by_tomogram():
    from tomocpt.predict.calibration import calibrate_filament_thresholds

    grid = [0.3, 0.6]
    # thr 0.6 is perfect on a small tomogram, thr 0.3 on a tomogram 10x longer: length decides.
    t1 = np.array([_row(50, 100, 100, 100), _row(100, 100, 100, 100)], float)
    t2 = np.array([_row(1000, 1000, 1000, 1000), _row(500, 1000, 400, 1000)], float)
    e = calibrate_filament_thresholds({1: [t1, t2]}, grid, beta=1.0)[1]
    assert e["threshold"] == 0.3
    assert e["n_true"] == 2
    assert e["true_length_ang"] == 1100
    assert e["precision"] == pytest.approx(1050 / 1100)


def test_G6b_C3_tie_unavailable_beta_and_none():
    from tomocpt.predict.calibration import calibrate_filament_thresholds as cal

    grid = [0.3, 0.6]
    same = np.array([_row(90, 100, 90, 100), _row(90, 100, 90, 100)], float)
    assert cal({1: [same]}, grid, 1.0)[1]["threshold"] == 0.6  # tie: higher
    assert cal({1: [None, same]}, grid, 1.0)[1]["threshold"] == 0.6  # unavailable: skipped
    # beta: high recall at low precision wins only with a large beta
    s = np.array([_row(30, 100, 100, 100), _row(100, 100, 50, 100)], float)
    assert cal({1: [s]}, grid, 1.0)[1]["threshold"] == 0.6
    assert cal({1: [s]}, grid, 4.0)[1]["threshold"] == 0.3
    assert cal({1: [None]}, grid, 1.0)[1]["threshold"] is None
    assert cal({1: []}, grid, 1.0)[1]["threshold"] is None
    nothing = np.array([_row(0, 0, 0, 100), _row(0, 0, 0, 100)], float)
    e = cal({1: [nothing]}, grid, 1.0)[1]
    assert e["threshold"] is None and e["kind"] == "filament"


# ---------------------------------------------------------------------------- C4
def test_G6b_C4_full_region_is_not_traced(monkeypatch):
    import tomocpt.labels.filaments as LF
    from tomocpt.predict import calibration as C

    assert C.FILAMENT_MAX_REGION_FRAC == 0.25
    calls = []
    real = LF._trace_and_fit
    monkeypatch.setattr(LF, "_trace_and_fit", lambda prob, thr, *a, **k: (calls.append(round(thr, 1)), real(prob, thr, *a, **k))[1])
    prob = np.full((30, 30, 30), 0.5, dtype=np.float32)
    prob[10:20, 14:16, 14:16] = 0.9
    s = C.filament_threshold_scores(prob, [np.array([[15.0, 15, 10], [15, 15, 19]])], 40.0, 10.0,
                                    _cfg(filament_min_length_ang=20.0), [0.3, 0.7])
    assert calls == [0.7]
    assert s[0, 1] == 0.0  # no predicted length at the over-full threshold
    assert s[0, 3] > 0.0   # the true length still counts


# ---------------------------------------------------------------------------- C5
def _val_df(fil_paths):
    return pd.DataFrame({"tomogram_path": ["/t/a.mrc"], "coords_paths": [["/c.csv"]], "dataset_id": [0],
                         "apix_native": [10.0], "class_available": [[True]], "filaments_paths": [fil_paths]},
                        index=[0])


def _fil_csv(path, fid):
    pd.DataFrame({"filament_id": [fid, fid], "x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
                  "class_number": [100, 100], "diameter_ang": [250.0, 250.0]}).to_csv(path, index=False)
    return str(path)


def test_G6b_C5_manifest(tmp_path):
    from tomocpt.dataManager.chunking import _v2_val_tomograms

    names = pd.Series({0: "a"})
    assert "filaments_path" not in _v2_val_tomograms(_val_df([]), names, tmp_path)[0]
    one = _fil_csv(tmp_path / "one.csv", 1)
    assert _v2_val_tomograms(_val_df([one]), names, tmp_path)[0]["filaments_path"] == one
    two = _fil_csv(tmp_path / "two.csv", 1)
    merged = _v2_val_tomograms(_val_df([one, two]), names, tmp_path)[0]["filaments_path"]
    df = pd.read_csv(merged)
    assert len(df) == 4 and df["filament_id"].nunique() == 2
    # no filaments column at all (particle-only chunks)
    legacy = _val_df([]).drop(columns="filaments_paths")
    assert "filaments_path" not in _v2_val_tomograms(legacy, names, tmp_path)[0]


# ---------------------------------------------------------------------------- C6-C8
PART = ((1, "ball", 100.0),)
FIL = ((101, "tube", 100.0),)


def _val_entry(root, stem):
    """A validation entry: one particle row and the strong tube as the true filament."""
    import mrcfile

    root.mkdir(parents=True, exist_ok=True)
    tomo = root / f"{stem}.mrc"
    with mrcfile.new(str(tomo), overwrite=True) as m:
        m.set_data(np.zeros(SHAPE, dtype=np.float32))
        m.voxel_size = 10.0
    pts = pd.DataFrame({"x": [60.0] * 30, "y": [40.0] * 30, "z": [20.0] * 30,
                        "class_number": [1] * 30, "diameter_ang": [100.0] * 30})
    coords = root / f"{stem}_pts.csv"
    pts.to_csv(coords, index=False)
    line = np.linspace(STRONG[0], STRONG[1], 71)
    fil = root / f"{stem}_fil.csv"
    pd.DataFrame({"filament_id": 0, "x": line[:, 0], "y": line[:, 1], "z": line[:, 2],
                  "class_number": 101, "diameter_ang": 100.0}).to_csv(fil, index=False)
    return {"tomogram_path": str(tomo), "coords_path": str(coords), "filaments_path": str(fil),
            "dataset_id": 0, "apix_native": 10.0, "class_available": [True, True]}


def _oracle(calls=None):
    def predict(model, tomo, cfg, device, classes=None, offsets_out=None):
        if calls is not None:
            calls.append(str(tomo))
        ball = _tube_map(SHAPE, [((60.0, 40.0, 20.0), (60.5, 40.0, 20.0), 0.9)], R=5.0)
        return {0: torch.from_numpy(ball), 1: torch.from_numpy(_two_tube_map())}, 10.0
    return predict


@pytest.fixture()
def mixed_ckpt(tmp_path):
    return G6.build_ckpt(tmp_path / "ck", PART, FIL)


def _patch(monkeypatch, calls=None):
    from tomocpt.predict import multiclass as MC

    monkeypatch.setattr(MC, "predict_class_maps", _oracle(calls))


def _calib_cfg():
    return _cfg(use_cuda=False, filament_min_length_ang=100.0, filament_smoothing_ang=5.0,
                calibration_min_particles=1)


def test_G6b_C6_checkpoint_calibration(tmp_path, monkeypatch, mixed_ckpt):
    from tomocpt.predict.calibration import calibrate_checkpoint, read_calibration, read_val_picks, val_picks_path
    from tomocpt.predict.multiclass import read_checkpoint_settings, resolve_thresholds

    _patch(monkeypatch)
    val = [_val_entry(tmp_path / "val", "a")]
    record = calibrate_checkpoint(mixed_ckpt, val, _calib_cfg())
    fil, part = record["classes"][101], record["classes"][1]
    assert fil["kind"] == "filament" and 0.4 <= fil["threshold"] <= 0.9 and fil["f1"] > 0.9
    assert part["threshold"] is not None and part["n_true"] == 30 and "kind" not in part

    stored = read_calibration(mixed_ckpt)
    table = read_checkpoint_settings(mixed_ckpt).class_table
    thr = resolve_thresholds(table, _cfg(), stored, log=False)
    assert thr[1] == (fil["threshold"], "calibration")

    saved = read_val_picks(val_picks_path(mixed_ckpt))
    assert saved["truths"][101] == [None]  # no point calibration for filament rows
    assert len(saved["picks"][101][0][0]) == 0
    assert saved["filament_scores"][101][0].shape == (len(saved["filament_grid"]), 5)
    assert saved["filament_settings"]


def test_G6b_C6_missing_filament_file_stays_uncalibrated(tmp_path, monkeypatch, mixed_ckpt):
    from tomocpt.predict.calibration import calibrate_checkpoint

    _patch(monkeypatch)
    entry = _val_entry(tmp_path / "val", "a")
    entry.pop("filaments_path")  # chunks built before Gate 6b
    record = calibrate_checkpoint(mixed_ckpt, [entry], _calib_cfg())
    assert record["classes"][101]["threshold"] is None
    assert record["classes"][1]["threshold"] is not None


def test_G6b_C7_particle_only_unchanged(tmp_path, monkeypatch):
    from tomocpt.predict import multiclass as MC
    from tomocpt.predict.calibration import calibrate_checkpoint, picks_fingerprint, read_val_picks, val_picks_path

    fingerprint = json.loads(picks_fingerprint(_cfg()))
    assert not [k for k in fingerprint if "filament" in k]
    ckpt = G6.build_ckpt(tmp_path / "ck", PART)
    monkeypatch.setattr(MC, "predict_class_maps", lambda model, tomo, cfg, device, classes=None, offsets_out=None: (
        {0: torch.from_numpy(_tube_map(SHAPE, [((60.0, 40.0, 20.0), (60.5, 40.0, 20.0), 0.9)], R=5.0))}, 10.0))
    entry = _val_entry(tmp_path / "val", "a")
    entry.pop("filaments_path")
    entry["class_available"] = [True]
    record = calibrate_checkpoint(ckpt, [entry], _calib_cfg())
    assert set(record["classes"][1]) == {"threshold", "fbeta", "n_true"}
    saved = read_val_picks(val_picks_path(ckpt))
    assert saved["filament_scores"] == {}


def _two_fold_ckpts(tmp_path, mixed_ckpt, monkeypatch):
    import shutil

    from tomocpt.predict import ensemble as E

    ckpts = [tmp_path / "f0.ckpt", tmp_path / "f1.ckpt"]
    vals = {}
    for k, c in enumerate(ckpts):
        shutil.copy(mixed_ckpt, c)
        vals[str(c.resolve())] = [_val_entry(tmp_path / "val", f"t{k}")]
    monkeypatch.setattr(E, "_val_tomograms", lambda ckpt: vals[str(__import__("pathlib").Path(ckpt).resolve())])
    return ckpts


def test_G6b_C8_ensemble_pools_and_reports(tmp_path, monkeypatch, mixed_ckpt):
    from tomocpt.predict.cv_report import write_cv_report
    from tomocpt.predict.ensemble import build_ensemble, read_oof_picks

    _patch(monkeypatch)
    ckpts = _two_fold_ckpts(tmp_path, mixed_ckpt, monkeypatch)
    out = tmp_path / "cv" / "ensemble.json"
    record = build_ensemble(ckpts, out, _calib_cfg())
    entry = record["calibration"]["classes"]["101"]
    assert entry["kind"] == "filament" and entry["n_true"] == 2 and 0.4 <= entry["threshold"] <= 0.9
    json.loads(out.read_text())  # the record stays JSON

    oof = read_oof_picks(out)
    assert len(oof["filament_scores"][101]) == 2

    rep = write_cv_report(out, tmp_path / "rep")
    c = rep["classes"]["tube"]
    assert c["kind"] == "filament" and c["threshold"] == entry["threshold"]
    assert c["f1"] == pytest.approx(entry["f1"]) and c["precision"] > 0.9
    assert set(c["per_fold_fbeta"]) == {"0", "1"}
    assert c["tp"] is None and c["loc"]["mean_R"] is None
    text = (tmp_path / "rep" / "cv_summary.md").read_text()
    assert "Filament classes" in text and "| tube |" in text


def test_G6b_C8_stale_tracing_settings_predict_again(tmp_path, monkeypatch, mixed_ckpt):
    from tomocpt.predict.calibration import calibrate_checkpoint
    from tomocpt.predict.ensemble import build_ensemble

    calls = []
    _patch(monkeypatch, calls)
    ckpts = _two_fold_ckpts(tmp_path, mixed_ckpt, monkeypatch)
    from tomocpt.predict import ensemble as E

    cfg = _calib_cfg()
    for c in ckpts:
        calibrate_checkpoint(c, E._val_tomograms(c), cfg)
    calls.clear()
    build_ensemble(ckpts, tmp_path / "cv" / "ensemble.json", cfg)
    assert calls == []  # same settings: the saved scores are reused
    other = _calib_cfg()
    other.filament_spacing_ang = 80.0
    build_ensemble(ckpts, tmp_path / "cv" / "ensemble.json", other)
    assert len(calls) == 2
