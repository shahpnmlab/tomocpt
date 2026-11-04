"""G4c-C5: CV report numbers and peak-profile statistics."""

import json
from pathlib import Path

import numpy as np
import pytest

from gate4c import g4c_helpers as H

CLASSES = ((1, "alpha"), (2, "beta"))
RADII = (10.0, 20.0)  # native voxels


def _fbeta(tp, fp, fn, beta=4.0):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0


@pytest.fixture
def hand_made(tmp_path):
    """Two folds, one tomogram each. Class 1 (R 10, match radius 5), threshold 0.5:
    fold 0: truths (100,100,100), (200,200,200); picks (101,100,100) 0.9 [TP, d=1],
    (200,203,200) 0.6 [TP, d=3], (300,300,300) 0.8 [FP], (100,100,102) 0.4 [below threshold].
    fold 1: truth (50,50,50); picks (50,50,54) 0.7 [TP, d=4], (60,60,60) 0.55 [FP at 17.3 > 5].
    Class 2 (R 20, match radius 10), threshold 0.3: fold 0 truth (400,400,400), no pick [FN];
    fold 1 truth (80,80,80), pick (80,82,80) 0.35 [TP, d=2]."""
    picks = [(0, 0, 1, 101, 100, 100, 0.9), (0, 0, 1, 200, 203, 200, 0.6), (0, 0, 1, 300, 300, 300, 0.8),
             (0, 0, 1, 100, 100, 102, 0.4),
             (1, 1, 1, 50, 50, 54, 0.7), (1, 1, 1, 60, 60, 60, 0.55),
             (1, 1, 2, 80, 82, 80, 0.35)]
    truths = [(0, 0, 1, 100, 100, 100), (0, 0, 1, 200, 200, 200), (1, 1, 1, 50, 50, 50),
              (0, 0, 2, 400, 400, 400), (1, 1, 2, 80, 80, 80)]
    out = tmp_path / "cv"
    out.mkdir()
    H.write_oof(out / "oof_picks.npz", ["/d/TS_0.mrc", "/d/TS_1.mrc"], [0, 1], CLASSES, RADII, picks, truths)
    H.write_record(out / "ensemble.json", {1: 0.5, 2: 0.3})
    return out


def test_G4c_C5_counts_fbeta_and_per_fold(hand_made):
    from tomocpt.predict.cv_report import write_cv_report

    rep = write_cv_report(hand_made / "ensemble.json", hand_made)
    a, b = rep["classes"]["alpha"], rep["classes"]["beta"]
    assert (a["tp"], a["fp"], a["fn"]) == (3, 2, 0)
    assert (b["tp"], b["fp"], b["fn"]) == (1, 0, 1)
    assert a["n_true"] == 3 and b["n_true"] == 2
    assert a["threshold"] == pytest.approx(0.5) and b["threshold"] == pytest.approx(0.3)
    assert a["precision"] == pytest.approx(3 / 5) and a["recall"] == pytest.approx(1.0)
    assert b["precision"] == pytest.approx(1.0) and b["recall"] == pytest.approx(0.5)
    assert a["fbeta"] == pytest.approx(_fbeta(3, 2, 0))
    assert b["fbeta"] == pytest.approx(_fbeta(1, 0, 1))
    pf = {int(k): v for k, v in a["per_fold_fbeta"].items()}
    assert pf[0] == pytest.approx(_fbeta(2, 1, 0)) and pf[1] == pytest.approx(_fbeta(1, 1, 0))
    assert rep["mean_fbeta"] == pytest.approx((_fbeta(3, 2, 0) + _fbeta(1, 0, 1)) / 2)
    assert (hand_made / "cv_summary.md").is_file()
    assert json.loads((hand_made / "cv_summary.json").read_text())["mean_fbeta"] == pytest.approx(rep["mean_fbeta"])
    assert (hand_made / "peak_profiles.png").is_file()


def test_G4c_C5_weighted_mean(hand_made):
    from tomocpt.predict.cv_report import write_cv_report

    rep = write_cv_report(hand_made / "ensemble.json", hand_made, class_weights="alpha:0,beta:2")
    assert rep["mean_fbeta"] == pytest.approx(_fbeta(1, 0, 1))
    rep = write_cv_report(hand_made / "ensemble.json", hand_made, class_weights="beta:3")
    assert rep["mean_fbeta"] == pytest.approx((_fbeta(3, 2, 0) + 3 * _fbeta(1, 0, 1)) / 4)
    with pytest.raises(ValueError):
        write_cv_report(hand_made / "ensemble.json", hand_made, class_weights="gamma:1")


def test_G4c_C5_localisation(hand_made):
    from tomocpt.predict.cv_report import write_cv_report

    loc = write_cv_report(hand_made / "ensemble.json", hand_made)["classes"]["alpha"]["loc"]
    d = np.array([1.0, 3.0, 4.0])
    assert loc["mean_vox"] == pytest.approx(d.mean())
    assert loc["median_vox"] == pytest.approx(np.median(d))
    assert loc["mean_R"] == pytest.approx(d.mean() / 10.0)
    assert loc["median_R"] == pytest.approx(np.median(d) / 10.0)
    # offsets pick - truth: (1,0,0), (0,3,0), (0,0,4)
    assert loc["mean_offset_xyz"] == pytest.approx([1 / 3, 1.0, 4 / 3])


@pytest.mark.parametrize("fn, fwhm", [(H.target_profile, 0.530), (H.wide_profile, 0.777)])
def test_G4c_C5_peak_profile_stats(fn, fwhm):
    from tomocpt.predict.calibration import peak_profile_accumulate
    from tomocpt.predict.cv_report import peak_profile_stats

    R = 20.0
    centres = [(40.0, 40.0, 40.0), (100.0, 45.0, 42.0)]
    m = H.profile_map((84, 90, 144), centres, R, fn)
    acc = peak_profile_accumulate(m, np.array(centres), R)
    assert np.asarray(acc["radial_sum"]).shape == (H.N_RADIAL,)
    assert np.asarray(acc["line_sum"]).shape == (3, H.N_LINE)
    assert int(acc["n"]) == 2
    st = peak_profile_stats(acc)
    r, t = np.asarray(st["r"]), np.asarray(st["t"])
    assert r.shape == (H.N_RADIAL,) and t.shape == (H.N_LINE,)
    assert r[0] == pytest.approx(0.025) and t[0] == pytest.approx(-1.5) and t[-1] == pytest.approx(1.5)
    ok = np.asarray(acc["radial_count"]) > 0
    np.testing.assert_allclose(np.asarray(st["radial"])[ok], fn(r)[ok], atol=0.02)
    for ax in range(3):
        np.testing.assert_allclose(np.asarray(st["line"])[ax], fn(np.abs(t)), atol=0.02)
    assert st["peak_height"] == pytest.approx(1.0, abs=0.02)
    assert st["fwhm"] == pytest.approx(fwhm, abs=0.05)
    np.testing.assert_allclose(np.asarray(st["target_radial"]), H.target_profile(r), atol=1e-6)


def test_G4c_C5_profiles_pool_by_summation():
    from tomocpt.predict.calibration import peak_profile_accumulate

    R = 10.0
    c1, c2 = [(20.0, 20.0, 20.0)], [(40.0, 22.0, 21.0)]
    m = H.profile_map((44, 44, 64), c1 + c2, R)
    both = peak_profile_accumulate(m, np.array(c1 + c2), R)
    a, b = peak_profile_accumulate(m, np.array(c1), R), peak_profile_accumulate(m, np.array(c2), R)
    for k in ("radial_sum", "radial_count", "line_sum", "line_count"):
        np.testing.assert_allclose(np.asarray(both[k]), np.asarray(a[k]) + np.asarray(b[k]), atol=1e-6)
    assert float(both["peak_sum"]) == pytest.approx(float(a["peak_sum"]) + float(b["peak_sum"]))
