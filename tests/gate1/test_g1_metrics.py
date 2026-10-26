"""G1-C10: picking metrics (``tomocpt.metrics``) on hand-made cases.

Return-structure assumptions (the spec gives names and argument lists only):
* ``match_points`` -> ``(tp_pairs, fp_idx, fn_idx)``; ``tp_pairs`` is anything
  ``np.asarray(...).reshape(-1, 2)`` turns into ``(pred_index, true_index)`` rows; the index
  collections are anything ``np.asarray(...).ravel()`` turns into integer indices.
* ``score_classes`` -> dict with the per-class metric dicts either at ``result[name]`` or at
  ``result["per_class"][name]``, and the macro F-beta at ``result["macro"]["fbeta"]``,
  ``result["macro"]`` (a float), or ``result["macro_fbeta"]``.
"""

import math

import numpy as np
import pytest


def _match(pred, true, radius):
    from tomocpt.metrics import match_points

    pred = np.asarray(pred, dtype=float).reshape(-1, 3)
    true = np.asarray(true, dtype=float).reshape(-1, 3)
    tp, fp, fn = match_points(pred, true, radius)
    tp = [tuple(int(i) for i in row) for row in np.asarray(tp, dtype=int).reshape(-1, 2)]
    fp = sorted(int(i) for i in np.asarray(fp, dtype=int).ravel())
    fn = sorted(int(i) for i in np.asarray(fn, dtype=int).ravel())
    return sorted(tp), fp, fn


def _per_class(res, name):
    if isinstance(res.get("per_class"), dict) and name in res["per_class"]:
        return res["per_class"][name]
    return res[name]


def _macro_fbeta(res):
    m = res.get("macro", res.get("macro_fbeta"))
    if isinstance(m, dict):
        return float(m["fbeta"])
    return float(m)


def _fbeta(p, r, beta):
    return 0.0 if p + r == 0 else (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


T = [[10, 10, 10], [50, 50, 50], [90, 20, 40]]


def test_G1_C10_match_exact():
    tp, fp, fn = _match(T, T, radius=5.0)
    assert tp == [(0, 0), (1, 1), (2, 2)] and fp == [] and fn == []


def test_G1_C10_match_one_miss_and_one_false_positive():
    pred = [[11, 10, 10], [90, 21, 40]]              # misses truth 1
    tp, fp, fn = _match(pred, T, radius=5.0)
    assert tp == [(0, 0), (1, 2)] and fp == [] and fn == [1]

    pred = T + [[200, 200, 200]]                       # one extra
    tp, fp, fn = _match(pred, T, radius=5.0)
    assert tp == [(0, 0), (1, 1), (2, 2)] and fp == [3] and fn == []


def test_G1_C10_match_just_outside_radius():
    tp, fp, fn = _match([[10 + 5.01, 10, 10]], [[10, 10, 10]], radius=5.0)
    assert tp == [] and fp == [0] and fn == [0]
    tp, fp, fn = _match([[10 + 4.99, 10, 10]], [[10, 10, 10]], radius=5.0)
    assert tp == [(0, 0)] and fp == [] and fn == []


def test_G1_C10_match_two_predictions_compete_for_one_truth():
    pred = [[13, 10, 10], [11, 10, 10]]               # pred 1 is closer
    tp, fp, fn = _match(pred, [[10, 10, 10]], radius=5.0)
    assert tp == [(1, 0)] and fp == [0] and fn == []


def test_G1_C10_match_is_greedy_by_increasing_distance():
    # pred 0 is 1 from truth 0 and 3 from truth 1; pred 1 is 2 from truth 0, 6 from truth 1.
    # Greedy by distance: (p0, t0) first (d=1); p1 then cannot reach t1 within r=5 -> FP, t1 FN.
    true = [[0, 0, 0], [4, 0, 0]]
    pred = [[1, 0, 0], [-2, 0, 0]]
    tp, fp, fn = _match(pred, true, radius=5.0)
    assert tp == [(0, 0)] and fp == [1] and fn == [1]


def test_G1_C10_match_empty_inputs():
    assert _match(np.zeros((0, 3)), np.zeros((0, 3)), 5.0) == ([], [], [])
    assert _match(np.zeros((0, 3)), T, 5.0) == ([], [], [0, 1, 2])
    assert _match(T, np.zeros((0, 3)), 5.0) == ([], [0, 1, 2], [])


def test_G1_C10_prf_beta():
    from tomocpt.metrics import prf_beta

    r = prf_beta(3, 1, 2, beta=4.0)
    p, rc = 3 / 4, 3 / 5
    assert r["precision"] == pytest.approx(p)
    assert r["recall"] == pytest.approx(rc)
    assert r["f1"] == pytest.approx(2 * p * rc / (p + rc))
    assert r["fbeta"] == pytest.approx(_fbeta(p, rc, 4.0))
    assert prf_beta(3, 1, 2)["fbeta"] == pytest.approx(_fbeta(p, rc, 4.0))  # beta defaults to 4

    perfect = prf_beta(5, 0, 0)
    assert all(perfect[k] == pytest.approx(1.0) for k in ("precision", "recall", "f1", "fbeta"))

    for counts in [(0, 0, 0), (0, 3, 0), (0, 0, 3), (0, 2, 2)]:
        z = prf_beta(*counts)
        for k in ("precision", "recall", "f1", "fbeta"):
            assert not math.isnan(z[k]), (counts, k)
            assert z[k] == 0.0, (counts, k)


def test_G1_C10_score_classes_per_class_and_macro():
    from tomocpt.metrics import score_classes

    true = {"ribo": np.array(T, float), "apo": np.array([[30, 30, 30], [70, 70, 70]], float)}
    radii = {"ribo": 100.0, "apo": 60.0}  # match radius = 0.5 * radius
    pred = {
        # ribo: two within 50 (one at 49), one at 51 -> 2 TP, 1 FP, 1 FN
        "ribo": np.array([[10, 10, 59], [50, 50, 50], [90, 20, 91]], float),
        # apo: nothing predicted
        "apo": np.zeros((0, 3)),
    }
    res = score_classes(pred, true, radii)
    ribo, apo = _per_class(res, "ribo"), _per_class(res, "apo")
    p = r = 2 / 3
    assert ribo["precision"] == pytest.approx(p) and ribo["recall"] == pytest.approx(r)
    assert ribo["fbeta"] == pytest.approx(_fbeta(p, r, 4.0))
    assert apo["recall"] == 0.0 and apo["fbeta"] == 0.0 and not math.isnan(apo["precision"])
    assert _macro_fbeta(res) == pytest.approx(ribo["fbeta"] / 2)

    # weights: weighted mean over classes
    res_w = score_classes(pred, true, radii, weights={"ribo": 3.0, "apo": 1.0})
    assert _macro_fbeta(res_w) == pytest.approx(0.75 * ribo["fbeta"])

    # match_radius_frac scales the match radius: at 0.6 the 51-away ribo prediction matches
    res_f = score_classes(pred, true, radii, match_radius_frac=0.6)
    assert _per_class(res_f, "ribo")["fbeta"] == pytest.approx(1.0)

    # beta is forwarded
    res_b = score_classes(pred, true, radii, beta=1.0)
    assert _per_class(res_b, "ribo")["fbeta"] == pytest.approx(_per_class(res_b, "ribo")["f1"])


def test_G1_C10_score_classes_perfect_and_missing_prediction_class():
    from tomocpt.metrics import score_classes

    true = {"a": np.array(T, float), "b": np.array([[30, 30, 30]], float)}
    res = score_classes({"a": np.array(T, float), "b": np.array([[30, 31, 30]], float)},
                        true, {"a": 20.0, "b": 20.0})
    assert _macro_fbeta(res) == pytest.approx(1.0)
    # a class with truth but no entry in pred counts as all-missed
    res = score_classes({"a": np.array(T, float)}, true, {"a": 20.0, "b": 20.0})
    assert _per_class(res, "b")["fbeta"] == 0.0
    assert _macro_fbeta(res) == pytest.approx(0.5)
