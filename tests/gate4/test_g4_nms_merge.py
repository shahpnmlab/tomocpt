"""G4-C5 cross_class_suppression, G4-C12 merge_class_maps.

Formats chosen where the spec is silent (README): ``peaks_by_class`` maps a class key to an
``[N, 4]`` array of rows ``(z, y, x, score)`` (the rows ``_extract_peaks_from_mask`` returns);
``radii`` maps the same keys to radii in voxels; ``removed_counts[(kept, removed)]`` counts the
peaks of class ``removed`` dropped because a peak of class ``kept`` won. ``merge_class_maps``
takes ``maps [C, D, H, W]``, ``class_numbers`` and ``thresholds`` as sequences in channel order.
"""

import numpy as np
import pytest


def _rows(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    return sorted(tuple(np.round(r, 6)) for r in a)


def _counts(removed):
    """Normalise to ``{(kept, removed): n}`` with zero entries dropped."""
    out = {}
    for k, v in dict(removed).items():
        if isinstance(v, dict):
            for k2, n in v.items():
                if int(n):
                    out[(int(k), int(k2))] = int(n)
        elif int(v):
            out[tuple(int(x) for x in k)] = int(v)
    return out


def _run(peaks, radii, **kw):
    from tomocpt.predict.helpers import cross_class_suppression

    peaks = {k: np.asarray(v, dtype=np.float64).reshape(-1, 4) for k, v in peaks.items()}
    kept, removed = cross_class_suppression(peaks, radii, **kw)
    return {k: _rows(v) for k, v in kept.items()}, _counts(removed)


R = {1: 10.0, 2: 6.0, 3: 8.0}


@pytest.mark.parametrize("s1, s2, winner", [(0.9, 0.6, 1), (0.6, 0.95, 2)])
def test_G4_C5_one_voxel_apart_the_higher_score_stays(s1, s2, winner):
    p1, p2 = [50, 50, 50, s1], [50, 50, 51, s2]
    kept, counts = _run({1: [p1], 2: [p2]}, {1: R[1], 2: R[2]})
    loser = 2 if winner == 1 else 1
    assert kept[winner] == _rows([p1 if winner == 1 else p2])
    assert kept.get(loser, []) == []
    assert counts == {(winner, loser): 1}


def test_G4_C5_three_radii_apart_both_stay():
    p1, p2 = [50, 50, 50, 0.9], [50, 50, 50 + 3 * max(R[1], R[2]), 0.6]
    kept, counts = _run({1: [p1], 2: [p2]}, {1: R[1], 2: R[2]})
    assert kept[1] == _rows([p1]) and kept[2] == _rows([p2])
    assert counts == {}


@pytest.mark.parametrize("dist, suppressed", [(2.5, True), (4.0, False)])
def test_G4_C5_distance_uses_factor_times_the_smaller_radius(dist, suppressed):
    """0.5 * min(10, 6) = 3 voxels: 2.5 apart -> suppressed, 4 apart -> both stay
    (4 < 0.5 * max = 5, so a max-radius rule would suppress)."""
    p1, p2 = [40, 40, 40, 0.9], [40, 40 + dist, 40, 0.6]
    kept, counts = _run({1: [p1], 2: [p2]}, {1: R[1], 2: R[2]}, factor=0.5)
    assert (kept.get(2, []) == []) is suppressed
    assert counts == ({(1, 2): 1} if suppressed else {})


def test_G4_C5_factor_argument_is_used():
    p1, p2 = [40, 40, 40, 0.9], [40, 44, 40, 0.6]  # 4 apart; factor 1 -> 1 * 6 = 6 > 4
    kept, counts = _run({1: [p1], 2: [p2]}, {1: R[1], 2: R[2]}, factor=1.0)
    assert kept.get(2, []) == [] and counts == {(1, 2): 1}


def test_G4_C5_same_class_peaks_are_untouched():
    a, b = [50, 50, 50, 0.9], [50, 50, 51, 0.5]
    far = [10, 10, 10, 0.4]
    kept, counts = _run({1: [a, b], 2: [far]}, {1: R[1], 2: R[2]})
    assert kept[1] == _rows([a, b]) and kept[2] == _rows([far])
    assert counts == {}


def test_G4_C5_counts_per_class_pair():
    A, B = [20, 20, 20, 0.9], [80, 80, 80, 0.5]
    C, D, E = [20, 20, 21, 0.6], [80, 81, 80, 0.7], [50, 50, 50, 0.3]
    F = [20, 21, 20, 0.2]
    kept, counts = _run({1: [A, B], 2: [C, D, E], 3: [F]}, R)
    assert kept[1] == _rows([A])
    assert kept[2] == _rows([D, E])
    assert kept.get(3, []) == []
    assert counts == {(1, 2): 1, (1, 3): 1, (2, 1): 1}


# ------------------------------------------------------------------------------------------
# G4-C12
# ------------------------------------------------------------------------------------------

def _np(a):
    try:
        import torch

        if torch.is_tensor(a):
            return a.detach().cpu().numpy()
    except Exception:  # pragma: no cover
        pass
    return np.asarray(a)


def _reference_classmap(maps, numbers, thr):
    passed = maps >= np.asarray(thr, dtype=maps.dtype)[:, None, None, None]
    masked = np.where(passed, maps, -np.inf)
    idx = masked.argmax(0)
    return np.where(passed.any(0), np.asarray(numbers)[idx], 0)


def test_G4_C12_merge_class_maps_random():
    from tomocpt.predict.helpers import merge_class_maps

    rs = np.random.RandomState(0)
    maps = rs.uniform(0, 1, (3, 9, 10, 11)).astype(np.float32)
    numbers, thr = [2, 5, 9], [0.5, 0.6, 0.7]
    heat, cls = merge_class_maps(maps, numbers, thr)
    heat, cls = _np(heat), _np(cls)
    assert heat.shape == maps.shape[1:] and cls.shape == maps.shape[1:]
    assert heat.dtype == np.float32
    np.testing.assert_array_equal(heat, maps.max(0))
    assert cls.dtype == np.uint8
    assert set(np.unique(cls).tolist()) <= {0, 2, 5, 9}
    np.testing.assert_array_equal(cls.astype(np.int64), _reference_classmap(maps, numbers, thr))


def test_G4_C12_specific_voxels():
    """Below every threshold -> 0; two classes pass -> the higher probability wins, even when
    a third class has a higher probability below its own threshold."""
    from tomocpt.predict.helpers import merge_class_maps

    maps = np.zeros((3, 2, 2, 2), dtype=np.float32)
    numbers, thr = [4, 7, 11], [0.3, 0.3, 0.95]
    maps[:, 0, 0, 0] = [0.29, 0.2, 0.9]     # nothing passes (class 11 is highest but < 0.95)
    maps[:, 0, 0, 1] = [0.6, 0.8, 0.9]      # 4 and 7 pass, 7 wins; 11 is higher but below
    maps[:, 0, 1, 0] = [0.7, 0.4, 0.96]     # all pass, 11 wins
    maps[:, 0, 1, 1] = [0.3, 0.1, 0.0]      # exactly at the threshold passes
    heat, cls = (_np(a) for a in merge_class_maps(maps, numbers, thr))
    assert int(cls[0, 0, 0]) == 0
    assert int(cls[0, 0, 1]) == 7
    assert int(cls[0, 1, 0]) == 11
    assert int(cls[0, 1, 1]) == 4
    assert int(cls[1, 1, 1]) == 0
    assert heat[0, 0, 0] == pytest.approx(0.9)


def test_G4_C12_uint16_above_255():
    from tomocpt.predict.helpers import merge_class_maps

    maps = np.zeros((2, 2, 2, 2), dtype=np.float32)
    maps[1, 0, 0, 0] = 0.9
    maps[0, 1, 1, 1] = 0.9
    heat, cls = (_np(a) for a in merge_class_maps(maps, [3, 300], [0.5, 0.5]))
    assert cls.dtype == np.uint16
    assert int(cls[0, 0, 0]) == 300 and int(cls[1, 1, 1]) == 3 and int(cls[0, 1, 0]) == 0
    _, cls8 = (_np(a) for a in merge_class_maps(maps, [3, 255], [0.5, 0.5]))
    assert cls8.dtype == np.uint8 and int(cls8[0, 0, 0]) == 255
