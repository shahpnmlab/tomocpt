"""G2-C2 band rule, G2-C3 plan_levels, G2-C4 Fourier resampling, G2-C17 read_apix."""

import math

import numpy as np
import pytest
import torch

BAND_MIN, BAND_MAX = 10.0, 32.0
RATIO = BAND_MAX / BAND_MIN

#: CZII 10440 species diameters (A) from the CZII challenge particle radii:
#: apo-ferritin 60, beta-amylase 65, beta-galactosidase 90, ribosome 150, thyroglobulin 130,
#: virus-like particle 135 (radius, A). The repo does not record them; these are the public values.
CZII_DIAMETERS = [120.0, 130.0, 180.0, 300.0, 260.0, 270.0]
CZII_APIX = 10.012


def _band_rule(apix, lengths):
    return max(apix, max(lengths) / BAND_MAX)


def _ref_groups(apix, lengths):
    """Reference greedy grouping from the spec (sort by L_eff, grow while ratio <= band ratio)."""
    leff = lambda L: max(L, apix * BAND_MIN)  # noqa: E731
    groups, cur = [], []
    for L in sorted(lengths, key=leff):
        if cur and leff(L) / leff(cur[0]) > RATIO:
            groups.append(cur)
            cur = []
        cur.append(L)
    groups.append(cur)
    return [(_band_rule(apix, g), sorted(g)) for g in groups]


def _norm_groups(groups):
    return sorted(((float(a), sorted(float(x) for x in g)) for a, g in groups), key=lambda t: t[1][0])


def _table(diameters):
    from tomocpt.labels.multiclass import ClassSpec

    return [ClassSpec(number=i + 1, name=f"c{i + 1}", diameter_ang=float(d)) for i, d in enumerate(diameters)]


# ------------------------------------------------------------------------------------------
# G2-C2
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("apix, lengths, expected", [
    (10.0, [70.0], 10.0),       # no upsampling
    (13.48, [110.0], 13.48),
    (10.0, [250.0], 10.0),
    (5.0, [350.0], 10.9375),
    (14.0, [140.0], 14.0),
    (CZII_APIX, CZII_DIAMETERS, CZII_APIX),  # CZII: no resampling
])
def test_G2_C2_band_rule_table(apix, lengths, expected):
    from tomocpt.dataManager.scaling import choose_working_apix

    got = choose_working_apix(apix, lengths, BAND_MIN, BAND_MAX)
    assert got == pytest.approx(expected, rel=1e-12, abs=0)
    assert got >= apix


@pytest.mark.parametrize("apix, lengths, expected", [
    (10.0, [120.0, 300.0], [(10.0, [120.0, 300.0])]),
    (10.0, [70.0, 300.0], [(10.0, [70.0, 300.0])]),         # 70 A counts as 100 A
    (10.0, [70.0, 500.0], [(10.0, [70.0]), (15.625, [500.0])]),  # two groups
    (CZII_APIX, CZII_DIAMETERS, [(CZII_APIX, sorted(CZII_DIAMETERS))]),
])
def test_G2_C2_group_lengths_table(apix, lengths, expected):
    from tomocpt.dataManager.scaling import group_lengths

    got = _norm_groups(group_lengths(apix, lengths, BAND_MIN, BAND_MAX))
    exp = _norm_groups(expected)
    assert len(got) == len(exp)
    for (ga, gl), (ea, el) in zip(got, exp):
        assert ga == pytest.approx(ea, rel=1e-12)
        assert gl == el  # original lengths, not L_eff


def test_G2_C2_no_single_value_raises():
    from tomocpt.dataManager.scaling import choose_working_apix

    with pytest.raises(ValueError):
        choose_working_apix(10.0, [70.0, 500.0], BAND_MIN, BAND_MAX)


def test_G2_C2_czii_one_level_no_resampling():
    from tomocpt.dataManager.scaling import plan_levels

    levels = plan_levels(CZII_APIX, _table(CZII_DIAMETERS), BAND_MIN, BAND_MAX)
    assert len(levels) == 1
    assert levels[0].apix_work == pytest.approx(CZII_APIX, rel=1e-12)
    assert sorted(levels[0].assigned) == list(range(6))
    assert sorted(levels[0].available) == list(range(6))


def test_G2_C2_random_cases_follow_the_rule_and_never_upsample():
    """500 random (apix, lengths) cases: group_lengths equals the spec's greedy grouping with
    the band rule per group, choose_working_apix agrees (or raises when there are several
    groups), and no case returns apix_work < apix_native."""
    from tomocpt.dataManager.scaling import choose_working_apix, group_lengths, plan_levels

    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(500):
        apix = float(rng.uniform(2.0, 25.0))
        lengths = sorted(set(float(round(x, 1)) for x in rng.uniform(40.0, 1200.0, size=rng.integers(1, 7))))
        leffs = sorted(max(L, apix * BAND_MIN) for L in lengths)
        ratios = [b / a for a in leffs for b in leffs if b > a]
        if any(abs(r - RATIO) < 1e-6 for r in ratios):
            continue  # skip exact-boundary cases
        n_checked += 1
        ref = _ref_groups(apix, lengths)
        got = _norm_groups(group_lengths(apix, lengths, BAND_MIN, BAND_MAX))
        assert len(got) == len(ref), (apix, lengths)
        for (ga, gl), (ea, el) in zip(got, _norm_groups(ref)):
            assert gl == el, (apix, lengths)
            assert ga == pytest.approx(ea, rel=1e-9)
            assert ga >= apix
        if len(ref) == 1:
            assert choose_working_apix(apix, lengths, BAND_MIN, BAND_MAX) == pytest.approx(ref[0][0], rel=1e-9)
        else:
            with pytest.raises(ValueError):
                choose_working_apix(apix, lengths, BAND_MIN, BAND_MAX)
        for lv in plan_levels(apix, _table(lengths), BAND_MIN, BAND_MAX):
            assert lv.apix_work >= apix
    assert n_checked > 400


# ------------------------------------------------------------------------------------------
# G2-C3
# ------------------------------------------------------------------------------------------

def _check_levels(apix, diameters):
    from tomocpt.dataManager.scaling import Level, plan_levels

    levels = plan_levels(apix, _table(diameters), BAND_MIN, BAND_MAX)
    assert len(levels) >= 1
    assigned_all = []
    for lv in levels:
        assert isinstance(lv, Level)
        assert lv.apix_work >= apix
        assert set(lv.assigned) <= set(lv.available)
        assigned_all.extend(lv.assigned)
        for c, D in enumerate(diameters):
            d = D / lv.apix_work
            lo = min(BAND_MIN, D / apix)
            inside = lo * (1 - 1e-9) <= d <= BAND_MAX * (1 + 1e-9)
            outside = d < lo * (1 - 1e-5) or d > BAND_MAX * (1 + 1e-5)
            if c in lv.available:
                assert not outside, (apix, diameters, lv, c, d)
            if inside:
                assert c in lv.available, (apix, diameters, lv, c, d)
    assert sorted(assigned_all) == list(range(len(diameters)))  # each class assigned exactly once
    return levels


@pytest.mark.parametrize("apix, diameters", [
    (10.0, [80.0, 400.0]),
    (10.0, [80.0, 120.0, 400.0]),
    (10.0, [70.0, 120.0, 300.0]),
    (5.0, [120.0, 350.0]),
    (7.0, [120.0, 300.0]),
    (CZII_APIX, CZII_DIAMETERS),
    (10.0, [60.0, 150.0, 480.0, 1500.0, 5000.0]),
])
def test_G2_C3_plan_levels_properties(apix, diameters):
    _check_levels(apix, diameters)


def test_G2_C3_plan_levels_random():
    rng = np.random.default_rng(1)
    for _ in range(300):
        apix = float(rng.uniform(2.0, 25.0))
        diameters = sorted(set(float(round(x, 1)) for x in rng.uniform(40.0, 2000.0, size=rng.integers(1, 7))))
        _check_levels(apix, diameters)


def test_G2_C3_below_band_class_only_at_native_level():
    """80 A at 10 A/px is 8 voxels (below the band): it is read at native size only. The
    400 A class needs its own level at 12.5 A, where neither 80 A nor 120 A is available."""
    from tomocpt.dataManager.scaling import plan_levels

    levels = plan_levels(10.0, _table([80.0, 120.0, 400.0]), BAND_MIN, BAND_MAX)
    assert len(levels) == 2
    by_apix = {round(lv.apix_work, 9): lv for lv in levels}
    assert set(by_apix) == {10.0, 12.5}
    assert sorted(by_apix[10.0].assigned) == [0, 1] and sorted(by_apix[10.0].available) == [0, 1]
    assert sorted(by_apix[12.5].assigned) == [2] and sorted(by_apix[12.5].available) == [2]
    for lv in levels:
        if 0 in lv.available:
            assert lv.apix_work == pytest.approx(10.0)


# ------------------------------------------------------------------------------------------
# G2-C4
# ------------------------------------------------------------------------------------------

def _resample(vol, shape):
    from tomocpt.dataManager.scaling import resample_volume
    from tomocpt.defaultConfigs.scale_config import ResampleMethod

    out = resample_volume(torch.as_tensor(vol, dtype=torch.float32), tuple(shape), ResampleMethod.fourier)
    assert isinstance(out, torch.Tensor)
    assert not torch.is_complex(out)
    assert tuple(out.shape) == tuple(shape)
    return out.double().numpy()


def _cosine(shape, axis, cycles):
    idx = np.arange(shape[axis], dtype=np.float64)
    wave = np.cos(2 * np.pi * cycles * idx / shape[axis])
    sh = [1, 1, 1]
    sh[axis] = shape[axis]
    return np.broadcast_to(wave.reshape(sh), shape).astype(np.float32)


def _blob(shape, centre, sigma=3.0):
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij")
    r2 = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
    return np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)


def _centroid(v):
    w = np.asarray(v, dtype=np.float64)
    grids = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in w.shape], indexing="ij")
    return np.array([(g * w).sum() / w.sum() for g in grids])


@pytest.mark.parametrize("shape_in, shape_out", [
    ((64, 64, 64), (32, 32, 32)),
    ((63, 65, 64), (31, 33, 32)),
    ((64, 65, 63), (39, 40, 38)),
])
def test_G2_C4_cosine_above_new_nyquist_is_removed(shape_in, shape_out):
    """A cosine above the new Nyquist keeps < 1 % of its power (trilinear would alias it);
    a cosine well below it keeps > 90 % (the method is not just a low-pass to zero)."""
    for axis in range(3):
        n_in, n_out = shape_in[axis], shape_out[axis]
        high = int(math.ceil(0.40 * n_in))           # 0.40 cycles/voxel > new Nyquist (~0.25-0.31)
        assert high / n_in > 0.5 * n_out / n_in + 0.05
        v = _cosine(shape_in, axis, high)
        out = _resample(v, shape_out)
        assert (out ** 2).mean() < 0.01 * (v.astype(np.float64) ** 2).mean(), (axis, high)
        low = 3
        v = _cosine(shape_in, axis, low)
        out = _resample(v, shape_out)
        assert (out ** 2).mean() > 0.9 * (v.astype(np.float64) ** 2).mean(), (axis, low)


@pytest.mark.parametrize("shape_in, shape_out", [
    ((64, 64, 64), (40, 40, 40)),
    ((63, 65, 64), (45, 41, 47)),
    ((40, 41, 39), (64, 63, 65)),   # zero-padding (upsampling) as well
])
def test_G2_C4_mean_is_kept(shape_in, shape_out):
    rng = np.random.default_rng(3)
    v = (rng.normal(size=shape_in) + 0.37).astype(np.float32)
    out = _resample(v, shape_out)
    assert out.mean() == pytest.approx(float(v.astype(np.float64).mean()), rel=1e-4)


@pytest.mark.parametrize("shape_in, shape_out", [
    ((64, 64, 64), (40, 40, 40)),
    ((63, 65, 64), (45, 41, 47)),
])
def test_G2_C4_blob_centroid_moves_less_than_0_1_voxel(shape_in, shape_out):
    """The DFT crop samples output voxel j at input position j * N_in / N_out (voxel 0 stays
    at the origin, as coordinates in pixels * apix assume), so a blob at input position c
    should have its centroid at c * N_out / N_in. Tolerance: 0.1 output voxel."""
    centre = np.array([30.3, 33.7, 31.2])
    v = _blob(shape_in, centre)
    out = _resample(v, shape_out)
    expected = centre * np.array(shape_out) / np.array(shape_in)
    assert np.abs(_centroid(out) - expected).max() < 0.1


@pytest.mark.parametrize("shape_in, shape_work", [
    ((64, 64, 64), (40, 40, 40)),
    ((65, 63, 64), (41, 45, 47)),
])
def test_G2_C4_round_trip_centroid_moves_less_than_0_25_native_voxel(shape_in, shape_work):
    centre = np.array([29.6, 33.2, 30.9])
    v = _blob(shape_in, centre)
    back = _resample(_resample(v, shape_work), shape_in)
    assert np.abs(_centroid(back) - _centroid(v)).max() < 0.25


def test_G2_C4_trilinear_is_the_legacy_resize():
    """resample_volume(..., trilinear) calls preprocessing.resize_volume unchanged."""
    from tomocpt.dataManager.preprocessing import resize_volume
    from tomocpt.dataManager.scaling import resample_volume
    from tomocpt.defaultConfigs.scale_config import ResampleMethod

    v = torch.from_numpy(np.random.default_rng(0).normal(size=(37, 50, 45)).astype(np.float32))
    for shape in [(20, 27, 24), (64, 70, 66)]:
        assert torch.equal(resample_volume(v.clone(), shape, ResampleMethod.trilinear),
                           resize_volume(v.clone(), shape))


def test_G2_C4_working_shape_uses_v1_rounding():
    from tomocpt.dataManager.scaling import working_shape

    assert tuple(working_shape((64, 128, 128), 7.0, 9.375)) == (47, 95, 95)
    assert tuple(working_shape((64, 128, 127), 10.0, 10.0)) == (64, 128, 127)
    assert tuple(working_shape((100, 99, 33), 5.0, 10.9375)) == tuple(int(s * 5.0 / 10.9375) for s in (100, 99, 33))


# ------------------------------------------------------------------------------------------
# G2-C17
# ------------------------------------------------------------------------------------------

def _mrc(path, apix):
    import mrcfile

    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.zeros((8, 8, 8), dtype=np.float32))
        m.voxel_size = apix
    return path


def test_G2_C17_read_apix_header_zero_raises_naming_apix_override(tmp_path):
    from tomocpt.dataManager.scaling import read_apix

    p = _mrc(tmp_path / "zero.mrc", 0.0)
    with pytest.raises(ValueError, match="apix_override"):
        read_apix(p)


def test_G2_C17_read_apix_override_wins(tmp_path):
    from tomocpt.dataManager.scaling import read_apix

    p10 = _mrc(tmp_path / "ten.mrc", 10.0)
    p0 = _mrc(tmp_path / "zero.mrc", 0.0)
    assert read_apix(p10) == pytest.approx(10.0)
    assert read_apix(str(p10)) == pytest.approx(10.0)
    assert read_apix(p10, override=7.5) == pytest.approx(7.5)
    assert read_apix(p0, override=8.25) == pytest.approx(8.25)
