"""G4-C2 plan_inference, G4-C3 (one level per class; level-tagging oracle through predict)."""

import math

import numpy as np
import pytest

from gate4 import g4_helpers as H

CZII_DIAMETERS = [120.0, 130.0, 180.0, 300.0, 260.0, 270.0]
CZII_APIX = 10.012


def _table(diams):
    from tomocpt.labels.multiclass import ClassSpec

    return [ClassSpec(number=i + 1, name=f"c{i + 1}", diameter_ang=float(d), kind="particle")
            for i, d in enumerate(diams)]


def _scale(band_min=10.0, band_max=32.0):
    from tomocpt.defaultConfigs.scale_config import ScaleConfig

    s = ScaleConfig()
    s.band_min_vox = band_min
    s.band_max_vox = band_max
    return s


def _plan(apix, diams, version=2, length=None, desired=20, **band):
    from tomocpt.predict.helpers import InferenceLevel, plan_inference

    levels = plan_inference(apix, _table(diams), _scale(**band), desired, version, length=length)
    for lv in levels:
        assert isinstance(lv, InferenceLevel)
    return sorted([(float(lv.apix_work), sorted(int(c) for c in lv.classes), int(lv.overlap)) for lv in levels])


def _ov(dmax):
    return min(32, max(16, math.ceil(dmax - 1e-9)))


@pytest.mark.parametrize("apix, length, expected_apix", [
    (10.0, 200.0, 10.0),     # 20 px: no rescale
    (10.0, 205.0, 10.0),     # 20.5 px: within 1 voxel of the target, no rescale
    (10.0, 300.0, 15.0),     # 30 px -> 20 px
    (5.0, 150.0, 7.5),
])
def test_G4_C2_v1_one_level(apix, length, expected_apix):
    got = _plan(apix, [length], version=1, length=length)
    assert len(got) == 1
    a, classes, overlap = got[0]
    assert a == pytest.approx(expected_apix, rel=1e-6)
    assert classes == [0]
    assert overlap == 16


@pytest.mark.parametrize("apix, diams, expected", [
    (10.0, [70.0], [(10.0, [0], 16)]),
    (13.48, [110.0], [(13.48, [0], 16)]),
    (10.0, [250.0], [(10.0, [0], 25)]),
    (5.0, [350.0], [(10.9375, [0], 32)]),
    (14.0, [140.0], [(14.0, [0], 16)]),
    (10.0, [120.0, 300.0], [(10.0, [0, 1], 30)]),
    (10.0, [70.0, 300.0], [(10.0, [0, 1], 30)]),
    (10.0, [70.0, 500.0], [(10.0, [0], 16), (15.625, [1], 32)]),
    (CZII_APIX, CZII_DIAMETERS, [(CZII_APIX, [0, 1, 2, 3, 4, 5], 30)]),
])
def test_G4_C2_v2_cases_from_G2_C2(apix, diams, expected):
    got = _plan(apix, diams)
    assert len(got) == len(expected)
    for (ga, gc, go), (ea, ec, eo) in zip(got, expected):
        assert ga == pytest.approx(ea, rel=1e-9)
        assert gc == ec
        assert go == eo


@pytest.mark.parametrize("apix, diam, band_max, overlap", [
    (10.0, 100.0, 32.0, 16),   # d_vox 10 -> 16
    (10.0, 280.0, 32.0, 28),   # d_vox 28 -> 28
    (10.0, 400.0, 64.0, 32),   # d_vox 40 -> 32
    (10.0, 202.0, 32.0, 21),   # d_vox 20.2 -> ceil 21
])
def test_G4_C2_overlap_rule(apix, diam, band_max, overlap):
    got = _plan(apix, [diam], band_max=band_max)
    assert len(got) == 1 and got[0][2] == overlap


def test_G4_C3_every_class_is_read_at_exactly_one_level():
    """Random tables: the classes of all levels partition the class table; the overlap rule
    holds per level with the classes of that level."""
    rs = np.random.RandomState(0)
    for _ in range(200):
        apix = float(rs.uniform(3, 20))
        diams = list(rs.uniform(60, 900, rs.randint(1, 7)).round(1))
        got = _plan(apix, diams)
        seen = sorted(c for _, cl, _ in got for c in cl)
        assert seen == list(range(len(diams))), (apix, diams, got)
        for a, cl, ov in got:
            assert a >= apix * (1 - 1e-12)
            assert ov == _ov(max(diams[c] / a for c in cl))


# ------------------------------------------------------------------------------------------
# through predict, with the level-tagging oracle (constant output = tag(apix))
# ------------------------------------------------------------------------------------------

def _central(a):
    s = [slice(n // 4, n - n // 4) for n in a.shape]
    return a[tuple(s)]


def test_G4_C3_two_levels_one_predict_call_maps_for_all_classes(tmp_path, monkeypatch, v2_ckpts):
    """Two-level conditioned checkpoint (100 A at 10 A/px, 500 A at 15.625 A/px): one
    ``predict`` call writes a map for every class; each class map carries the tag of exactly
    one working voxel size (its own level), so no class is read (or mixed) at two levels."""
    info = v2_ckpts("two_level_cond")
    tomo = H.write_mrc(tmp_path / "t" / "TS_tag.mrc", H.noise_tomogram((64, 96, 96)), 10.0)
    pd, made = H.run_predict(tmp_path, monkeypatch, dict(
        weights=info["path"], tomogram_file=tomo, save_prediction_confidence_map=True,
        save_predicted_coords=False), oracle=H.tag_oracle_factory(conditioned=True))
    seen = sorted({round(a, 4) for o in made for a in o.calls if a is not None})
    assert seen == [10.0, 15.625], seen
    for name, apix in (("s100", 10.0), ("l500", 15.625)):
        m, a = H.read_map(pd / f"TS_tag_{name}.mrc")
        assert m.shape == (64, 96, 96) and a == pytest.approx(10.0)
        np.testing.assert_allclose(_central(m.astype(np.float32)), H.tag_of_apix(apix), atol=1e-3,
                                   err_msg=f"class {name} must be read at {apix} A/px only")
    heat, a = H.read_map(pd / "TS_tag_heatmap.mrc")
    assert heat.dtype == np.float32 and heat.shape == (64, 96, 96) and a == pytest.approx(10.0)
    np.testing.assert_allclose(_central(heat), H.tag_of_apix(15.625), atol=1e-3)
    cm, a = H.read_map(pd / "TS_tag_classmap.mrc")
    assert np.issubdtype(cm.dtype, np.integer) and cm.shape == (64, 96, 96) and a == pytest.approx(10.0)
    assert np.all(_central(cm) == 2)  # both pass 0.3; l500 (class 2) has the higher value


def test_G4_C2_infer_classes_drops_the_empty_level(tmp_path, monkeypatch, v2_ckpts):
    """Classes {c60, c300, c500} at 5 A/px: levels {c60} at 5 A/px and {c300, c500} at
    15.625 A/px. ``infer.classes = c300`` runs only the second level, at its planned size
    (not re-planned for c300 alone, which would give 9.375 A/px), and writes only c300."""
    info = v2_ckpts("three_cond")
    tomo = H.write_mrc(tmp_path / "t" / "TS_cls.mrc", H.noise_tomogram((64, 96, 96)), 5.0)
    pd, made = H.run_predict(tmp_path, monkeypatch, dict(
        weights=info["path"], tomogram_file=tomo, save_prediction_confidence_map=True,
        save_predicted_coords=False, classes="c300"), oracle=H.tag_oracle_factory(conditioned=True))
    seen = sorted({round(a, 4) for o in made for a in o.calls if a is not None})
    assert seen == [15.625], seen
    m, _ = H.read_map(pd / "TS_cls_c300.mrc")
    np.testing.assert_allclose(_central(m.astype(np.float32)), H.tag_of_apix(15.625), atol=1e-3)
    assert not (pd / "TS_cls_c60.mrc").exists()
    assert not (pd / "TS_cls_c500.mrc").exists()
