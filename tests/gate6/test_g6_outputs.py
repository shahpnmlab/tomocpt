"""G6-C5 (filament outputs through ``predict``) and the ``predict`` -> ``read_filament_coords``
round trip of G6-C12.

``predict()`` runs in-process on the CPU (``g4_helpers.run_predict``) with tiny v2 checkpoints
written by the code under test: joint = class 1 "ribo" (particle, 120 A) + class 5
"microtubule" (filament, 250 A); particle-only = class 1 "ribo". The network is replaced by
``g6_helpers.sign_oracle_factory``: particle channels ``clip(-x_norm - 0.5)``, filament channels
``clip(x_norm - 0.5)``, on a noise-free tomogram (``signed_tomogram``) with three dark 120 A
blobs and two bright 250 A tubes (an arc and a straight line), 10 A/px: one level at the native
voxel size for both checkpoints, so the particle channel is the same in both runs.
Infer settings: defaults (``confidence_threshold`` 0.3, ``filament_spacing_ang`` 40,
``filament_min_length_ang`` 200, ``filament_smoothing_ang`` 10, ``cross_class_nms`` on).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from gate4 import g4_helpers as G4
from gate6 import g6_helpers as H

APIX = 10.0
SHAPE = (64, 128, 160)
PARTICLES = ((1, "ribo", 120.0),)
FILAMENTS = ((5, "microtubule", 250.0),)
SPACING = 40.0
BLOBS = ((1, 120.0, (32.3, 82.2, 60.4)), (1, 120.0, (31.6, 30.7, 129.8)), (1, 120.0, (20.4, 20.3, 60.6)))


def tube_truths():
    arc = H.arc_xyz((72.0, -80.0), 140.0, math.radians(72.5), math.radians(107.5), z=32.0, n=601)
    line = H.line_xyz((30.0, 105.0, 32.0), (130.0, 105.0, 32.0), n=201)
    return [arc, line]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Predict runs: relion_31 (joint and particle-only), warp and imod (joint)."""
    tmp = tmp_path_factory.mktemp("g6_outputs")
    joint = H.build_ckpt(tmp / "joint", PARTICLES, FILAMENTS)
    ponly = H.build_ckpt(tmp / "ponly", PARTICLES, ())
    vol = H.signed_tomogram(SHAPE, APIX, BLOBS, [(5, 250.0, t) for t in tube_truths()])
    tomo = H.write_mrc(tmp / "tomograms" / "TS_f.mrc", vol, APIX)
    out = dict(tmp=tmp, tomo=tomo)
    cases = {
        "relion": (joint, "relion_31"),
        "ponly": (ponly, "relion_31"),
        "warp": (joint, "warp"),
        "imod": (joint, "imod"),
    }
    for tag, (ckpt, fmt) in cases.items():
        with pytest.MonkeyPatch.context() as mp:
            pd_, _ = G4.run_predict(tmp / tag, mp, dict(
                weights=ckpt, tomogram_file=tomo, predictions_coord_format=fmt,
                predictions_coord_filename="picks.star"), oracle=H.sign_oracle_factory())
        out[tag] = pd_
    return out


def _tubes_from_star(df):
    """``{tube_id: (xyz [K, 3], track_length [K], rows)}`` ordered by track length."""
    out = {}
    for tid, g in df.groupby(df["rlnHelicalTubeID"].astype(int)):
        g = g.sort_values("rlnHelicalTrackLengthAngst")
        out[int(tid)] = (g[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float),
                         g["rlnHelicalTrackLengthAngst"].to_numpy(float), g)
    return out


def test_G6_C5_one_file_per_class(runs):
    assert sorted(p.name for p in runs["relion"].glob("*.star")) == ["picks_microtubule.star", "picks_ribo.star"]
    assert sorted(p.name for p in runs["warp"].glob("*.star")) == ["picks_microtubule.star", "picks_ribo.star"]
    assert sorted(p.name for p in runs["imod"].glob("*.mod")) == ["TS_f_microtubule.mod", "TS_f_ribo.mod"]
    assert not (runs["relion"] / "picks.star").exists()


def test_G6_C5_relion_filament_star(runs):
    import starfile

    path = runs["relion"] / "picks_microtubule.star"
    d = starfile.read(str(path))  # reads back with starfile
    df = H.read_star_df(path)
    need = {"rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnClassNumber", "rlnHelicalTubeID",
            "rlnHelicalTrackLengthAngst", "rlnAutopickFigureOfMerit", "rlnAngleRot", "rlnAngleTilt",
            "rlnAnglePsi", "rlnAngleTiltPrior", "rlnAnglePsiPrior", "rlnAnglePsiFlipRatio"}
    assert need <= set(df.columns), sorted(need - set(df.columns))
    assert "rlnMicrographName" in df.columns and set(df["rlnMicrographName"].astype(str)) == {"TS_f"}
    assert set(df["rlnClassNumber"].astype(int)) == {5}
    np.testing.assert_allclose(df["rlnAngleRot"].astype(float), 0.0, atol=1e-6)
    np.testing.assert_allclose(df["rlnAnglePsiFlipRatio"].astype(float), 0.5, atol=1e-6)
    np.testing.assert_allclose(df["rlnAngleTiltPrior"].astype(float), df["rlnAngleTilt"].astype(float), atol=1e-3)
    np.testing.assert_allclose(df["rlnAnglePsiPrior"].astype(float), df["rlnAnglePsi"].astype(float), atol=1e-3)
    fom = df["rlnAutopickFigureOfMerit"].astype(float)
    assert fom.between(0.5, 1.0).all(), "mean probability along the filament (the oracle gives ~1)"

    tubes = _tubes_from_star(df)
    assert sorted(tubes) == [1, 2], f"rlnHelicalTubeID must be 1..n per tomogram and class, got {sorted(tubes)}"
    truths = tube_truths()
    m = H.match_polylines([t[0] for t in tubes.values()], truths)
    assert sorted(m) == [0, 1], m
    ez = np.array([0.0, 0.0, 1.0])
    for (tid, (xyz, track, rows)), k in zip(tubes.items(), m):
        assert track[0] == pytest.approx(0.0, abs=1e-3)
        dt = np.diff(track)
        assert np.all(np.abs(dt[:-1] / SPACING - 1.0) <= 0.01), f"tube {tid}: track length steps {dt}"
        assert 0 < dt[-1] <= SPACING * 1.01
        L_true = H.polyline_length(truths[k]) * APIX
        assert track[-1] >= 0.8 * L_true - 2 * 125.0, f"tube {tid}: traced {track[-1]:.0f} A of {L_true:.0f} A"
        md = H.mean_dist_to_curve(xyz, truths[k])
        assert md <= 1.0, f"tube {tid}: mean distance {md:.3f} voxel to the true centreline"
        # the angles describe the local filament axis (RELION Euler_angles2matrix, G6-C11)
        tilt, psi = rows["rlnAngleTilt"].to_numpy(float), rows["rlnAnglePsi"].to_numpy(float)
        for i in range(1, len(xyz) - 1):
            d = xyz[i + 1] - xyz[i - 1]
            d /= np.linalg.norm(d)
            v = H.relion_euler_matrix(0.0, tilt[i], psi[i]) @ ez
            assert abs(abs(float(v @ d)) - 1.0) < 2e-3, f"tube {tid} point {i}: axis {v} vs direction {d}"


def test_G6_C5_particle_file_unchanged(runs):
    """The particle STAR file of the joint model equals that of the particle-only model (same
    particle channel, particles not suppressed by filaments) and finds every blob."""
    a = G4.picks_by_class(runs["relion"] / "picks_ribo.star")[1]
    b = G4.picks_by_class(runs["ponly"] / "picks_ribo.star")[1]
    oa, ob = np.lexsort(a[:, :3].T[::-1]), np.lexsort(b[:, :3].T[::-1])
    assert a.shape == b.shape, f"{len(a)} vs {len(b)} particle picks"
    np.testing.assert_allclose(a[oa], b[ob], atol=1e-6)
    true_xyz = np.array([[x, y, z] for _, _, (z, y, x) in BLOBS])
    assert np.all(G4.nearest_dist(a[:, :3], true_xyz) <= 1.0)


def test_G6_C5_warp_filament_star_has_angles(runs):
    df = H.read_star_df(runs["warp"] / "picks_microtubule.star")
    assert {"rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"} <= set(df.columns)
    np.testing.assert_allclose(df["rlnAngleRot"].astype(float), 0.0, atol=1e-6)
    rel = H.read_star_df(runs["relion"] / "picks_microtubule.star")
    cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    w, r = df[cols].to_numpy(float), rel[cols].to_numpy(float)
    assert w.shape == r.shape
    assert np.all(G4.nearest_dist(w, r) <= 0.01) and np.all(G4.nearest_dist(r, w) <= 0.01)


def test_G6_C5_imod_one_open_contour_per_filament(runs):
    import imodmodel

    df = imodmodel.read(str(runs["imod"] / "TS_f_microtubule.mod"))
    contours = df.groupby(["object_id", "contour_id"])
    assert contours.ngroups == 2, f"{contours.ngroups} contours, expected one per filament"
    rel = _tubes_from_star(H.read_star_df(runs["relion"] / "picks_microtubule.star"))
    for _, g in contours:
        xyz = g[["x", "y", "z"]].to_numpy(float)
        best = min(rel.values(), key=lambda t: H.mean_dist_to_curve(xyz, t[0]))[0]
        assert xyz.shape == best.shape
        assert np.all(G4.nearest_dist(xyz, best) <= 0.01)


@pytest.mark.parametrize("which", ["relion", "imod"])
def test_G6_C12_predict_output_round_trip(runs, which):
    """``read_filament_coords`` on the ``predict`` filament output gives the written polylines
    (one per filament, in order)."""
    from tomocpt.labels.filaments import read_filament_coords

    rel = _tubes_from_star(H.read_star_df(runs["relion"] / "picks_microtubule.star"))
    f = runs["relion"] / "picks_microtubule.star" if which == "relion" else runs["imod"] / "TS_f_microtubule.mod"
    polys = [np.asarray(p, float) for p in read_filament_coords(f, APIX)]
    assert len(polys) == len(rel) == 2
    for p in polys:
        want = min(rel.values(), key=lambda t: H.mean_dist_to_curve(p, t[0]))[0]
        assert p.shape == want.shape
        if np.linalg.norm(p[0] - want[0]) > np.linalg.norm(p[0] - want[-1]):
            want = want[::-1]
        np.testing.assert_allclose(p, want, atol=0.01)
