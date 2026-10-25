"""Tests for scripts/download_portal_data.py that need no network and no portal client."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_portal_data.py"
_spec = importlib.util.spec_from_file_location("download_portal_data", _SCRIPT)
dpd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dpd  # dataclasses look the module up by name
_spec.loader.exec_module(dpd)


def test_safe_name():
    assert dpd.safe_name("beta-galactosidase") == "beta-galactosidase"
    assert dpd.safe_name(" virus like particle ") == "virus_like_particle"
    assert dpd.safe_name("a/b:c") == "a_b_c"
    assert dpd.safe_name("") == "unnamed"


def test_parse_ndjson_points_skips_blank_lines_and_records_without_location():
    lines = [
        json.dumps({"type": "orientedPoint", "location": {"x": 1, "y": 2, "z": 3}}),
        "",
        json.dumps({"type": "other"}),
        json.dumps({"location": {"x": 4.5, "y": 5.5, "z": 6.5}}),
    ]
    pts = dpd.parse_ndjson_points(lines)
    assert pts.shape == (2, 3)
    assert np.allclose(pts, [[1, 2, 3], [4.5, 5.5, 6.5]])


def test_parse_ndjson_points_empty():
    assert dpd.parse_ndjson_points([]).shape == (0, 3)


def test_rescale_points():
    pts = np.array([[10.0, 20.0, 30.0]])
    assert np.allclose(dpd.rescale_points(pts, 5.0, 10.0), [[5.0, 10.0, 15.0]])
    with pytest.raises(ValueError):
        dpd.rescale_points(pts, 0.0, 10.0)


def test_check_points_inside():
    size = (100, 50, 20)
    pts = np.array([[0, 0, 0], [99, 49, 19], [100.6, 0, 0], [0, 0, -1]])
    assert dpd.check_points_inside(pts, size) == 2
    assert dpd.check_points_inside(np.zeros((0, 3)), size) == 0


def test_particles_table_columns_and_rows():
    df = dpd.particles_table({"TS_2": np.array([[1, 2, 3]]), "TS_1": np.array([[4, 5, 6], [7, 8, 9]])}, 3)
    assert list(df.columns[:5]) == [
        "rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnClassNumber",
    ]
    assert df["rlnMicrographName"].tolist() == ["TS_1", "TS_1", "TS_2"]
    assert (df["rlnClassNumber"] == 3).all()


def test_assign_class_numbers_is_sorted_and_stable():
    assert dpd.assign_class_numbers(["ribosome", "apo-ferritin", "ribosome"]) == {
        "apo-ferritin": 1, "ribosome": 2,
    }


def test_parse_diameters():
    assert dpd.parse_diameters("ribosome=300, apo ferritin=120") == {"ribosome": 300.0, "apo_ferritin": 120.0}
    assert dpd.parse_diameters(None) == {}
    with pytest.raises(ValueError):
        dpd.parse_diameters("ribosome")


def test_prepdata_snippet_marks_missing_diameters():
    stars = {"ribosome": Path("/d/coords/ribosome.star"), "apo-ferritin": Path("/d/coords/apo-ferritin.star")}
    text = dpd.prepdata_snippet(Path("/d/tomograms"), stars, {"ribosome": 300.0}, Path("/d/training_data"))
    assert "raw_data_dir: /d/tomograms,/d/tomograms" in text
    assert "coordinate_files: /d/coords/apo-ferritin.star,/d/coords/ribosome.star" in text
    assert "particle_length_ang: ???,300.0" in text
    assert "class_id: all,all" in text
    assert "Fill in particle_length_ang" in text


def _cand(i, vs, proc="raw", rec="WBP", std=False, vis=False):
    return dpd.TomoCandidate(i, vs, proc, rec, std, vis)


def test_choose_tomogram_filters_and_prefers_portal_standard():
    cands = [
        _cand(5, 10.012, "denoised"),
        _cand(3, 10.012, "raw"),
        _cand(4, 10.012, "raw", std=True),
        _cand(1, 4.99, "raw", std=True),
    ]
    assert dpd.choose_tomogram(cands, 10.012, 0.05, "raw", None).id == 4
    assert dpd.choose_tomogram(cands, 10.0, 0.05, "Denoised", None).id == 5
    assert dpd.choose_tomogram(cands, None, 0.05, None, None).id == 1
    assert dpd.choose_tomogram(cands, 20.0, 0.05, None, None) is None
    assert dpd.choose_tomogram(cands, 10.012, 0.05, None, "SIRT") is None


class _FakeResponse:
    def __init__(self, data: bytes, start: int = 0, status: int = 200):
        self._data = data[start:]
        self.status_code = 206 if start and status == 200 else status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk):
        for i in range(0, len(self._data), chunk):
            yield self._data[i : i + chunk]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_file_resumes_and_checks_size(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    payload = bytes(range(256)) * 40
    calls = []

    def fake_get(url, stream, headers, timeout):
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        calls.append(start)
        return _FakeResponse(payload, start)

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "a.mrc"
    (tmp_path / "a.mrc.part").write_bytes(payload[:1000])  # interrupted earlier
    dpd.download_file("https://x/a.mrc", dest, expected_size=len(payload))
    assert dest.read_bytes() == payload
    assert calls == [1000]
    assert not (tmp_path / "a.mrc.part").exists()

    # A complete file is not downloaded again.
    dpd.download_file("https://x/a.mrc", dest, expected_size=len(payload))
    assert calls == [1000]


def test_download_file_fails_on_short_file(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(b"abc"))
    monkeypatch.setattr(dpd.time, "sleep", lambda s: None)
    with pytest.raises(IOError):
        dpd.download_file("https://x/b", tmp_path / "b", expected_size=10, retries=2)


def test_star_round_trip_through_tomocpt_reader(tmp_path):
    """The STAR file the script writes gives back the same voxel coordinates in tomocpt."""
    pytest.importorskip("starfile")
    from tomocpt.labels.star import Star

    pts = {"TS_1": np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])}
    path = tmp_path / "ribosome.star"
    dpd.write_star(path, dpd.particles_table(pts), pixel_size=10.012)
    star = Star(path)
    subset, class_id = star.get_particle_subset("all")
    coords = star.get_shifted_scaled_coordinates(subset, tomogram_pixel_size=10.012)
    assert class_id == "all"
    assert np.array_equal(coords["TS_1"], np.array([[10, 20, 30], [40, 50, 60]]))


def test_download_file_restarts_when_partial_file_is_too_large(tmp_path, monkeypatch):
    requests = pytest.importorskip("requests")
    payload = b"0123456789"
    monkeypatch.setattr(requests, "get", lambda url, stream, headers, timeout: _FakeResponse(payload))
    dest = tmp_path / "c.mrc"
    (tmp_path / "c.mrc.part").write_bytes(b"x" * 50)
    dpd.download_file("https://x/c", dest, expected_size=len(payload))
    assert dest.read_bytes() == payload
