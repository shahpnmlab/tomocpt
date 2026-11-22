"""Fix 22 Nov 2026: tomogram files are found by exact name, not by the substring glob ``*{name}*``.

The DS-10490 boss run paired the coordinates of ``sample_25`` with ``sample_257.mrc`` (whichever the
filesystem listed first), and 27 tomograms whose name is a prefix of another lost their particles.
"""
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest

from tomocpt.labels.helpers import find_tomogram_file


def _touch(d: Path, *names):
    for n in names:
        (d / n).write_bytes(b"")


def test_exact_name_wins_over_longer_names(tmp_path):
    _touch(tmp_path, "sample_257.mrc", "sample_25.mrc", "sample_2.mrc", "sample_250.mrc")
    for name in ("sample_25", "sample_257", "sample_2", "sample_250"):
        assert find_tomogram_file(tmp_path, name) == tmp_path / f"{name}.mrc"


def test_prefix_of_another_name_is_not_matched(tmp_path):
    _touch(tmp_path, "sample_257.mrc")
    assert find_tomogram_file(tmp_path, "sample_25") is None


def test_name_as_token_inside_a_longer_stem(tmp_path):
    _touch(tmp_path, "TS_sample_25_rec.mrc", "TS_sample_257_rec.mrc")
    assert find_tomogram_file(tmp_path, "sample_25") == tmp_path / "TS_sample_25_rec.mrc"
    assert find_tomogram_file(tmp_path, "sample_257") == tmp_path / "TS_sample_257_rec.mrc"


def test_ambiguous_token_matches_raise(tmp_path):
    _touch(tmp_path, "a_sample_25.mrc", "b_sample_25.mrc")
    with pytest.raises(ValueError, match="Several tomogram files"):
        find_tomogram_file(tmp_path, "sample_25")


def test_mrc_before_rec_and_other_files_ignored(tmp_path):
    _touch(tmp_path, "t1.rec", "t1.mrc", "t1.csv", "t10.mrc")
    assert find_tomogram_file(tmp_path, "t1") == tmp_path / "t1.mrc"
    _touch(tmp_path, "t2.rec")
    assert find_tomogram_file(tmp_path, "t2") == tmp_path / "t2.rec"


def test_v2_tracking_rows_pair_each_tomogram_with_its_own_coordinates(tmp_path):
    """End to end through the pipeline-v2 tracking columns: every coordinate file belongs to the
    tomogram of the same name, also when one name is a prefix of others."""
    from tomocpt.labels.helpers import _v2_tracking_columns

    tomos = tmp_path / "tomos"
    tomos.mkdir()
    names = ["s_2", "s_25", "s_257", "s_1", "s_11"]
    for n in names:
        with mrcfile.new(tomos / f"{n}.mrc") as m:
            m.set_data(np.zeros((4, 4, 4), np.float32))
            m.voxel_size = 6.8
    pairs = {n: np.full((i + 1, 3), float(i)) for i, n in enumerate(names)}
    extra = _v2_tracking_columns(pairs, tomos, tmp_path / "out", 1, 100.0, None, "1")
    for i, n in enumerate(names):
        df = pd.read_csv(extra[n]["coords_path"])
        assert Path(extra[n]["coords_path"]).stem == n
        assert len(df) == i + 1
