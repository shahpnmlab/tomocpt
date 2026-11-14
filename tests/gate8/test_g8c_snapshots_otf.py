"""Gate 8c: ``train.snapshot_chunks`` with ``train.sampling=on_the_fly`` (found by the DS-10490 run,
13 Nov 2026: the panels read chunk files, which the on-the-fly data module does not have)."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def phantom_store(tmp_path_factory):
    from gate8 import g8c_helpers as G

    root = tmp_path_factory.mktemp("g8c_snap")
    return G.build(G.write_inputs(root / "in"), root / "store", storage="tomogram")


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_G8c_snapshots_on_the_fly(phantom_store, tmp_path, device):
    from gate8 import g8c_helpers as G

    from tomocpt.training.snapshots import SNAPSHOT_DIR_NAME, VolumeSnapshots

    dm = G.otf_data(phantom_store, augment_device=device)
    trainer = SimpleNamespace(global_rank=0, datamodule=dm, logger=SimpleNamespace(log_dir=str(tmp_path)),
                              default_root_dir=str(tmp_path))
    cb = VolumeSnapshots(per_class=1)
    cb.on_fit_start(trainer, None)
    assert cb.batch is not None
    n = cb.batch["input_data"].shape[0]
    assert cb.batch["input_data"].shape[1:] == (1, 64, 64, 64)
    assert cb.batch["target_data"].shape[:2] == (n, len(dm.class_table))
    z = np.load(tmp_path / SNAPSHOT_DIR_NAME / "chunks.npz")
    assert set(z["split"]) <= {"train", "val"} and "train" in set(z["split"])
    for k, c in enumerate(z["cls"]):  # each crop was chosen for a class it shows
        assert z["target"][k, c].max() > 0
