"""G4-C1: v1 inference lock against the ``v2-gate-0-base`` golden."""

import json

import numpy as np
import pytest
import torch

from gate4 import g4_helpers as H


def test_G4_C1_v1_infer_tomos_matches_base_golden(tmp_path, legacy_ckpt):
    """Legacy tiny checkpoint (weights re-initialised deterministically) on a synthetic
    tomogram through ``infer_tomos`` (no Dask): the saved confidence map equals the golden
    (atol 1e-5) and the coordinates are the same (count, atol 1e-3 voxel)."""
    g4 = H.load_gate4_common()
    golden = np.load(H.GOLDEN_DIR / "gate4_v1_infer.npz")
    meta = json.loads((H.GOLDEN_DIR / "gate4_v1_infer.json").read_text())
    ckpt = g4.make_reinit_ckpt(legacy_ckpt, tmp_path / "reinit.ckpt")
    tomo = g4.write_c1_tomogram(tmp_path / "tomograms" / "c1tomo.mrc")
    n_threads = torch.get_num_threads()
    try:
        for case in g4.C1_CASES:
            assert meta["cases"][case] == g4.C1_CASES[case]
            cmap, coords, map_apix, _ = g4.run_infer_tomos(ckpt, tomo, tmp_path / f"pred_{case}", case)
            ref_map = golden[f"{case}_map"].astype(np.float32)
            assert cmap.shape == ref_map.shape
            np.testing.assert_allclose(cmap, ref_map, rtol=0, atol=1e-5, err_msg=f"{case}: confidence map")
            assert map_apix == pytest.approx(float(golden[f"{case}_map_apix"]))
            ref = golden[f"{case}_coords"]
            assert len(ref) > 0, "the golden must contain peaks"
            assert coords.shape == ref.shape, f"{case}: {len(coords)} peaks, golden {len(ref)}"
            np.testing.assert_allclose(coords[:, :3], ref[:, :3], rtol=0, atol=1e-3, err_msg=f"{case}: coords")
            np.testing.assert_allclose(coords[:, 3], ref[:, 3], rtol=1e-4, atol=1e-5, err_msg=f"{case}: scores")
    finally:
        torch.set_num_threads(n_threads)
