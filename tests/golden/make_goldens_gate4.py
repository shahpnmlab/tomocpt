"""Generate the Gate 4 v1 golden (G4-C1) with ``v2-gate-0-base`` code plus the unpad fix.

Human decision (27 Oct 2026): the legacy ``unpad`` axis-order defect is fixed in Gate 4 for v1
and v2, so the golden is "base + unpad fix". The script patches
``tomocpt.predict.helpers.unpad`` at runtime with ``gate4_common.fixed_unpad`` (the base
worktree's files are not edited); everything else is base code.

Run in a worktree of the tag, with that worktree first on the import path::

    cd ../tomocpt-base
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python ../tomocpt/tests/golden/make_goldens_gate4.py \\
        --expect-root $PWD --legacy-ckpt ../tomocpt/tests/.cache/legacy/legacy_picking_<...>.ckpt

``--legacy-ckpt`` is the tiny legacy checkpoint of ``tests/utils/legacy.py`` (built with base
code). Its network weights are replaced by ``gate3_common.reinit_`` values
(``gate4_common.make_reinit_ckpt``), so the golden does not depend on that checkpoint's fit.

Output (next to this script): ``gate4_v1_infer.npz`` with, per case of
``gate4_common.C1_CASES`` (``<case>_map``: the confidence map MRC that ``infer_tomos`` writes,
as stored (float16); ``<case>_coords``: ``[N, 4]`` (z, y, x, score) returned by ``infer_tomos``;
``<case>_map_apix``), and ``gate4_v1_infer.json`` (provenance and summaries).
"""

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate4_common as g4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-root", required=True)
    ap.add_argument("--legacy-ckpt", required=True)
    args = ap.parse_args()

    import tomocpt

    root = Path(args.expect_root).resolve()
    got = Path(tomocpt.__file__).resolve()
    if root not in got.parents:
        sys.exit(f"tomocpt imported from {got}, expected under {root}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()

    import torch

    import tomocpt.predict.helpers as base_helpers

    base_helpers.unpad = g4.fixed_unpad(base_helpers.unpad)
    torch.manual_seed(0)
    arrays, summary = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = g4.make_reinit_ckpt(args.legacy_ckpt, tmp / "reinit.ckpt")
        tomo = g4.write_c1_tomogram(tmp / "tomograms" / "c1tomo.mrc")
        for case in g4.C1_CASES:
            cmap, coords, map_apix, _ = g4.run_infer_tomos(ckpt, tomo, tmp / f"pred_{case}", case)
            arrays[f"{case}_map"] = cmap.astype(np.float16)
            arrays[f"{case}_coords"] = coords
            arrays[f"{case}_map_apix"] = np.float64(map_apix)
            summary[case] = {"map_sum": float(cmap.astype(np.float64).sum()), "map_shape": list(cmap.shape),
                             "n_coords": int(len(coords)), "coords": coords.tolist(), "map_apix": map_apix}
    np.savez_compressed(HERE / "gate4_v1_infer.npz", **arrays)
    out = {
        "provenance": {
            "base_ref": "v2-gate-0-base",
            "code": "base + unpad fix (human decision, 27 Oct 2026)",
            "base_commit": commit,
            "tomocpt_file": str(got),
            "legacy_ckpt": Path(args.legacy_ckpt).name,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "num_threads": g4.C1_NUM_THREADS,
        },
        "cases": {k: dict(v) for k, v in g4.C1_CASES.items()},
        "summary": summary,
    }
    (HERE / "gate4_v1_infer.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
