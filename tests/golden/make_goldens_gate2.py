"""Generate the Gate 2 legacy golden (G2-C1) with ``v2-gate-0-base`` code.

Run in a worktree of the tag, with that worktree first on the import path::

    cd ../tomocpt-base
    PYTHONPATH=$PWD python ../tomocpt/tests/golden/make_goldens_gate2.py --expect-root $PWD

Output (next to this script): ``gate2_v1_chunks.json``.

For each variant in ``gate2_common.VARIANTS`` it runs ``make_dataset(n_tomos=2)`` (default
species), the legacy ``prepare_data`` and ``do_chunking`` with its legacy arguments only, in
the controlled environment of ``gate2_common.inprocess_chunking`` (in-process Dask, CPU,
seeded drop RNG). It records every file under ``chunks_dir`` (relative path) with, for
``.nii.gz`` files, the decoded array's shape, dtype, sha256 of its bytes, sum and max; JSON
files parsed; other files as text.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(TESTS))  # utils.phantoms (frozen Gate 1 helper)

import gate2_common as gc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-root", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch

    import tomocpt

    root = Path(args.expect_root).resolve()
    got = Path(tomocpt.__file__).resolve()
    if root not in got.parents:
        sys.exit(f"tomocpt imported from {got}, expected under {root}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()

    out = {"variants": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        _, tdd = gc.prepare_v1_dataset(tmp)
        for variant in gc.VARIANTS:
            torch.manual_seed(0); np.random.seed(0); random.seed(0)
            chunks = tmp / f"chunks_{variant}"
            gc.run_v1_chunking(tdd, chunks, variant)
            out["variants"][variant] = gc.summarize_chunks(chunks)

    out["provenance"] = {
        "base_commit": commit,
        "tomocpt_file": str(got),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dataset": gc.DATASET_KWARGS,
        "drop_seed": gc.DROP_SEED,
        "variants": {k: v for k, v in gc.VARIANTS.items()},
    }
    path = HERE / "gate2_v1_chunks.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    n = {k: len(v) for k, v in out["variants"].items()}
    print(f"wrote {path} ({path.stat().st_size} bytes); files per variant: {n}")


if __name__ == "__main__":
    main()
