"""Generate the Gate 6 no-filament golden (G6-C1) with the ``v2-dev`` code before Gate 6.

G6-C1 compares against Gate 4 behaviour (Gates 2-5 on ``v2-dev``; no filament code exists),
not against ``v2-gate-0-base``. Run from the repository root on that commit, CPU only::

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python tests/golden/make_goldens_gate6.py --expect-root $PWD

Output (next to this script): ``gate6_c1.npz`` (arrays, see ``gate6_common``) and
``gate6_c1.json`` (provenance, file lists, the chunk manifest with relative paths).
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
import gate6_common as g6  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-root", required=True)
    args = ap.parse_args()

    import tomocpt

    root = Path(args.expect_root).resolve()
    got = Path(tomocpt.__file__).resolve()
    if root not in got.parents:
        sys.exit(f"tomocpt imported from {got}, expected under {root}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tomocpt"], cwd=root, capture_output=True,
                           text=True).stdout.strip()
    try:
        import tomocpt.labels.filaments  # noqa: F401
        sys.exit("tomocpt.labels.filaments exists: the G6-C1 golden must come from pre-Gate-6 code")
    except ImportError:
        pass

    with tempfile.TemporaryDirectory(prefix="g6_golden_") as tmp:
        arrays, info = g6.all_arrays(Path(tmp))

    np.savez_compressed(HERE / "gate6_c1.npz", **arrays)
    meta = {
        "criterion": "G6-C1",
        "commit": commit,
        "tomocpt_source_modified": bool(dirty),
        "reference": "v2-dev before Gate 6 (Gate 4/4b/5 behaviour), CPU, fp32, in-process",
        "command": "CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python tests/golden/make_goldens_gate6.py --expect-root $PWD",
        "n_arrays": len(arrays),
        "info": info,
    }
    (HERE / "gate6_c1.json").write_text(json.dumps(meta, indent=1, sort_keys=True, default=str))
    print(f"wrote {len(arrays)} arrays; commit {commit}")


if __name__ == "__main__":
    main()
