"""Generate the Gate 7 lock golden (G7-C1 state-dict keys, G7-C7 "both options off") with the
Gate 6 code (tag ``v2-gate-6-passed``).

Run with ``tomocpt`` imported from a worktree of that tag and the test helpers of this tree
(they are frozen and identical there), CPU only::

    git worktree add ../tomocpt-g6 v2-gate-6-passed
    cd ../tomocpt-g6 && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD:<this repo>/tests \\
        python <this repo>/tests/golden/make_goldens_gate7.py --expect-root $PWD

Output (next to this script): ``gate7_lock.json`` (records of ``gate7_common.CASES`` and
provenance).
"""

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/ (helpers as utils.*, gate3.*, gate6.*)
sys.path.insert(0, str(HERE))
import gate7_common as g7  # noqa: E402


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
        import tomocpt.training.discriminative  # noqa: F401
        sys.exit("tomocpt.training.discriminative exists: the golden must come from Gate 6 code")
    except ImportError:
        pass

    records = {}
    with tempfile.TemporaryDirectory(prefix="g7_golden_") as tmp:
        for case in g7.CASES:
            records[case] = g7.public(g7.lock_record(Path(tmp) / case, case))
    meta = {
        "criterion": "G7-C1, G7-C7 (both options off)",
        "commit": commit,
        "tomocpt_source_modified": bool(dirty),
        "reference": "v2-gate-6-passed, CPU, fp32, in-process",
        "records": records,
    }
    (HERE / "gate7_lock.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "state_dict_keys"} for k, v in records.items()},
                     indent=1))


if __name__ == "__main__":
    main()
