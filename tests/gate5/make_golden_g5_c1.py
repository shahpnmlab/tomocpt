"""Generate the G5-C1 golden (SSL v1 ``_step``) with ``v2-gate-0-base`` code.

Run in a worktree of the tag, with that worktree first on the import path::

    cd ../tomocpt-base
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python ../tomocpt/tests/gate5/make_golden_g5_c1.py --expect-root $PWD

Output (next to this script): ``g5_c1_golden.json`` with
* ``step`` - ``[total, rot, contrastive, recon]`` of ``SelfSupervisedModel._step`` on
  ``g5_common.ssl_batch_input()`` for the tiny model (``feature_size=12``, dropout 0,
  ``use_checkpoint=False``, CPU fp32) with ``g5_common.reinit_by_name_`` weights, numpy and
  torch seeded with ``g5_common.STEP_SEED`` right before the call;
* ``forward_sums`` - sums of the eval-mode ``(rot, contrastive, recon)`` outputs on the same
  input (a cross-check of the weights);
* provenance.
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g5_common as g5  # noqa: E402


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

    import torch

    torch.set_num_threads(4)
    from tomocpt.mainConfig import mainConfig
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    g5.configure_tiny_ssl(mainConfig)
    torch.manual_seed(0)
    model = SelfSupervisedModel(lr=1e-3, train_continue=None)
    g5.reinit_by_name_(model)
    x = g5.ssl_batch_input()
    sums = g5.ssl_v1_forward_sums(model, x)
    step = g5.ssl_v1_step(model, x)
    step_again = g5.ssl_v1_step(model, x)
    assert np.allclose(step, step_again, rtol=1e-6), (step, step_again)

    out = {
        "provenance": {
            "base_ref": "v2-gate-0-base",
            "base_commit": commit,
            "tomocpt_file": str(got),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "x_sum": float(x.astype(np.float64).sum()),
        "param_names": sorted(n for n, _ in model.named_parameters()),
        "forward_sums": sums,
        "step": step,
    }
    (HERE / "g5_c1_golden.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("forward_sums", "step")}, indent=1))


if __name__ == "__main__":
    main()
