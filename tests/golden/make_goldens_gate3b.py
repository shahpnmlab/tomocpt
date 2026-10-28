"""Generate the G3b-C1 golden with the Gate 3 loss (``v2-gate-4-passed`` code, before Gate 3b).

Run from the repository root at a commit without ``cross_class_neg_weight``::

    CUDA_VISIBLE_DEVICES= python tests/golden/make_goldens_gate3b.py

Output (next to this script): ``gate3b_c1_loss.npz`` with, for every case of
``gate3b_common.CASES``: ``<case>_ck<0|1>_loss`` (float64 scalar of the float32 loss, for
``checkpoint`` False / True) and ``<case>_grad`` (float32 gradient w.r.t. ``y_pred`` with
``checkpoint=False``; the generator checks that ``checkpoint=True`` gives the identical array),
plus ``<case>_ytrue_sum`` / ``<case>_ypred_sum`` to check that the rebuilt inputs are the same,
and the commit in ``provenance``.
"""

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate3b_common as g  # noqa: E402


def main():
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    if "cross_class_neg_weight" in inspect.signature(multiclass_picking_loss).parameters:
        sys.exit("run this on the Gate 3 loss (before Gate 3b)")
    out = {}
    for name in g.CASES:
        y_pred, y_true, *_ = g.case_inputs(name)
        out[f"{name}_ytrue_sum"] = np.float64(y_true.double().sum())
        out[f"{name}_ypred_sum"] = np.float64(y_pred.double().sum())
        grads = {}
        for ck in (True, False):
            loss, grads[ck] = g.loss_and_grad(multiclass_picking_loss, name, ck)
            out[f"{name}_ck{int(ck)}_loss"] = np.float64(loss)
            print(name, ck, loss, float(grads[ck].abs().sum()))
        assert np.array_equal(grads[True].numpy(), grads[False].numpy())
        out[f"{name}_grad"] = grads[False].numpy().astype(np.float32)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=HERE).stdout.strip()
    import tomocpt
    out["provenance"] = np.array(f"commit {commit}; tomocpt from {Path(tomocpt.__file__).parent}")
    np.savez_compressed(HERE / "gate3b_c1_loss.npz", **out)


if __name__ == "__main__":
    main()
