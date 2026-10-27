"""Generate the Gate 3 legacy golden with ``v2-gate-0-base`` code.

Run in a worktree of the tag, with that worktree first on the import path::

    cd ../tomocpt-base
    PYTHONPATH=$PWD python ../tomocpt/tests/golden/make_goldens_gate3.py --expect-root $PWD

Output (next to this script): ``gate3_goldens.json`` with
* ``G3_C1_training_step_loss`` - the loss ``BasePickingModel.training_step`` returns on
  ``gate3_common.v1_batch()`` for the tiny single-class model (``feature_size=12``, dropout 0,
  ``use_checkpoint=False``, CPU fp32) with ``gate3_common.reinit_`` weights, through
  ``gate3_common.first_training_step_loss`` (a one-batch ``trainer.fit``);
* ``G3_C1_logits_sum`` - sum of the eval-mode logits of that model on the batch input
  (a cross-check of the weights);
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
import gate3_common as g3  # noqa: E402

TINY = dict(in_channels=1, out_channels=1, feature_size=12, use_v2=True, use_checkpoint=False,
            drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0)


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
    from tomocpt.defaultConfigs.network_config import PrecisionType
    from tomocpt.mainConfig import mainConfig
    from tomocpt.networks.pickingModel import BasePickingModel

    for k, v in TINY.items():
        setattr(mainConfig.train.network.SwinUNETR, k, v)
    mainConfig.train.network.TORCH_FLOAT_PRECISION = PrecisionType.float
    mainConfig.train.batch_size = 2
    mainConfig.train.use_gpus = False
    mainConfig.train.n_cpus_for_dataloading = 0

    torch.manual_seed(0)
    model = BasePickingModel(lr=1e-3, train_continue=None)
    g3.reinit_(model.model)
    batch = g3.v1_batch()
    model.eval()
    with torch.no_grad():
        logits_sum = float(model(torch.from_numpy(batch["input_data"])).double().sum())
    model.train()
    loss = g3.first_training_step_loss(model, batch)

    out = {
        "provenance": {
            "base_ref": "v2-gate-0-base",
            "base_commit": commit,
            "tomocpt_file": str(got),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "G3_C1_training_step_loss": loss,
        "G3_C1_logits_sum": logits_sum,
        "G3_C1_input_sum": float(batch["input_data"].astype(np.float64).sum()),
        "G3_C1_target_sum": float(batch["target_data"].astype(np.float64).sum()),
    }
    (HERE / "gate3_goldens.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
