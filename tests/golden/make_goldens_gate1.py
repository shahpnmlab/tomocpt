"""Generate the Gate 1 legacy goldens with ``v2-gate-0-base`` code.

Run in a worktree of the tag, with that worktree first on the import path::

    cd ../tomocpt-base
    PYTHONPATH=$PWD python ../tomocpt/tests/golden/make_goldens_gate1.py --expect-root $PWD

Outputs (next to this script):
* ``gate1_spheres.npz``  - G1-C1: ``generate_gaussian_sphere(R, 2R+1)`` for R in {5, 10, 17}.
* ``gate1_forward.npz``  - G1-C3: tiny ``MySwinUNETR`` (seed 0, eval, CPU fp32) on
  ``gate1_common.forward_input()``: ``logits``, ``hidden4``.
* ``gate1_goldens.json`` - G1-C2 loss value, G1-C8 transform description, provenance.
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
import gate1_common as gc  # noqa: E402

TINY = dict(img_size=64, in_channels=1, out_channels=1, feature_size=12, use_v2=True,
            use_checkpoint=False, drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0)


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
    from tomocpt import constants
    from tomocpt.dataManager.dataloading import build_training_transforms
    from tomocpt.defaultConfigs.train_config import AugmentationConfig
    from tomocpt.labels.helpers import generate_gaussian_sphere
    from tomocpt.mainConfig import mainConfig
    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.networks.swinunetr import MySwinUNETR

    # G1-C1
    spheres = {r: np.array(generate_gaussian_sphere(r, 2 * r + 1)) for r in gc.SPHERE_RADII}
    np.savez_compressed(HERE / "gate1_spheres.npz", **{f"R{r}": s for r, s in spheres.items()})

    # G1-C2
    for k, v in TINY.items():
        if k != "img_size":
            setattr(mainConfig.train.network.SwinUNETR, k, v)
    mainConfig.train.batch_size = 2
    mainConfig.train.use_gpus = False
    model = BasePickingModel(lr=1e-3, train_continue=None)
    y_pred, y_true = gc.loss_inputs(spheres)
    with torch.no_grad():
        loss = float(model.loss(torch.from_numpy(y_pred), torch.from_numpy(y_true)))

    # G1-C3
    torch.manual_seed(0)
    net = MySwinUNETR(**TINY).eval()
    with torch.no_grad():
        logits, hidden4 = net(torch.from_numpy(gc.forward_input()))
    np.savez_compressed(HERE / "gate1_forward.npz", logits=logits.numpy(), hidden4=hidden4.numpy())

    # G1-C8
    transforms = gc.describe_transform(build_training_transforms(AugmentationConfig()))

    out = {
        "provenance": {
            "base_ref": "v2-gate-0-base",
            "base_commit": commit,
            "tomocpt_file": str(got),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "G1_C2_loss": {"value": loss, "y_true_sum": float(y_true.sum()), "y_pred_sum": float(y_pred.astype(np.float64).sum())},
        "G1_C3_forward": {"logits_sum": float(logits.double().sum()), "hidden4_sum": float(hidden4.double().sum())},
        "G1_C8_transforms": transforms,
        "G1_C8_contrast_inversion_prob": constants.CONTRAST_INVERSION_PROB,
    }
    (HERE / "gate1_goldens.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["G1_C2_loss"]), out["G1_C3_forward"])


if __name__ == "__main__":
    main()
