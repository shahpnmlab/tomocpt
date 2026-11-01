"""Size-discrimination loss ablation on the frozen G6-C8 phantom (joint model, one seed).

Arms (all softmax head + cross_class_neg_weight=0.25):
  A  baseline (current v2 loss)
  B  A + class-balanced voxel CE on object cores (max target > 0.5), softmax over [bg=0, classes]
  C  B + object-level CE: logits box-averaged over radius ~d/4 at true particle centres
Scratch only: monkeypatches BasePickingModel.training_step; nothing in the repo changes.
"""
import json, shutil, sys, tempfile, time
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path("/home/pranav/Desktop/pranav/software/tomocpt")
sys.path[:0] = [str(REPO / "tests"), str(REPO), str(Path(__file__).parent)]
from gate6 import test_g6_learning as L
from gate6 import g6_helpers as H
from tomocpt.networks.pickingModel import BasePickingModel
from g6_confusion_lib import diag

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ARMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["A", "B", "C"]
import os
K_CCN = float(os.environ.get("K_CCN", "0.25"))
LAMBDA_CORE, LAMBDA_OBJ = 1.0, 1.0
N_PART = len(L.PARTICLES)
ARM = {"name": "A"}


def _with_bg(logits, avail):
    B, C = logits.shape[:2]
    m = avail.view(B, C, *([1] * (logits.ndim - 2)))
    x = torch.where(m, logits.float(), torch.full_like(logits.float(), -1e4))
    return torch.cat([torch.zeros_like(x[:, :1]), x], 1)  # [B, 1+C, ...]


def core_ce(logits, y, avail):
    """Class-balanced CE where the strongest target > 0.5; label = argmax class (+1 for bg slot)."""
    z = _with_bg(logits, avail)
    ymax, yarg = y.max(1)
    core = ymax > 0.5
    if not bool(core.any()):
        return 0 * logits.sum()
    ce = F.cross_entropy(z, yarg + 1, reduction="none")  # [B, D, H, W]
    terms = [ce[core & (yarg == c)].mean() for c in range(y.shape[1]) if bool((core & (yarg == c)).any())]
    return torch.stack(terms).mean()


def object_ce(logits, y, avail, d_vox):
    """CE at particle centres (local maxima >= 0.5 of each particle target), logits box-averaged
    with half-width ~d/4 of the true class."""
    z = _with_bg(logits, avail)
    terms = []
    for c in range(N_PART):
        yc = y[:, c:c + 1]
        peak = (yc == F.max_pool3d(yc, 3, 1, 1)) & (yc >= 0.5)
        if not bool(peak.any()):
            continue
        r = max(1, int(round(float(d_vox[0, c]) / 4)))
        zp = F.avg_pool3d(z, 2 * r + 1, 1, r, count_include_pad=False)  # [B, 1+C, D, H, W]
        b, _, i, j, k = peak.nonzero(as_tuple=True)
        terms.append(F.cross_entropy(zp[b, :, i, j, k], torch.full_like(b, c + 1)))
    if not terms:
        return 0 * logits.sum()
    return torch.stack(terms).mean()


def training_step(self, batch, batch_idx):
    x, y_raw = self.resolve_batch(batch)
    logits = self(x, apix=self._batch_apix(batch))
    y_pred, y = self._probabilities(logits, y_raw, batch)
    loss = self.multiclass_loss(y_pred, y, batch)
    avail = batch["class_available"].bool()
    if ARM["name"] in ("B", "C"):
        loss = loss + LAMBDA_CORE * core_ce(logits, y_raw, avail)
    if ARM["name"] == "C":
        loss = loss + LAMBDA_OBJ * object_ce(logits, y_raw, avail, batch["class_d_vox"])
    self.log("loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
    return loss


BasePickingModel.training_step = training_step

_orig_cfg = H.v2_config_with_filaments


def cfg_softmax(tmp, particles, filaments, cond=False, class_head="sigmoid"):
    cfg = _orig_cfg(tmp, particles, filaments, cond=cond, class_head="softmax")
    cfg.train.particles.cross_class_neg_weight = K_CCN
    return cfg


H.v2_config_with_filaments = cfg_softmax

phs, held = L._data(SEED)
table = L.PARTICLES + L.FILAMENTS
for arm in ARMS:
    ARM["name"] = arm
    scratch = Path(tempfile.mkdtemp(prefix=f"g6abl_{arm}_"))
    t0 = time.time()
    m = L.train_picker(scratch / "joint", L.PARTICLES, L.FILAMENTS, [L.Tomo(p, table) for p in phs], L.STEPS, SEED)
    res = {"arm": arm, "seed": SEED, "minutes": round((time.time() - t0) / 60, 1),
           "summary": L.evaluate(m, held, table), **diag(m, held, table)}
    print("ABL " + json.dumps(res), flush=True)
    del m
    torch.cuda.empty_cache()
    shutil.rmtree(scratch, ignore_errors=True)
