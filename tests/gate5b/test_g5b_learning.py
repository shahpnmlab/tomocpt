"""G5b-C6, G5b-C7, G5b-C8 (slow, CUDA): SSL v2 learning with the Gate 5b defaults.

Training reuses the frozen Gate 5 machinery (``tests/gate5/test_g5_learning.py``): the Gate 5
phantom set (128^3 sources at 10 A/px: 100 A solid, 200 A solid, 200 A hollow, 300 A solid, one
250 A tube; wedge 60 deg, SNR 1), items from ``make_training_pair``, ``SelfSupervisedModel`` with
``train.ssl.version = 2`` and every other SSL setting at its (Gate 5b) default, GPU bf16-mixed,
gradient clip 1.0, lr 1e-3, batch 8, ``STEPS`` steps. One phantom training per seed (C6); the
seed-0 model also serves C7 and C8 (C15); one noise training serves C8 (C16).

C6 metrics on held-out phantom pairs (3 unseen sources, ``N_EVAL_BATCHES`` batches of
``EVAL_BATCH`` pairs, run through ``trainer.validate`` in eval mode, bf16):
* global NT-Xent: the model's ``ProjectionHead`` outputs recorded by a forward hook during the
  validation step (all calls of one step concatenated = ``[view a rows; view b rows]``), scored with
  the Gate 5 ``nt_xent(z_a, z_b, CONTRAST_LOSS_TEMPERATURE, 0)`` **without the queue**; chance =
  ``ln(2B - 1)``.
* dense InfoNCE: the ``DenseProjectionHead`` outputs recorded the same way (a ``[2B, D, d, h, w]``
  map scored with the Gate 5 ``dense_info_nce`` at the batch's ``pts_a``/``pts_b``, temperature 0.1;
  or per-anchor embeddings ``[2B, n, D]`` / ``[2B * n, D]`` scored with the same symmetric InfoNCE);
  chance = ``ln(B * n_anchors)``.
* global embedding spread: L2-normalised projections of all held-out views,
  ``sqrt(sum_d var_d)`` (RMS distance to their mean; ~1 when spread, 0 when collapsed; the per-dim
  std would be ~1/sqrt(256) = 0.06 even for uniformly spread embeddings, so the total is used).
* ``hidden[4]``: ``p = avgpool(swinViT(x)[4])`` of every held-out ``input_a`` view;
  ``mean_c std_over_samples(p) / mean(|p|)`` (CZII collapse: 0.004 / 0.76 = 0.005).

Thresholds are the spec's (0.8 x chance, 0.1, 0.1). They have NOT been calibrated by the TA (no
Gate 5b implementation exists); a correct implementation should pass >= 9/10 seeds. Margins: the
collapsed CZII run sat at 1.00 x chance and a spread of ~6e-4, i.e. far from the thresholds.
Knobs: ``TOMOCPT_G5B_SEEDS`` (default ``0,1,2``), ``TOMOCPT_G5B_STEPS`` (default 1000; Gate 5
used 1500). Runtime on an RTX A6000: ~0.45 s per step, so ~8 min per training, 4 trainings.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gate5 import g5_helpers as H
from gate5 import test_g5_learning as L5

pytestmark = pytest.mark.slow

SEEDS = tuple(int(s) for s in os.environ.get("TOMOCPT_G5B_SEEDS", "0,1,2").split(","))
STEPS = int(os.environ.get("TOMOCPT_G5B_STEPS", "1000"))
EVAL_BATCH = 16
N_EVAL_BATCHES = 4
DENSE_TEMPERATURE = 0.1
REPORT = Path(__file__).resolve().parents[2] / ".claude" / "reports" / "gate_05_report.json"


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.fixture(scope="module")
def phantom_model(tmp_path_factory):
    """``get(seed)``: SSL v2 model trained on the Gate 5 phantom set (cached per seed)."""
    _need_cuda()
    cache = {}

    def get(seed):
        if seed not in cache:
            tmp = tmp_path_factory.mktemp(f"g5b_phantom_{seed}")
            cache[seed] = L5.train_ssl_v2(tmp, L5.phantom_sources(10, 1000 * seed), seed, steps=STEPS)
        # trainer.validate (C6) moves the model back to the CPU at teardown
        return cache[seed].cuda().eval()

    return get


@pytest.fixture(scope="module")
def noise_model(tmp_path_factory):
    _need_cuda()
    seed = SEEDS[0]
    tmp = tmp_path_factory.mktemp("g5b_noise")
    return L5.train_ssl_v2(tmp, L5.noise_sources(10, 1000 * seed), seed + 1, steps=STEPS)


# --------------------------------------------------------------------------------------
# held-out evaluation through the model's own validation step
# --------------------------------------------------------------------------------------

def _info_nce(ea, eb, t):
    ea, eb = F.normalize(ea.float(), dim=1), F.normalize(eb.float(), dim=1)
    logits = ea @ eb.t() / t
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _dense_loss(f, pts_a, pts_b, n_pairs):
    from tomocpt.networks.ssl.losses import dense_info_nce

    if f.ndim == 5:
        return float(dense_info_nce(f[:n_pairs], f[n_pairs:], pts_a.to(f.device), pts_b.to(f.device),
                                    DENSE_TEMPERATURE, view_size=64))
    e = f.reshape(2, -1, f.shape[-1])
    return float(_info_nce(e[0], e[1], DENSE_TEMPERATURE))


@torch.no_grad()
def heldout_metrics(model, tmp, seed):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    from tomocpt.networks.ssl.heads import DenseProjectionHead, ProjectionHead
    from tomocpt.networks.ssl.losses import nt_xent

    ssl = H.resolved(model.hparams.config.train.ssl)
    t_global = float(model.hparams.config.train.network.CONTRAST_LOSS_TEMPERATURE)
    sources = L5.phantom_sources(3, 1000 * seed + 500)
    ds = L5.PairDataset(sources, EVAL_BATCH * N_EVAL_BATCHES, seed + 4242, ssl)
    rec = {"z": [], "f": [], "pts": []}
    cur = {}

    def hook(key):
        def _h(mod, inp, out):
            cur.setdefault(key, []).append(out.detach().float())
        return _h

    hooks = [m.register_forward_hook(hook("z")) for m in model.modules() if isinstance(m, ProjectionHead)]
    hooks += [m.register_forward_hook(hook("f")) for m in model.modules() if isinstance(m, DenseProjectionHead)]
    assert len(hooks) == 2, "expected one ProjectionHead and one DenseProjectionHead"

    class _Rec(pl.Callback):
        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            cur.clear()

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            rec["z"].append(torch.cat(cur["z"]))
            rec["f"].append(torch.cat(cur["f"]))
            rec["pts"].append((batch["pts_a"].float(), batch["pts_b"].float()))

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", callbacks=[_Rec()],
                         logger=TensorBoardLogger(str(tmp / "tb_eval")), enable_progress_bar=False,
                         enable_model_summary=False)
    try:
        trainer.validate(model, dataloaders=DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=8))
    finally:
        for h in hooks:
            h.remove()

    B = EVAL_BATCH
    g_loss = [float(nt_xent(z[:B], z[B:], t_global, 0.0)) for z in rec["z"]]
    d_loss = [_dense_loss(f, pa, pb, B) for f, (pa, pb) in zip(rec["f"], rec["pts"])]
    zn = F.normalize(torch.cat(rec["z"]), dim=1)
    spread = float(zn.var(dim=0).sum().sqrt())

    model.cuda().eval()
    net = model.model
    pooled = []
    for i in range(len(ds)):
        x = L5._vol(ds[i]["input_a"]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = net.swinViT(x, net.normalize)
        pooled.append(hs[4].float().mean(dim=(2, 3, 4)))
    p = torch.cat(pooled)
    h4_ratio = float(p.std(dim=0).mean() / p.abs().mean())
    return dict(global_vs_chance=float(np.mean(g_loss)) / math.log(2 * B - 1),
                dense_vs_chance=float(np.mean(d_loss)) / math.log(B * int(ssl.n_anchors)),
                global_spread=spread, hidden4_ratio=h4_ratio)


# --------------------------------------------------------------------------------------
# G5b-C6
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_G5b_C6_contrastive_below_chance_and_no_collapse(phantom_model, tmp_path, seed):
    model = phantom_model(seed)
    m = heldout_metrics(model, tmp_path, seed)
    print(f"G5B_RESULT C6 seed={seed} steps={STEPS} {m}")
    assert m["global_vs_chance"] < 0.8, m
    assert m["dense_vs_chance"] < 0.8, m
    assert m["global_spread"] > 0.1, m
    assert m["hidden4_ratio"] > 0.1, m


# --------------------------------------------------------------------------------------
# G5b-C7
# --------------------------------------------------------------------------------------

def _gate5_c17_value() -> float:
    """The Gate 5 C17 kNN score if ``gate_05_report.json`` records one (a metric whose key contains
    ``C17``), else the Gate 5 threshold 0.8 (the report at the time of writing records none)."""
    try:
        metrics = json.loads(REPORT.read_text()).get("metrics", {})
    except (OSError, ValueError):
        return 0.8
    vals = [float(v) for k, v in metrics.items() if "C17" in k and isinstance(v, (int, float))]
    return max([0.8] + vals)


def test_G5b_C7_dense_embeddings_still_separate_species(phantom_model, tmp_path):
    seed = SEEDS[0]
    model = phantom_model(seed)
    assert H.resolved(model.hparams.config.train.ssl).dense_grad_scale == 1.0  # full dense gradient
    phantoms = [L5._phantom((96,) * 3, L5.EVAL_SPECIES, (), 1000 * seed + 700 + i) for i in range(6)]
    trained, trained_all = L5._knn_score(model, phantoms)
    random_init, random_all = L5._knn_score(L5.untrained_ssl_v2(tmp_path, seed), phantoms)
    floor = _gate5_c17_value()
    print(f"G5B_RESULT C7 seed={seed} trained={trained_all} random={random_all} floor={floor}")
    assert trained >= floor
    assert trained - random_init >= 0.2  # Gate 5 C17 second condition


# --------------------------------------------------------------------------------------
# G5b-C8
# --------------------------------------------------------------------------------------

def test_G5b_C8_scale_learning_still_holds(phantom_model):
    seed = SEEDS[0]
    held_out = L5.phantom_sources(3, 1000 * seed + 500)
    pred, true = L5.predicted_ratios(phantom_model(seed), held_out, 180, seed + 31)
    mae = float(np.mean(np.abs(pred - true)))
    print(f"G5B_RESULT C8/C15 seed={seed} mae={mae:.4f} baseline(0)={np.mean(np.abs(true)):.4f}")
    assert mae < 0.15


def test_G5b_C8_no_scale_shortcut_on_noise(noise_model):
    seed = SEEDS[0]
    held_out = L5.noise_sources(3, 1000 * seed + 500)
    pred, true = L5.predicted_ratios(noise_model, held_out, 200, seed + 37)
    r = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 else 0.0
    print(f"G5B_RESULT C8/C16 seed={seed} pearson={r:.4f}")
    assert abs(r) < 0.2
