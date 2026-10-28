"""G5-C15, G5-C16, G5-C17 (slow, CUDA): SSL v2 learning tests with the tiny model (fs = 12).

Training data are built **in memory**: 128^3 source chunks at 10 A/px (``utils.phantoms``
phantoms, or wedged white noise for C16), robust-normalised, and items from
``make_training_pair(source=..., apix_ref=10, ssl_cfg=..., rng=...)``. Training runs
``SelfSupervisedModel`` (``train.ssl.version = 2``, all other SSL settings at their defaults)
through a Lightning trainer (GPU, bf16-mixed, gradient clip 1.0, lr 1e-3, batch 8).

* C15: held-out phantom pairs; the relative-scale prediction is read with the model's
  ``RelativeScaleHead`` on ``p = concat(avgpool(hidden[3]), avgpool(hidden[4]))`` of each view
  (``hidden`` = ``swinViT`` outputs); MAE of the ``log2`` ratio < 0.15.
* C16: the same pipeline trained and evaluated on pure (wedged) noise; |Pearson r| < 0.2.
* C17: held-out 96^3 phantoms with four species (100 A solid, 200 A solid, 200 A hollow,
  300 A solid); dense embeddings at the true centres (the ``DenseProjectionHead`` output on the
  FiLM-conditioned ``decoder2`` features, or those features themselves, whichever scores higher),
  leave-one-tomogram-out 5-NN (cosine), balanced accuracy >= 0.8 and >= 0.2 above the same
  measurement on a randomly initialised v2 network.

Thresholds are the spec's. They have NOT been calibrated by the TA (no implementation exists
yet); ``TOMOCPT_G5_SEED`` (default 0) selects the seed for the implementer's seed sweep.
Budget: < 15 min for the three tests on an RTX A6000 (two trainings of ``STEPS`` steps).
"""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gate5 import g5_helpers as H
from utils.phantoms import make_phantom
from utils.tiny import applied_main_config

pytestmark = pytest.mark.slow

SEED = int(os.environ.get("TOMOCPT_G5_SEED", "0"))
STEPS = int(os.environ.get("TOMOCPT_G5_STEPS", "1500"))
BATCH = 8
APIX = H.APIX_REF
#: (class, diameter A, count, hollow) per 128^3 training source
TRAIN_SPECIES = ((1, 100.0, 10, False), (2, 200.0, 5, False), (3, 200.0, 5, True), (4, 300.0, 3, False))
TRAIN_TUBES = ((5, 250.0, 1),)
#: per 96^3 held-out phantom (C17)
EVAL_SPECIES = ((1, 100.0, 5, False), (2, 200.0, 3, False), (3, 200.0, 3, True), (4, 300.0, 2, False))


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def _phantom(shape, species, tubes, seed):
    for k in range(20):
        try:
            return make_phantom(shape=shape, apix=APIX, species=species, tubes=tubes, wedge_deg=60.0,
                                snr=1.0, seed=seed * 101 + k)
        except RuntimeError:  # could not place every sphere; try another layout
            continue
    raise RuntimeError("phantom placement failed")


def phantom_sources(n, seed):
    return [H.robust_normalise(_phantom((128,) * 3, TRAIN_SPECIES, TRAIN_TUBES, seed + i).volume)
            for i in range(n)]


def noise_sources(n, seed):
    return [H.robust_normalise(H.wedged_noise((128,) * 3, seed=seed + i)) for i in range(n)]


class PairDataset(Dataset):
    """Item ``i``: ``make_training_pair`` on a random source, seeded by ``(seed, i)``."""

    def __init__(self, sources, n, seed, ssl_cfg):
        self.sources, self.n, self.seed, self.cfg = sources, n, seed, ssl_cfg

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from tomocpt.networks.ssl.views import make_training_pair

        rng = np.random.default_rng([self.seed, i])
        np.random.seed((self.seed * 1000003 + i) % 2 ** 32)
        src = self.sources[int(rng.integers(len(self.sources)))]
        return make_training_pair(source=src, apix_ref=APIX, ssl_cfg=self.cfg, rng=rng)


def train_ssl_v2(tmp, sources, seed, steps=STEPS):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    cfg = H.ssl_v2_main_config(tmp)
    cfg.train.optimizer.lr = 1e-3
    cfg.train.batch_size = BATCH
    with applied_main_config(cfg) as g:
        ssl = g.train.ssl
        train = PairDataset(sources, steps * BATCH, seed, ssl)
        val = PairDataset(sources, BATCH, seed + 99991, ssl)

        class _D(pl.LightningDataModule):
            def train_dataloader(self):
                return DataLoader(train, batch_size=BATCH, shuffle=False, num_workers=12)

            def val_dataloader(self):
                return DataLoader(val, batch_size=BATCH, shuffle=False, num_workers=0)

        torch.manual_seed(seed)
        model = SelfSupervisedModel(lr=1e-3, train_continue=None)
        trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs=1,
                             limit_val_batches=1, gradient_clip_val=1.0, gradient_clip_algorithm="norm",
                             logger=TensorBoardLogger(str(tmp / "tb")), enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(model, datamodule=_D())
    return model.cuda().eval()


def untrained_ssl_v2(tmp, seed):
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    with applied_main_config(H.ssl_v2_main_config(tmp)):
        torch.manual_seed(seed + 7)
        model = SelfSupervisedModel(lr=1e-3, train_continue=None)
    return model.cuda().eval()


def _single(model, cls):
    mods = [m for m in model.modules() if isinstance(m, cls)]
    assert len(mods) == 1, f"expected one {cls.__name__}, found {len(mods)}"
    return mods[0]


def _vol(x):
    t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))
    return t.float().reshape(1, 1, 64, 64, 64)


@torch.no_grad()
def predicted_ratios(model, sources, n_pairs, seed):
    """``(predicted, true)`` log2 ratios on ``n_pairs`` held-out ``make_training_pair`` items."""
    _, _, RelativeScaleHead = H.head_classes()
    head = _single(model, RelativeScaleHead)
    ds = PairDataset(sources, n_pairs, seed, H.resolved(model.hparams.config.train.ssl))
    pred, true = [], []
    for i in range(n_pairs):
        it = ds[i]
        xa, xb = _vol(it["input_a"]).cuda(), _vol(it["input_b"]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pa, pb = H.encoder_pooled(model.model, xa), H.encoder_pooled(model.model, xb)
        r = head(pa.float(), pb.float())
        pred.append(float(r.reshape(-1)[0]))
        true.append(H.to_scalar(it["target_log2"]))
    return np.array(pred), np.array(true)


# --------------------------------------------------------------------------------------
# shared trainings
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def phantom_model(tmp_path_factory):
    _need_cuda()
    tmp = tmp_path_factory.mktemp("g5_phantom")
    return train_ssl_v2(tmp, phantom_sources(10, 1000 * SEED), SEED)


@pytest.fixture(scope="module")
def noise_model(tmp_path_factory):
    _need_cuda()
    tmp = tmp_path_factory.mktemp("g5_noise")
    return train_ssl_v2(tmp, noise_sources(10, 1000 * SEED), SEED + 1)


def test_G5_C15_scale_learning(phantom_model):
    held_out = phantom_sources(3, 1000 * SEED + 500)
    pred, true = predicted_ratios(phantom_model, held_out, 180, SEED + 31)
    mae = float(np.mean(np.abs(pred - true)))
    print(f"G5_RESULT C15 seed={SEED} mae={mae:.4f} baseline(0)={np.mean(np.abs(true)):.4f}")
    assert mae < 0.15


def test_G5_C16_no_scale_shortcut_on_noise(noise_model):
    held_out = noise_sources(3, 1000 * SEED + 500)
    pred, true = predicted_ratios(noise_model, held_out, 200, SEED + 37)
    r = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 else 0.0
    print(f"G5_RESULT C16 seed={SEED} pearson={r:.4f}")
    assert abs(r) < 0.2


# --------------------------------------------------------------------------------------
# C17: dense embeddings
# --------------------------------------------------------------------------------------

def _embeddings_at(model, vol, centres_zyx):
    """``{name: [N, D]}`` L2-normalised embeddings at the centres (``decoder2`` features and,
    if the dense head accepts them, its output)."""
    _, DenseProjectionHead, _ = H.head_classes()
    x = torch.from_numpy(vol)[None, None].cuda()
    apix = torch.full((1, 1), APIX, device="cuda")
    size = torch.tensor(vol.shape, dtype=torch.float32)
    c = torch.as_tensor(centres_zyx, dtype=torch.float32)
    grid = ((c + 0.5) / size * 2 - 1).flip(-1).reshape(1, 1, 1, -1, 3).cuda()  # (x, y, z) for grid_sample
    out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats = {"decoder2": H.decoder2_features(model.model, x, apix)}
        heads = [m for m in model.modules() if isinstance(m, DenseProjectionHead)]
        if len(heads) == 1:
            try:
                feats["dense_head"] = heads[0](feats["decoder2"])
            except Exception:  # head with another input contract: report decoder2 only
                pass
    for name, f in feats.items():
        e = F.grid_sample(f.float(), grid, mode="bilinear", align_corners=False)
        out[name] = F.normalize(e.reshape(f.shape[1], -1).T, dim=1).cpu().numpy()
    return out


def _balanced_knn(emb_per_tomo, labels_per_tomo, k=5):
    """Leave-one-tomogram-out k-NN (cosine) balanced accuracy."""
    correct, total = {}, {}
    for t in range(len(emb_per_tomo)):
        train_e = np.concatenate([e for i, e in enumerate(emb_per_tomo) if i != t])
        train_l = np.concatenate([l for i, l in enumerate(labels_per_tomo) if i != t])
        sim = emb_per_tomo[t] @ train_e.T
        nn = np.argsort(-sim, axis=1)[:, :k]
        for row, lab in zip(nn, labels_per_tomo[t]):
            votes = np.bincount(train_l[row], minlength=train_l.max() + 1)
            correct[lab] = correct.get(lab, 0) + int(np.argmax(votes) == lab)
            total[lab] = total.get(lab, 0) + 1
    return float(np.mean([correct[c] / total[c] for c in total]))


def _knn_score(model, phantoms):
    embs, labels = {}, []
    for ph in phantoms:
        vol = H.robust_normalise(ph.volume)
        cents, labs = [], []
        for cls, xyz in ph.coords.items():
            for x, y, z in np.asarray(xyz).reshape(-1, 3):
                cents.append([z, y, x])
                labs.append(cls - 1)
        for name, e in _embeddings_at(model, vol, np.asarray(cents)).items():
            embs.setdefault(name, []).append(e)
        labels.append(np.asarray(labs))
    scores = {name: _balanced_knn(e, labels) for name, e in embs.items()}
    return max(scores.values()), scores


def test_G5_C17_dense_embeddings_separate_species(phantom_model, tmp_path):
    phantoms = [_phantom((96,) * 3, EVAL_SPECIES, (), 1000 * SEED + 700 + i) for i in range(6)]
    trained, trained_all = _knn_score(phantom_model, phantoms)
    random_init, random_all = _knn_score(untrained_ssl_v2(tmp_path, SEED), phantoms)
    print(f"G5_RESULT C17 seed={SEED} trained={trained_all} random={random_all}")
    assert trained >= 0.8
    assert trained - random_init >= 0.2
