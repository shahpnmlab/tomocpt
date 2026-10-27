"""G3-C12, G3-C13 (slow, CUDA): learning tests with tiny models (``feature_size=12``) on phantoms.

Training data are built **in memory** (not through chunking): ``utils.phantoms.make_phantom``
tomograms, robust-normalised like pipeline v1/v2 (``clip((x - median) / (p95 - p5), -3, 3)``),
random 64^3 crops with random axis flips, and pipeline-v2 batch dicts (CLAUDE.md 5.1) whose
targets are drawn with ``labels.helpers.render_targets``. Training goes through
``BasePickingModel.training_step`` with a Lightning trainer (bf16-mixed, gradient clip 1.0).
Evaluation: held-out 64^3 phantom chunks at 10 A/px, ``predict_step``, peaks per channel with
``predict.helpers.peak_local_max_torch`` (threshold 0.3, min distance d/2), matches within
d/4 voxels with ``tomocpt.metrics.match_points``.

``TOMOCPT_G3_SEED`` (default 0) selects the seed (used for the seed sweep in README).
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from gate3 import g3_helpers as H
from utils.phantoms import make_phantom
from utils.tiny import applied_main_config

SEED = int(os.environ.get("TOMOCPT_G3_SEED", "0"))
CHUNK = 64
THRESHOLD = 0.3

pytestmark = pytest.mark.slow


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def normalise(vol):
    vol = np.asarray(vol, dtype=np.float32)
    med = np.median(vol)
    p5, p95 = np.percentile(vol, [5, 95])
    return np.clip((vol - med) / (p95 - p5), -3, 3).astype(np.float32)


class Tomo:
    """A normalised volume with its particles as ``[N, 5]`` rows ``z, y, x, class_index, d_vox``."""

    def __init__(self, volume, coords, apix, table, available):
        self.volume = normalise(volume)
        self.apix = float(apix)
        idx = {number: i for i, (number, _, _) in enumerate(table)}
        rows = []
        for number, xyz in coords.items():
            d = dict((n, dd) for n, _, dd in table)[number] / apix
            for x, y, z in np.asarray(xyz).reshape(-1, 3):
                rows.append([z, y, x, idx[number], d])
        self.points = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
        self.d_vox = np.array([d / apix for _, _, d in table], dtype=np.float32)
        self.available = np.asarray(available, dtype=bool)


class CropDataset(Dataset):
    """Random 64^3 crops with random flips; item ``i`` is seeded by ``(seed, i)``."""

    def __init__(self, tomos, n, seed, weights=None, flips=True):
        self.tomos, self.n, self.seed, self.flips = tomos, n, seed, flips
        w = np.ones(len(tomos)) if weights is None else np.asarray(weights, dtype=float)
        self.p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        t = self.tomos[rng.choice(len(self.tomos), p=self.p)]
        o = [int(rng.integers(0, s - CHUNK + 1)) for s in t.volume.shape]
        vol = t.volume[o[0]:o[0] + CHUNK, o[1]:o[1] + CHUNK, o[2]:o[2] + CHUNK].copy()
        pts = t.points.copy()
        pts[:, :3] -= np.asarray(o, dtype=np.float64)
        r = pts[:, 4:5] / 2
        keep = np.all((pts[:, :3] + r > -1) & (pts[:, :3] - r < CHUNK), axis=1)
        pts = pts[keep]
        if self.flips:
            for a in range(3):
                if rng.random() < 0.5:
                    vol = np.flip(vol, axis=a)
                    pts[:, a] = CHUNK - 1 - pts[:, a]
        target = H.render(pts, len(t.d_vox))
        return {
            "input_data": torch.from_numpy(np.ascontiguousarray(vol))[None],
            "target_data": torch.from_numpy(target),
            "class_available": torch.from_numpy(t.available.copy()),
            "class_d_vox": torch.from_numpy(t.d_vox.copy()),
            "apix_work": torch.tensor([t.apix], dtype=torch.float32),
            "dataset_id": torch.tensor(0, dtype=torch.int64),
        }


def train_picker(tmp, classes, cond, train_tomos, steps, seed, lr=1e-3, batch_size=4, weights=None):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    from tomocpt.networks.pickingModel import BasePickingModel

    train = CropDataset(train_tomos, steps * batch_size, seed, weights=weights)
    val = CropDataset(train_tomos, 2 * batch_size, seed + 99991, weights=weights, flips=False)

    class _D(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=6,
                              persistent_workers=False)

        def val_dataloader(self):
            return DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)

    cfg = H.v2_config(tmp, classes, cond=cond)
    cfg.train.optimizer.lr = lr
    with applied_main_config(cfg):
        torch.manual_seed(seed)
        model = BasePickingModel(lr=lr, train_continue=None)
        trainer = pl.Trainer(accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs=1,
                             limit_val_batches=1, gradient_clip_val=1.0, gradient_clip_algorithm="norm",
                             logger=TensorBoardLogger(str(tmp / "tb")), enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(model, datamodule=_D())
    return model.eval()


@torch.no_grad()
def predict(model, vol, apix, cond):
    x = torch.from_numpy(vol)[None, None].cuda()
    a = H.apix_tensor(apix).cuda() if cond else None
    return model.cuda().predict_step(x, 0, apix=a)[0].float().cpu()


def peaks(pred_c, d_vox):
    from tomocpt.predict.helpers import peak_local_max_torch

    p = peak_local_max_torch(pred_c, min_distance=int(d_vox / 2), threshold_abs=THRESHOLD)
    return p.cpu().numpy().astype(np.float64).reshape(-1, 3)  # z, y, x


def score(pred_zyx, true_zyx, d_vox):
    from tomocpt.metrics import match_points

    tp, fp, fn = match_points(pred_zyx, true_zyx, d_vox / 4.0)
    return len(tp), len(fp), len(fn)


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def _true_zyx(coords, number):
    xyz = np.asarray(coords.get(number, np.zeros((0, 3)))).reshape(-1, 3)
    return xyz[:, ::-1].astype(np.float64)


def _report(name, payload):
    print(f"\nG3_RESULT {name} seed={SEED} " + json.dumps(payload, sort_keys=True))


@pytest.fixture
def scratch():
    d = tempfile.mkdtemp(prefix="g3_slow_")
    from pathlib import Path
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------------------
# G3-C12: scale disambiguation
# --------------------------------------------------------------------------------------

C12_CLASSES = ((1, "A", 150.0), (2, "B", 300.0))  # A: 15 vox at 10 A/px; B: 30 vox (15 at 20 A/px)
C12_SNR = 1.0
C12_STEPS = 2000
C12_SHAPE = (64, 128, 128)


def _c12_data(seed):
    table = C12_CLASSES
    train = []
    for k in range(6):  # 10 A/px: A and B, both annotated
        ph = make_phantom(shape=C12_SHAPE, apix=10.0, species=((2, 300.0, 4, False), (1, 150.0, 14, False)),
                          snr=C12_SNR, seed=seed * 1000 + k)
        train.append(Tomo(ph.volume, ph.coords, 10.0, table, [True, True]))
    for k in range(6):  # 20 A/px: B only (15 vox); A would be 7.5 vox < band_min -> unavailable
        ph = make_phantom(shape=C12_SHAPE, apix=20.0, species=((2, 300.0, 14, False),),
                          snr=C12_SNR, seed=seed * 1000 + 100 + k)
        train.append(Tomo(ph.volume, ph.coords, 20.0, table, [False, True]))
    held = [make_phantom(shape=(CHUNK,) * 3, apix=10.0, species=((2, 300.0, 1, False), (1, 150.0, 6, False)),
                         snr=C12_SNR, seed=seed * 1000 + 500 + k) for k in range(12)]
    return train, held


def _c12_eval(model, held, cond):
    counts = {"A": [0, 0, 0], "B": [0, 0, 0]}
    b_at_a = []
    for ph in held:
        vol = normalise(ph.volume)
        pred = predict(model, vol, 10.0, cond)
        for c, (number, name, d) in enumerate(C12_CLASSES):
            dv = d / 10.0
            s = score(peaks(pred[c], dv), _true_zyx(ph.coords, number), dv)
            counts[name] = [u + v for u, v in zip(counts[name], s)]
        for z, y, x in np.rint(_true_zyx(ph.coords, 1)).astype(int):
            b_at_a.append(float(pred[1, z, y, x]))
    return {"f1_A": f1(*counts["A"]), "f1_B": f1(*counts["B"]), "counts": counts,
            "mean_B_at_A": float(np.mean(b_at_a))}


def test_G3_C12_scale_disambiguation(scratch):
    """Class B (300 A) is trained at 10 A/px (30 vox) and at 20 A/px (15 vox, the voxel size of
    class A at 10 A/px). The conditioned model, on held-out 10 A/px chunks: F1 >= 0.8 for A and
    B, and mean B prediction at A centres < 0.2. The unconditioned model's number is reported.

    Calibration (scratch reference implementation, seeds 0-9, see README): conditioned F1_A
    min 0.923 (threshold 0.8), F1_B 1.0 in all seeds, mean_B_at_A 0.000-0.216 (median ~0.12;
    threshold 0.2): 9 of 10 seeds pass (seed 7: 0.216). Default seed 0: 0.130.
    """
    _need_cuda()
    train, held = _c12_data(SEED)
    weights = [1.0] * 6 + [1.0] * 6
    res = {}
    model = train_picker(scratch / "cond", C12_CLASSES, True, train, C12_STEPS, SEED, weights=weights)
    res["cond"] = _c12_eval(model, held, True)
    del model
    if os.environ.get("TOMOCPT_G3_SKIP_UNCOND") != "1":
        model = train_picker(scratch / "uncond", C12_CLASSES, False, train, C12_STEPS, SEED, weights=weights)
        res["uncond"] = _c12_eval(model, held, False)
        del model
    _report("C12", res)
    assert res["cond"]["f1_A"] >= 0.8, res
    assert res["cond"]["f1_B"] >= 0.8, res
    assert res["cond"]["mean_B_at_A"] < 0.2, res


# --------------------------------------------------------------------------------------
# G3-C13: same size, different shape
# --------------------------------------------------------------------------------------

#: class 2 (solid) and class 3 (hollow) have the same diameter
C13_SMALL, C13_PAIR, C13_LARGE = 100.0, 200.0, 300.0
C13_SNR = 2.0
C13_WEDGE = 60.0
C13_STEPS = 1800
C13_LR = 1e-3


def c13_classes():
    return ((1, "small", C13_SMALL), (2, "solid", C13_PAIR), (3, "hollow", C13_PAIR), (4, "large", C13_LARGE))


def _c13_data(seed):
    table = c13_classes()
    species = ((4, C13_LARGE, 3, False), (2, C13_PAIR, 5, False), (3, C13_PAIR, 5, True), (1, C13_SMALL, 8, False))
    train = [Tomo(ph.volume, ph.coords, 10.0, table, [True] * 4)
             for ph in (make_phantom(shape=(64, 128, 128), apix=10.0, species=species, snr=C13_SNR,
                                     wedge_deg=C13_WEDGE, seed=seed * 1000 + 200 + k) for k in range(8))]
    held_species = ((4, C13_LARGE, 1, False), (2, C13_PAIR, 2, False), (3, C13_PAIR, 2, True), (1, C13_SMALL, 3, False))
    held = [make_phantom(shape=(CHUNK,) * 3, apix=10.0, species=held_species, snr=C13_SNR, wedge_deg=C13_WEDGE,
                         seed=seed * 1000 + 700 + k) for k in range(12)]
    return train, held


def _c13_eval(model, held):
    table = c13_classes()
    counts = {name: [0, 0, 0] for _, name, _ in table}
    cross = {"solid_in_hollow": [0, 0], "hollow_in_solid": [0, 0]}
    for ph in held:
        pred = predict(model, normalise(ph.volume), 10.0, False)
        pk = {}
        for c, (number, name, d) in enumerate(table):
            pk[number] = peaks(pred[c], d / 10.0)
            s = score(pk[number], _true_zyx(ph.coords, number), d / 10.0)
            counts[name] = [u + v for u, v in zip(counts[name], s)]
        for key, true_n, chan_n in (("solid_in_hollow", 2, 3), ("hollow_in_solid", 3, 2)):
            t = _true_zyx(ph.coords, true_n)
            tp, _, _ = score(pk[chan_n], t, C13_PAIR / 10.0)
            cross[key][0] += tp
            cross[key][1] += len(t)
    return {"f1": {k: f1(*v) for k, v in counts.items()}, "counts": counts,
            "cross": {k: v[0] / max(1, v[1]) for k, v in cross.items()}, "cross_counts": cross}


def test_G3_C13_same_size_different_shape(scratch):
    """Four classes at 10 A/px, two of the same diameter (solid vs hollow spheres). Held-out
    F1 >= 0.8 for every class, and fewer than 15 % of the true particles of one same-size
    class are picked in the other's channel (a pick = a peak of the other channel within d/4).

    Calibration (scratch reference implementation): only seeds 0-2 were run (budget). Seeds 0 and
    2 pass (min F1 0.847 / 0.842, cross-pick 0.0); seed 1 fails (the solid channel fires on every
    hollow particle: solid F1 0.571, hollow_in_solid 1.0; still so after 3000 steps). The
    9-of-10 requirement is NOT demonstrated for this criterion; see README.
    """
    _need_cuda()
    train, held = _c13_data(SEED)
    model = train_picker(scratch / "c13", c13_classes(), False, train, C13_STEPS, SEED, lr=C13_LR)
    res = _c13_eval(model, held)
    _report("C13", res)
    for name, v in res["f1"].items():
        assert v >= 0.8, (name, res)
    for key, v in res["cross"].items():
        assert v < 0.15, (key, res)
