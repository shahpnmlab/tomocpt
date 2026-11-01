"""G7-C9 and G7-C10 (slow, CUDA): learning tests with tiny models (``feature_size=12``) on a
look-alike phantom.

Phantom (10 A/px, 64 x 128 x 128, ``utils.phantoms.make_phantom`` without wedge and noise, then
:func:`g7_helpers.add_sheets`, then the 60 deg missing wedge and white noise at ``SNR``):

* particles 1 "solid" and 2 "hollow": spheres of the **same** diameter (160 A) and dark contrast,
  solid vs hollow shells (wall 0.2 R, at least 1.5 voxels), ``N_PER_SPECIES`` of each;
* filament 3 "actin": solid 70 A tubes (7 voxels, below the band, used at native size);
* ``N_SHEETS`` unlabelled flat dark slabs (membrane-like planes, ``SHEET_THICKNESS`` voxels) that
  avoid the labelled objects; their masks (slab + 1 voxel) are the ``negative_mask`` batch key.

Training (as ``tests/gate6/test_g6_learning.py``): random 64^3 crops with random axis flips,
pipeline-v2 batch dicts (CLAUDE.md 5.1, plus ``negative_mask``), ``BasePickingModel`` (bf16-mixed,
gradient clip 1.0), **softmax class head**, k = 0 (handoff ruling: softmax is the multi-class
head). Arms, each with seeds 0 and 1 (same data and initialisation per seed):

* ``off``: Gate 6 loss (both options off);
* ``contrastive``: ``contrastive_weight = CONTRASTIVE_WEIGHT``, other fields at their defaults;
* ``focal``: ``value_loss = "focal"`` (contrastive off): since 1 Nov 2026 (human decision) the
  soft focal penalty with the Huber weights on the particle channels, Huber on the actin channel.

Evaluation on held-out phantoms (whole volume through ``predict_step``), v2 picking defaults
(peak spacing 0.5 x diameter, no cross-class suppression): particles with
``peak_local_max_torch`` (``min_distance = d/2``) at threshold 0.5, matched within d/4
(``tomocpt.metrics.match_points``); the actin channel traced with ``trace_filaments(prob, 0.3,
apix, 70, 200, 40)``.

* **G7-C9** (spec): mean F1 of the look-alike pair (mean over the two classes and the two seeds)
  improves by >= 0.05 with contrastive on; tube false picks on sheets (traced actin length, in
  0.5-voxel samples, within 2 voxels of a sheet and farther than 0.5 x d from every true actin
  centreline; summed over seeds) fall by >= 30 %.
* **G7-C10** (orchestrator ruling, ``HANDOFF_gate7.md`` 2.2, peak shape): with the focal value
  term, particle F1 at threshold 0.3 without cross-class suppression (pooled over both particle
  classes, mean over seeds) does not drop by more than ``PEAK_MARGIN`` vs the Huber term.
  Cross-entropy terms that include the background flattened the peaks (split peaks at 0.3); this
  check catches that failure even when F1 at 0.5 looks fine.

Calibration of the ``off`` arm (TA scratch run, see README): the look-alike pair is not solved
(mean F1 below 0.9, room for +0.05) and tubes are picked on sheets (``MIN_SHEET_FP``). The
``contrastive`` and ``focal`` arms could not be calibrated before the implementation.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from gate3 import g3_helpers as G3
from gate6 import g6_helpers as H6
from gate7 import g7_helpers as H
from utils.phantoms import make_phantom
from utils.tiny import applied_main_config
from utils.wedge import apply_missing_wedge

pytestmark = pytest.mark.slow

SEEDS = (0, 1)
CHUNK = 64
APIX = 10.0
SHAPE = (64, 128, 128)
SNR = float(os.environ.get("TOMOCPT_G7_SNR", "0.25"))
STEPS = int(os.environ.get("TOMOCPT_G7_STEPS", "400"))
N_TRAIN, N_HELD = 8, 4
PARTICLES = ((1, "solid", 160.0), (2, "hollow", 160.0))
FILAMENTS = ((3, "actin", 70.0),)
TABLE = PARTICLES + FILAMENTS
N_PER_SPECIES = 5
SPECIES = ((1, 160.0, N_PER_SPECIES, False), (2, 160.0, N_PER_SPECIES, True))
TUBES = ((3, 70.0, 2),)
#: Wall of the hollow spheres as a fraction of the radius (thick wall: 78 % of the solid mass).
HOLLOW_WALL = float(os.environ.get("TOMOCPT_G7_WALL", "0.4"))
N_SHEETS = 2
SHEET_THICKNESS = 5.0
PICK_THRESHOLD = 0.5
TRACE_THRESHOLD = 0.3
CONTRASTIVE_WEIGHT = 0.1
#: Spec thresholds.
MIN_LOOKALIKE_GAIN = 0.05
MAX_SHEET_FP_RATIO = 0.7
#: Margin for the peak-shape check (seed-to-seed spread of pooled F1 at 0.3 in the calibration
#: runs of the ``off`` arm is about 0.02; see README).
PEAK_MARGIN = 0.03
#: The ``off`` arm must pick tubes on sheets (otherwise the C9 sheet criterion is void).
MIN_SHEET_FP = 40


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def normalise(vol):
    vol = np.asarray(vol, dtype=np.float32)
    med = np.median(vol)
    p5, p95 = np.percentile(vol, [5, 95])
    return np.clip((vol - med) / (p95 - p5), -3, 3).astype(np.float32)


class Lookalike:
    """One phantom tomogram: normalised volume, points, segments, sheet mask, truths."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed + 7919)
        ph = make_phantom(shape=SHAPE, apix=APIX, species=SPECIES, tubes=TUBES, wedge_deg=None, snr=None,
                          seed=seed, hollow_wall_fraction=HOLLOW_WALL)
        zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in SHAPE], indexing="ij")
        exclude = np.zeros(SHAPE, bool)
        for number, xyz in ph.coords.items():
            r = dict((n, d) for n, _, d in PARTICLES)[number] / (2 * APIX)
            for x, y, z in np.asarray(xyz).reshape(-1, 3):
                exclude |= (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= (r + 3) ** 2
        for m in ph.masks.values():
            exclude |= ndimage.binary_dilation(m, iterations=3)
        clean, self.sheet_mask = H.add_sheets(ph.volume, N_SHEETS, SHEET_THICKNESS, rng, exclude)
        signal = apply_missing_wedge(clean, max_angle_deg=60.0, tilt_axis="y")
        noise = rng.normal(0.0, np.sqrt(signal.var() / SNR), SHAPE).astype(np.float32)
        self.volume = normalise(signal + noise)
        self.coords = ph.coords
        self.polylines = ph.polylines
        idx = {n: i for i, (n, _, _) in enumerate(TABLE)}
        diam = {n: d for n, _, d in TABLE}
        rows = [[z, y, x, idx[n], diam[n] / APIX] for n, xyz in ph.coords.items()
                for x, y, z in np.asarray(xyz).reshape(-1, 3)]
        self.points = np.asarray(rows, np.float64).reshape(-1, 5)
        segs = [np.zeros((0, 8), np.float32)]
        for n, polys in ph.polylines.items():
            for p in polys:
                segs.append(H6.segments_from_polyline(p, idx[n], diam[n] / APIX))
        self.segments = np.concatenate(segs).astype(np.float64)
        self.d_vox = np.array([d / APIX for _, _, d in TABLE], np.float32)


class CropDataset(Dataset):
    def __init__(self, tomos, n, seed, flips=True):
        self.tomos, self.n, self.seed, self.flips = tomos, n, seed, flips

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from tomocpt.labels.filaments import render_tubes

        rng = np.random.default_rng([self.seed, i])
        t = self.tomos[int(rng.integers(len(self.tomos)))]
        o = np.array([int(rng.integers(0, s - CHUNK + 1)) for s in SHAPE], np.float64)
        sl = tuple(slice(int(a), int(a) + CHUNK) for a in o)
        vol, msk = t.volume[sl].copy(), t.sheet_mask[sl].copy()
        pts = t.points.copy()
        pts[:, :3] -= o
        seg = t.segments.copy()
        seg[:, 0:3] -= o
        seg[:, 3:6] -= o
        if self.flips:
            for a in range(3):
                if rng.random() < 0.5:
                    vol, msk = np.flip(vol, axis=a), np.flip(msk, axis=a)
                    pts[:, a] = CHUNK - 1 - pts[:, a]
                    seg[:, a] = CHUNK - 1 - seg[:, a]
                    seg[:, 3 + a] = CHUNK - 1 - seg[:, 3 + a]
        C = len(TABLE)
        target = np.maximum(G3.render(pts, C), render_tubes(seg.astype(np.float32), (CHUNK,) * 3, C))
        return {
            "input_data": torch.from_numpy(np.ascontiguousarray(vol))[None],
            "target_data": torch.from_numpy(target),
            "class_available": torch.ones(C, dtype=torch.bool),
            "class_d_vox": torch.from_numpy(t.d_vox.copy()),
            "apix_work": torch.tensor([APIX], dtype=torch.float32),
            "dataset_id": torch.tensor(0, dtype=torch.int64),
            "negative_mask": torch.from_numpy(np.ascontiguousarray(msk))[None],
        }


def train_picker(tmp, tomos, arm, seed, steps=STEPS, lr=1e-3, batch_size=4):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger

    from tomocpt.networks.pickingModel import BasePickingModel

    train = CropDataset(tomos, steps * batch_size, seed)
    val = CropDataset(tomos, 2 * batch_size, seed + 99991, flips=False)

    class _D(pl.LightningDataModule):
        def train_dataloader(self):
            return DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=6)

        def val_dataloader(self):
            return DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)

    disc = {"off": {}, "contrastive": {"contrastive_weight": CONTRASTIVE_WEIGHT},
            "focal": {"value_loss": "focal"}}[arm]
    cfg = H6.v2_config_with_filaments(tmp, PARTICLES, FILAMENTS, class_head="softmax")
    for k, v in disc.items():
        setattr(cfg.train.discriminative, k, v)
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
def predict(model, vol):
    x = torch.from_numpy(vol)[None, None].cuda()
    return model.cuda().predict_step(x, 0)[0].float().cpu()


def particle_counts(pred, tomo, threshold):
    """Per particle class ``(tp, fp, fn)``: peaks with min distance d/2 (v2 default spacing), no
    cross-class suppression (v2 default), matched within d/4."""
    from tomocpt.metrics import match_points
    from tomocpt.predict.helpers import peak_local_max_torch

    out = {}
    for c, (number, _, d) in enumerate(PARTICLES):
        dv = d / APIX
        p = peak_local_max_torch(pred[c], min_distance=int(dv / 2), threshold_abs=threshold)
        p = p.cpu().numpy().astype(np.float64).reshape(-1, 3)
        t = np.asarray(tomo.coords.get(number, np.zeros((0, 3)))).reshape(-1, 3)[:, ::-1].astype(np.float64)
        a, b, e = match_points(p, t, dv / 4.0)
        out[number] = (len(a), len(b), len(e))
    return out


def actin_lengths(pred_c, tomo):
    """``(near-truth, predicted, true-covered, true, on-sheet false)`` samples (0.5 voxel)."""
    from tomocpt.labels.filaments import trace_filaments

    number, _, d = FILAMENTS[0]
    tol = 0.5 * d / APIX
    polys = trace_filaments(pred_c.numpy(), TRACE_THRESHOLD, APIX, d, 200.0, 40.0)
    Pl = [H6.densify(np.asarray(p, float), 0.5) for p in polys if len(p) >= 2]
    T = [H6.densify(np.asarray(p, float), 0.5) for p in tomo.polylines.get(number, []) if len(p) >= 2]
    P = np.concatenate(Pl) if Pl else np.zeros((0, 3))
    Tc = np.concatenate(T) if T else np.zeros((0, 3))
    dP = np.min([H6.point_to_polyline_distance(P, c) for c in T], axis=0) if (len(P) and T) \
        else np.full(len(P), np.inf)
    dT = np.min([H6.point_to_polyline_distance(Tc, c) for c in Pl], axis=0) if (len(Tc) and Pl) \
        else np.full(len(Tc), np.inf)
    near_sheet = ndimage.binary_dilation(tomo.sheet_mask, iterations=2)
    on_sheet = 0
    for (x, y, z), dist in zip(P, dP):
        zi, yi, xi = (int(np.clip(round(v), 0, s - 1)) for v, s in zip((z, y, x), SHAPE))
        if near_sheet[zi, yi, xi] and dist > tol:
            on_sheet += 1
    return int((dP <= tol).sum()), len(P), int((dT <= tol).sum()), len(Tc), on_sheet


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def evaluate(model, held):
    per = {n: [0, 0, 0] for n, _, _ in PARTICLES}
    raw = [0, 0, 0]
    act = [0, 0, 0, 0, 0]
    for t in held:
        pred = predict(model, t.volume)
        for n, cnt in particle_counts(pred, t, PICK_THRESHOLD).items():
            per[n] = [u + v for u, v in zip(per[n], cnt)]
        for cnt in particle_counts(pred, t, 0.3).values():
            raw = [u + v for u, v in zip(raw, cnt)]
        act = [u + v for u, v in zip(act, actin_lengths(pred[len(PARTICLES)], t))]
    out = {f"f1_{name}": f1(*per[n]) for n, name, _ in PARTICLES}
    out["counts"] = {name: per[n] for n, name, _ in PARTICLES}
    out["lookalike_f1"] = float(np.mean([out[f"f1_{name}"] for _, name, _ in PARTICLES]))
    out["particle_f1_thr0.3_no_nms"] = f1(*raw)
    prec, rec = act[0] / max(1, act[1]), act[2] / max(1, act[3])
    out["actin_f1"] = 2 * prec * rec / max(1e-9, prec + rec)
    out["actin_precision"], out["actin_recall"] = prec, rec
    out["actin_on_sheet_fp"] = act[4]
    return out


def run_arms(arms, seeds=SEEDS, steps=STEPS):
    """``{arm: {seed: metrics}}``."""
    res = {a: {} for a in arms}
    scratch = Path(tempfile.mkdtemp(prefix="g7_slow_"))
    try:
        for seed in seeds:
            train = [Lookalike(seed * 1000 + k) for k in range(N_TRAIN)]
            held = [Lookalike(seed * 1000 + 500 + k) for k in range(N_HELD)]
            for arm in arms:
                model = train_picker(scratch / f"{arm}_{seed}", train, arm, seed, steps=steps)
                res[arm][seed] = evaluate(model, held)
                del model
                torch.cuda.empty_cache()
                print(f"\nG7_RESULT arm={arm} seed={seed} snr={SNR} wall={HOLLOW_WALL} steps={steps} "
                      + json.dumps(res[arm][seed], sort_keys=True), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return res


@pytest.fixture(scope="module")
def trained():
    _need_cuda()
    return run_arms(("off", "contrastive", "focal"))


def _mean(res, arm, key):
    return float(np.mean([res[arm][s][key] for s in SEEDS]))


def test_G7_C9_contrastive_separates_lookalikes(trained):
    off, on = _mean(trained, "off", "lookalike_f1"), _mean(trained, "contrastive", "lookalike_f1")
    assert on >= off + MIN_LOOKALIKE_GAIN, {"off": trained["off"], "contrastive": trained["contrastive"]}


def test_G7_C9_contrastive_reduces_tube_picks_on_sheets(trained):
    off = sum(trained["off"][s]["actin_on_sheet_fp"] for s in SEEDS)
    on = sum(trained["contrastive"][s]["actin_on_sheet_fp"] for s in SEEDS)
    assert off >= MIN_SHEET_FP, f"calibration: the off arm picks only {off} samples on sheets"
    assert on <= MAX_SHEET_FP_RATIO * off, {"off": off, "contrastive": on}


def test_G7_C10_focal_keeps_peak_shape(trained):
    huber = _mean(trained, "off", "particle_f1_thr0.3_no_nms")
    focal = _mean(trained, "focal", "particle_f1_thr0.3_no_nms")
    assert focal >= huber - PEAK_MARGIN, {"off": trained["off"], "focal": trained["focal"]}


def test_G7_C10_focal_picks_particles_as_well_as_huber(trained):
    """Reconciled focal term (human decision, 1 Nov 2026): look-alike F1 at the picking threshold
    (0.5, v2 peak spacing) is not lower than the Huber term's by more than ``PEAK_MARGIN``. Both
    terms have the minimiser ``p = y``, so one threshold is comparable across arms."""
    huber = _mean(trained, "off", "lookalike_f1")
    focal = _mean(trained, "focal", "lookalike_f1")
    assert focal >= huber - PEAK_MARGIN, {"off": trained["off"], "focal": trained["focal"]}
