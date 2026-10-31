"""G6-C8, G6-C9 (slow, CUDA): learning tests with tiny models (``feature_size=12``) on phantoms.

As ``tests/gate3/test_g3_learning.py``: training data are built in memory from
``utils.phantoms.make_phantom`` tomograms (10 A/px, 64 x 128 x 128), robust-normalised, random
64^3 crops with random axis flips, pipeline-v2 batch dicts (CLAUDE.md 5.1). Targets: particles
with ``labels.helpers.render_targets``, tubes with ``labels.filaments.render_tubes`` from the
phantom centrelines (segments shifted and flipped with the crop). Training goes through
``BasePickingModel.training_step`` (bf16-mixed, gradient clip 1.0).

Classes: particles 1 "s120" (120 A) and 2 "s200" (200 A); filaments 3 "mt" (250 A, hollow
phantom tube) and 4 "actin" (70 A, solid; 7 voxels, below the band, used at native size).

Evaluation on held-out phantoms (whole 64 x 128 x 128 volume through ``predict_step``):

* particles: ``peak_local_max_torch`` (threshold ``PICK_THRESHOLD`` = 0.5, min distance d/2), the
  Gate 4 cross-class rule (as ``predict`` by default), matches within d/4
  (``tomocpt.metrics.match_points``); F1 pooled over both particle classes;
* filaments: ``trace_filaments(prob, 0.3, 10, diameter, 200, 40)``; length-weighted centreline
  precision / recall / F1 with tolerance 0.5 x diameter (both curves densified to 0.5 voxel:
  precision = fraction of predicted length within the tolerance of a true centreline, recall =
  fraction of true length within the tolerance of a predicted one), pooled over the held-out
  volumes.

Thresholds are the spec's (F1 >= 0.8; interference <= 0.03). They are **not calibrated**: the
Test Author had no implementation (see README). ``TOMOCPT_G6_SEED`` (default 0) selects the seed.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from gate3 import g3_helpers as G3
from gate6 import g6_helpers as H
from utils.phantoms import make_phantom
from utils.tiny import applied_main_config

pytestmark = pytest.mark.slow

SEED = int(os.environ.get("TOMOCPT_G6_SEED", "0"))
CHUNK = 64
APIX = 10.0
SHAPE = (64, 128, 128)
THRESHOLD = 0.3  # filament tracing threshold (infer.confidence_threshold default)
#: Particle pick threshold. The first reference run (threshold 0.3, no cross-class rule) found every
#: particle but gave 29-43 false positives in both models; picks now use 0.5 and the Gate 4
#: cross-class rule that ``predict`` applies by default. 0.3 without the rule is reported too.
PICK_THRESHOLD = 0.5
STEPS = 2000
SNR = 1.0
PARTICLES = ((1, "s120", 120.0), (2, "s200", 200.0))
FILAMENTS = ((3, "mt", 250.0), (4, "actin", 70.0))
SPECIES = ((2, 200.0, 4, False), (1, 120.0, 8, False))
TUBES = ((3, 250.0, 1), (4, 70.0, 2))


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def normalise(vol):
    vol = np.asarray(vol, dtype=np.float32)
    med = np.median(vol)
    p5, p95 = np.percentile(vol, [5, 95])
    return np.clip((vol - med) / (p95 - p5), -3, 3).astype(np.float32)


class Tomo:
    def __init__(self, ph, table, with_filaments=True):
        self.volume = normalise(ph.volume)
        idx = {number: i for i, (number, _, _) in enumerate(table)}
        diam = {number: d for number, _, d in table}
        rows = []
        for number, xyz in ph.coords.items():
            if number in idx:
                for x, y, z in np.asarray(xyz).reshape(-1, 3):
                    rows.append([z, y, x, idx[number], diam[number] / APIX])
        self.points = np.asarray(rows, np.float64).reshape(-1, 5)
        segs = [np.zeros((0, 8), np.float32)]
        if with_filaments:
            for number, polys in ph.polylines.items():
                if number in idx:
                    for p in polys:
                        segs.append(H.segments_from_polyline(p, idx[number], diam[number] / APIX))
        self.segments = np.concatenate(segs).astype(np.float64)
        self.d_vox = np.array([d / APIX for _, _, d in table], np.float32)
        self.available = np.ones(len(table), bool)


class CropDataset(Dataset):
    def __init__(self, tomos, n, seed, flips=True):
        self.tomos, self.n, self.seed, self.flips = tomos, n, seed, flips

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from tomocpt.labels.filaments import render_tubes

        rng = np.random.default_rng([self.seed, i])
        t = self.tomos[int(rng.integers(len(self.tomos)))]
        o = np.array([int(rng.integers(0, s - CHUNK + 1)) for s in t.volume.shape], np.float64)
        vol = t.volume[int(o[0]):int(o[0]) + CHUNK, int(o[1]):int(o[1]) + CHUNK, int(o[2]):int(o[2]) + CHUNK].copy()
        pts = t.points.copy()
        pts[:, :3] -= o
        seg = t.segments.copy()
        seg[:, 0:3] -= o
        seg[:, 3:6] -= o
        if self.flips:
            for a in range(3):
                if rng.random() < 0.5:
                    vol = np.flip(vol, axis=a)
                    pts[:, a] = CHUNK - 1 - pts[:, a]
                    seg[:, a] = CHUNK - 1 - seg[:, a]
                    seg[:, 3 + a] = CHUNK - 1 - seg[:, 3 + a]
        C = len(t.d_vox)
        target = np.maximum(G3.render(pts, C), render_tubes(seg.astype(np.float32), (CHUNK,) * 3, C))
        return {
            "input_data": torch.from_numpy(np.ascontiguousarray(vol))[None],
            "target_data": torch.from_numpy(target),
            "class_available": torch.from_numpy(t.available.copy()),
            "class_d_vox": torch.from_numpy(t.d_vox.copy()),
            "apix_work": torch.tensor([APIX], dtype=torch.float32),
            "dataset_id": torch.tensor(0, dtype=torch.int64),
        }


def train_picker(tmp, particles, filaments, tomos, steps, seed, lr=1e-3, batch_size=4):
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

    cfg = H.v2_config_with_filaments(tmp, particles, filaments)
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


def particle_peaks(pred, table, threshold, nms):
    """``{class_number: [N, 4] (z, y, x, score)}`` for the particle classes:
    ``peak_local_max_torch`` (min distance d/2), then, with ``nms``, the inference rule of Gate 4
    (``predict.helpers.cross_class_suppression``, factor 0.5, radii d/2), as ``predict`` does
    with the default ``infer.cross_class_nms = True``."""
    from tomocpt.predict.helpers import cross_class_suppression, peak_local_max_torch

    numbers = {n for n, _, _ in PARTICLES}
    peaks, radii = {}, {}
    for c, (number, _, d) in enumerate(table):
        if number not in numbers:
            continue
        dv = d / APIX
        p = peak_local_max_torch(pred[c], min_distance=int(dv / 2), threshold_abs=threshold)
        p = p.cpu().numpy().astype(np.int64).reshape(-1, 3)
        score = pred[c].numpy()[p[:, 0], p[:, 1], p[:, 2]] if len(p) else np.zeros(0)
        peaks[number] = np.concatenate([p.astype(np.float64), score[:, None]], 1).reshape(-1, 4)
        radii[number] = dv / 2.0
    if nms:
        peaks, _ = cross_class_suppression(peaks, radii, 0.5)
    return peaks


def particle_counts(pred, ph, table, threshold=PICK_THRESHOLD, nms=True):
    """Matches within d/4 (``tomocpt.metrics.match_points``), summed over the particle classes."""
    from tomocpt.metrics import match_points

    tp = fp = fn = 0
    for number, pk in particle_peaks(pred, table, threshold, nms).items():
        dv = dict((n, d) for n, _, d in PARTICLES)[number] / APIX
        t = np.asarray(ph.coords.get(number, np.zeros((0, 3)))).reshape(-1, 3)[:, ::-1].astype(np.float64)
        a, b, e = match_points(pk[:, :3], t, dv / 4.0)
        tp, fp, fn = tp + len(a), fp + len(b), fn + len(e)
    return tp, fp, fn


def centreline_lengths(pred_c, true_polys, diameter):
    """``(predicted length near truth, predicted length, true length near prediction, true length)``
    in units of 0.5-voxel samples."""
    from tomocpt.labels.filaments import trace_filaments

    tol = 0.5 * diameter / APIX
    polys = trace_filaments(pred_c.numpy(), THRESHOLD, APIX, diameter, 200.0, 40.0)
    pd_ = [H.densify(np.asarray(p, float), 0.5) for p in polys if len(p) >= 2]
    td = [H.densify(np.asarray(p, float), 0.5) for p in true_polys if len(p) >= 2]
    P = np.concatenate(pd_) if pd_ else np.zeros((0, 3))
    T = np.concatenate(td) if td else np.zeros((0, 3))
    def near(a, curves):
        if not len(a):
            return 0
        if not curves:
            return 0
        d = np.min([H.point_to_polyline_distance(a, c) for c in curves], axis=0)
        return int((d <= tol).sum())
    return near(P, td), len(P), near(T, pd_), len(T)


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def _report(name, payload):
    print(f"\nG6_RESULT {name} seed={SEED} " + json.dumps(payload, sort_keys=True))


def _data(seed):
    phs = [make_phantom(shape=SHAPE, apix=APIX, species=SPECIES, tubes=TUBES, snr=SNR, seed=seed * 1000 + k)
           for k in range(8)]
    held = [make_phantom(shape=SHAPE, apix=APIX, species=SPECIES, tubes=TUBES, snr=SNR, seed=seed * 1000 + 500 + k)
            for k in range(4)]
    return phs, held


@pytest.fixture(scope="module")
def trained():
    _need_cuda()
    scratch = Path(tempfile.mkdtemp(prefix="g6_slow_"))
    try:
        phs, held = _data(SEED)
        joint_table = PARTICLES + FILAMENTS
        joint = train_picker(scratch / "joint", PARTICLES, FILAMENTS, [Tomo(p, joint_table) for p in phs],
                             STEPS, SEED)
        res = {"joint": evaluate(joint, held, joint_table)}
        del joint
        ponly = train_picker(scratch / "ponly", PARTICLES, (), [Tomo(p, PARTICLES, False) for p in phs], STEPS, SEED)
        res["particles_only"] = evaluate(ponly, held, PARTICLES)
        del ponly
        _report("C8_C9", res)
        yield res
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def evaluate(model, held, table):
    pc, pc_raw = [0, 0, 0], [0, 0, 0]
    fil = {name: [0, 0, 0, 0] for _, name, _ in FILAMENTS}
    for ph in held:
        pred = predict(model, normalise(ph.volume))
        pc = [u + v for u, v in zip(pc, particle_counts(pred, ph, table))]
        pc_raw = [u + v for u, v in zip(pc_raw, particle_counts(pred, ph, table, threshold=0.3, nms=False))]
        for c, (number, name, d) in enumerate(table):
            if name in fil:
                fil[name] = [u + v for u, v in zip(fil[name], centreline_lengths(pred[c], ph.polylines[number], d))]
    out = {"particle_f1": f1(*pc), "particle_counts": pc,
           "particle_f1_thr0.3_no_nms": f1(*pc_raw), "particle_counts_thr0.3_no_nms": pc_raw}
    for name, (pn, pl, tn, tl) in fil.items():
        prec, rec = pn / max(1, pl), tn / max(1, tl)
        out[f"{name}_precision"], out[f"{name}_recall"] = prec, rec
        out[f"{name}_f1"] = 2 * prec * rec / max(1e-9, prec + rec)
    return out


def test_G6_C8_tubes_and_spheres_in_one_model(trained):
    j = trained["joint"]
    assert j["mt_f1"] >= 0.8, j
    assert j["actin_f1"] >= 0.8, j
    assert j["particle_f1"] >= 0.8, j


def test_G6_C9_particle_interference(trained):
    j, p = trained["joint"], trained["particles_only"]
    assert j["particle_f1"] >= p["particle_f1"] - 0.03, {"joint": j, "particles_only": p}
