"""Shared machinery of the Gate 5 SSL benchmarks (T5.7): SSL pretraining through the CLI,
tomogram splits, dense embeddings at particle centres and the k-NN probe.

Pretraining goes through the real CLI (``python -m tomocpt.main [prepare_data] train``, one
subprocess per command), like ``benchmarks/common.py``:

* SSL v2 (``train.ssl.version = 2``): no labels and no particle size; ``training_data_dir`` is a
  directory of tomogram symlinks.
* SSL v1 (legacy): the legacy chunking needs a particle size, so ``prepare_data`` runs with one
  species (``--ssl_v1_class``); its coordinates only set the fixed-particle rescale, the SSL
  losses use no label.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmarks.common import (  # noqa: E402
    ClassInfo,
    TrainSet,
    TrainSettings,
    _find_checkpoint,
    _log,
    _run_cli,
    _train_section,
    _write_config,
    isolated_main_config,
)

VIEW = 64


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def read_star(star: Path):
    """``(particles DataFrame, stems Series)`` of a RELION STAR file."""
    import starfile

    data = starfile.read(star)
    particles = data["particles"] if isinstance(data, dict) else data
    name_col = "rlnMicrographName" if "rlnMicrographName" in particles else "rlnTomoName"
    stems = particles[name_col].astype(str).map(lambda s: Path(s).stem)
    return data, particles, stems


def coords_of(star: Path, stem: str) -> Dict[int, np.ndarray]:
    """``{class_number: (N, 3) xyz}`` (tomogram voxels) of one tomogram."""
    _, particles, stems = read_star(star)
    rows = particles[stems == stem]
    return {int(c): g[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(np.float64)
            for c, g in rows.groupby("rlnClassNumber")}


def subset_trainset(tomo_dir: Path, star: Path, stems: Sequence[str], work: Path) -> TrainSet:
    """Symlinks of ``stems`` and their STAR rows under ``work``."""
    import starfile

    data, particles, all_stems = read_star(star)
    tdir = work / "tomograms"
    tdir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = Path(tomo_dir) / f"{stem}.mrc"
        link = tdir / src.name
        if not link.exists():
            os.symlink(src.resolve(), link)
    out = dict(data) if isinstance(data, dict) else {"particles": data}
    out["particles"] = particles[all_stems.isin(set(stems))]
    sub_star = work / "particles.star"
    starfile.write(out, sub_star, overwrite=True)
    return TrainSet(tdir, sub_star)


def tomogram_stems(tomo_dir: Path) -> List[str]:
    return sorted(p.stem for p in Path(tomo_dir).glob("*.mrc"))


# --------------------------------------------------------------------------------------
# SSL pretraining (CLI)
# --------------------------------------------------------------------------------------

def pretrain_ssl(work: Path, data: TrainSet, version: int, settings: TrainSettings,
                 v1_class: Optional[ClassInfo] = None) -> Path:
    """Pretrain SSL ``version`` (1 or 2) on the tomograms of ``data``; returns the checkpoint."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    train = _train_section(work, settings, 1)
    train["mode"] = "selfSupervised"
    train["ssl"] = {"version": int(version)}
    cfg = {"train": train, "infer": {"predictions_dir": str(work / "predictions")}}
    if version == 2:
        # No labels, no particle size: the tomograms themselves are the training data.
        train["training_data_dir"] = str(Path(data.tomo_dir))
        cfg["shared"] = {"training_data_dir": str(Path(data.tomo_dir)), "chunks_dir": str(work / "chunks")}
        config = _write_config(work / "config.yaml", cfg)
    else:
        if v1_class is None:
            raise ValueError("SSL v1 pretraining needs a species for the fixed-particle rescale")
        cfg["shared"] = {"training_data_dir": str(work / "training_data"), "chunks_dir": str(work / "chunks")}
        cfg["prepData"] = {
            "raw_data_dir": str(data.tomo_dir),
            "training_data_dir": str(work / "training_data"),
            "coordinate_files": str(data.star_path),
            "coordinate_file_type": "star",
            "particle_length_ang": repr(float(v1_class.diameter_ang)),
            "class_id": str(v1_class.number),
        }
        config = _write_config(work / "config.yaml", cfg)
        _run_cli("prepare_data", config, work / "prepare_data.log")
    _log(f"SSL v{version}: pretraining in {work}")
    _run_cli("train", config, work / "train.log")
    return _find_checkpoint(work / "model")


# --------------------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------------------

class Embedder:
    """Dense features of one network at particle centres.

    ``kind``: ``"ssl_v2"`` (FiLM-conditioned ``decoder2`` output, plus the dense head when the
    model has one), ``"ssl_v1"`` (unconditioned ``decoder2`` output of the legacy network) or
    ``"random_v2"`` (an untrained SSL v2 model built with the same settings as ``ckpt``).
    ``read_apix(apix_native)`` gives the voxel size at which the tomogram is read."""

    def __init__(self, kind: str, ckpt: Path, device: torch.device, seed: int = 0,
                 v1_diameter_ang: Optional[float] = None):
        from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

        self.kind = kind
        self.device = device
        with isolated_main_config():
            if kind == "random_v2":
                raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)["hyper_parameters"]
                torch.manual_seed(seed)
                model = SelfSupervisedModel(lr=1e-3, config=raw["config"], constants_dict=raw.get("constants_dict"))
            else:
                model = SelfSupervisedModel.load_from_checkpoint(str(ckpt), map_location="cpu")
        self.model = model.to(device).eval()
        cfg = model.hparams.config
        if kind == "ssl_v1":
            if v1_diameter_ang is None:
                raise ValueError("ssl_v1 embeddings need the species diameter used for its rescale")
            self._apix = lambda a: float(v1_diameter_ang) / float(cfg.prepData.desired_particle_pixel_size)
        else:
            ssl = cfg.train.ssl
            self._apix = lambda a: float(min(max(float(a), ssl.apix_ref_min), ssl.apix_ref_max))

    def read_apix(self, apix_native: float) -> float:
        return self._apix(apix_native)

    @torch.no_grad()
    def features(self, x: torch.Tensor, apix: float) -> Dict[str, torch.Tensor]:
        net = self.model.model
        hs = net.swinViT(x, net.normalize)
        enc1 = net.encoder2(hs[0])
        enc2 = net.encoder3(hs[1])
        enc3 = net.encoder4(hs[2])
        cond = getattr(net, "cond", None)
        if cond is not None:
            e = cond.scale_embed(torch.full((x.shape[0], 1), float(apix), device=x.device))
            film = cond.film
            f = lambda i, v: film[i](v, e)  # noqa: E731
        else:
            f = lambda i, v: v  # noqa: E731
        dec4 = f(0, net.encoder10(hs[4]))
        dec3 = f(1, net.decoder5(dec4, hs[3]))
        dec2 = f(2, net.decoder4(dec3, enc3))
        dec1 = f(3, net.decoder3(dec2, enc2))
        dec0 = f(4, net.decoder2(dec1, enc1))
        out = {"decoder2": dec0}
        head = getattr(self.model, "dense_head", None)
        if head is not None:
            out["dense_head"] = head(dec0)
        return out


def load_tomogram_at(path: Path, apix_read: float) -> Tuple[np.ndarray, float]:
    """Tomogram resampled (Fourier) to ``apix_read`` and robust-normalised; ``(vol, apix_native)``."""
    from tomocpt.dataManager.preprocessing import load_mrc, robust_normalization
    from tomocpt.dataManager.scaling import read_apix, resample_volume, working_shape

    apix_native = read_apix(path)
    vol = load_mrc(str(path), normalize=False)
    shape = working_shape(vol.shape, apix_native, apix_read)
    if tuple(shape) != tuple(vol.shape):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        vol = resample_volume(torch.from_numpy(vol).to(dev), shape, "fourier").cpu().numpy()
    return robust_normalization(np.asarray(vol, np.float32)).astype(np.float32), apix_native


def embeddings_at(emb: Embedder, path: Path, coords_xyz: Dict[int, np.ndarray], batch: int = 8):
    """``({feature name: [N, D] L2-normalised}, labels [N])`` at the particle centres of one
    tomogram: one 64^3 cube per particle (reflect padding), feature read at the centre."""
    import torch.nn.functional as F

    from tomocpt.dataManager.scaling import read_apix

    apix_read = emb.read_apix(read_apix(path))
    vol, apix_native = load_tomogram_at(path, apix_read)
    factor = apix_native / apix_read
    pad = VIEW
    padded = np.pad(vol, pad, mode="reflect")
    centres, labels = [], []
    for cls, xyz in sorted(coords_xyz.items()):
        for x, y, z in np.asarray(xyz, dtype=np.float64).reshape(-1, 3):
            centres.append(np.array([z, y, x]) * factor)
            labels.append(int(cls))
    feats: Dict[str, List[np.ndarray]] = {}
    for i in range(0, len(centres), batch):
        cubes, local = [], []
        for c in centres[i:i + batch]:
            lo = np.floor(c).astype(int) - VIEW // 2 + pad
            cubes.append(padded[lo[0]:lo[0] + VIEW, lo[1]:lo[1] + VIEW, lo[2]:lo[2] + VIEW])
            local.append(c - (lo - pad))
        x = torch.from_numpy(np.stack(cubes)[:, None]).to(emb.device)
        with torch.autocast(emb.device.type, dtype=torch.bfloat16, enabled=emb.device.type == "cuda"):
            out = emb.features(x, apix_read)
        pts = torch.as_tensor(np.stack(local), dtype=torch.float32, device=emb.device)
        grid = ((pts + 0.5) / VIEW * 2 - 1).flip(-1).reshape(len(local), 1, 1, 1, 3)
        for name, f in out.items():
            e = F.grid_sample(f.float(), grid, mode="bilinear", align_corners=False).reshape(len(local), -1)
            feats.setdefault(name, []).append(F.normalize(e, dim=1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in feats.items()}, np.asarray(labels)


def knn_probe(embs: Sequence[np.ndarray], labels: Sequence[np.ndarray], k: int = 5) -> Dict:
    """Leave-one-tomogram-out k-NN (cosine): ``{"per_class": {c: acc}, "balanced": mean}``."""
    correct: Dict[int, int] = {}
    total: Dict[int, int] = {}
    for t in range(len(embs)):
        if len(embs[t]) == 0:
            continue
        train_e = np.concatenate([e for i, e in enumerate(embs) if i != t])
        train_l = np.concatenate([lab for i, lab in enumerate(labels) if i != t])
        if len(train_e) == 0:
            continue
        nn = np.argsort(-(embs[t] @ train_e.T), axis=1)[:, :k]
        for row, lab in zip(nn, labels[t]):
            votes = np.bincount(train_l[row], minlength=int(max(train_l.max(), lab)) + 1)
            correct[lab] = correct.get(lab, 0) + int(np.argmax(votes) == lab)
            total[lab] = total.get(lab, 0) + 1
    per_class = {int(c): correct[c] / total[c] for c in sorted(total)}
    return {"per_class": per_class, "balanced": float(np.mean(list(per_class.values()))) if per_class else None}
