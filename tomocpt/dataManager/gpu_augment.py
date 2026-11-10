"""Augmentation and target rendering on the device of the batch (Gate 8c.2,
``train.otf.augment_device=cuda``).

With the GPU path the loader yields crop descriptors (``store_index``, ``corner``, ``otf_train``
and the per-sample metadata). :class:`GpuAugment` runs in
``OnTheFlyData.on_after_batch_transfer`` and turns them into the Section 5.1 batch:

- **Spatial.** One linear map ``A`` per sample (rotation and TorchIO's +-10 % per-axis scale from
  Kornia's ``RandomAffine3D``, or axis flips with ``rotation_mode=flips``; zoom augmentation
  divides it by ``s``) and, in ``affine`` mode, the elastic/blur choice of the CPU pipeline.
  Output voxel ``j`` reads the whole tomogram at
  ``corner + c + A (j + e(j) - c) + t`` (``c = (n - 1) / 2``, ``e`` the elastic displacement,
  ``t`` the zoom offset), trilinear; outside the tomogram the value is its median (the CPU path
  pads with ``otsu``: a documented distribution change). Pure flips are exact slices.
- **Targets** are rendered after the transform: points (``p -> c + A^-1 (p - corner - c - t)``,
  ``d_vox * |det A|^(-1/3)``) and segment ends are mapped and drawn with
  :mod:`tomocpt.labels.render_torch`. With an elastic field the target is rendered in the affine
  frame (with a margin) and warped by ``j + e(j)``, ``linear`` or ``nearest`` as
  ``label_interpolation``. Negative masks are read nearest with the same grid.
- **Intensity** (input only): blur (per-axis std in ``[0, blur_std]``, Kornia ``filter3d``), then
  contrast inversion at ``CONTRAST_INVERSION_PROB`` (always on).
  The Fourier degradation and the optional intensity transforms are not ported yet:
  :func:`check_gpu_augmentation` raises ``ValueError`` naming the CPU path.

Device agnostic: tests run it on CPU tensors.
"""
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tomocpt import constants
from tomocpt.labels.render_torch import render_targets_torch, render_tubes_torch

#: augmentation fields whose GPU port does not exist yet (value that means "off")
_UNPORTED = {"use_mw_aug": False, "use_fourier_aug": False, "brightness_gradient_p": 0.0, "local_gamma_p": 0.0,
             "brightness_p": 0.0, "contrast_p": 0.0, "anisotropy_p": 0.0, "noise_p": 0.0, "gamma_p": 0.0,
             "bias_field_p": 0.0}


def _ev(v):
    return getattr(v, "value", v)


def check_gpu_augmentation(cfg) -> None:
    """``ValueError`` when ``cfg`` turns on an augmentation the GPU path does not have yet."""
    if cfg is None:
        return
    on = [k for k, off in _UNPORTED.items() if getattr(cfg, k, off) not in (off, None) and getattr(cfg, k, off)]
    if on:
        raise ValueError(f"train.otf.augment_device='cuda' does not support {on} yet (no GPU port); "
                         "use train.otf.augment_device=cpu for these augmentations")


# --------------------------------------------------------------------------------------------
# random parameters
# --------------------------------------------------------------------------------------------

def random_affine(n: int, degrees: float = 45.0, p: float = 0.8, scale=(0.9, 1.1)) -> torch.Tensor:
    """``[n, 3, 3]`` float64 linear maps (z, y, x): Kornia ``RandomAffine3D`` rotation and
    per-axis scale with probability ``p``, identity otherwise."""
    import kornia.augmentation as K

    if n <= 0:
        return torch.zeros(0, 3, 3, dtype=torch.float64)
    aug = K.RandomAffine3D(degrees=float(degrees), scale=(tuple(scale),) * 3, p=float(p), same_on_batch=False)
    aug(torch.zeros(n, 1, 2, 2, 2))
    M = aug.transform_matrix[:, :3, :3].to(torch.float64)       # (x, y, z) order
    return M.flip(1).flip(2).contiguous()                        # -> (z, y, x)


def _bspline3(x: torch.Tensor) -> torch.Tensor:
    a = x.abs()
    return torch.where(a < 1, (4 - 6 * a ** 2 + 3 * a ** 3) / 6,
                       torch.where(a < 2, (2 - a) ** 3 / 6, torch.zeros_like(a)))


def _bspline_matrix(size: int, n_ctrl: int, dtype=torch.float64) -> torch.Tensor:
    """``[size, n_ctrl]`` weights of SimpleITK's cubic B-spline over a ``size``-voxel axis
    (``BSplineTransformInitializer``: mesh ``n_ctrl - 3`` over the voxel-edge domain)."""
    h = size / (n_ctrl - 3)
    u = (torch.arange(size, dtype=dtype) + 0.5) / h + 1.0
    return _bspline3(u[:, None] - torch.arange(n_ctrl, dtype=dtype)[None])


def elastic_field(n: int, size: int, num_control_points: int = 7, max_displacement: float = 7.5,
                  num_locked_borders: int = 2, device=None) -> torch.Tensor:
    """``[n, 3, size, size, size]`` float32 displacement fields (voxels, z, y, x components):
    TorchIO's ``RandomElasticDeformation`` control grid (its own ``get_params``, its defaults)
    upsampled with its cubic B-spline."""
    from torchio.transforms.augmentation.spatial.random_elastic_deformation import RandomElasticDeformation

    cp = (int(num_control_points),) * 3
    md = (float(max_displacement),) * 3
    coarse = torch.stack([torch.as_tensor(RandomElasticDeformation.get_params(cp, md, int(num_locked_borders)))
                          for _ in range(int(n))]).to(torch.float64)          # [n, k, k, k, 3]
    Bm = _bspline_matrix(int(size), cp[0])
    f = torch.einsum("ia,jb,kc,nabcd->ndijk", Bm, Bm, Bm, coarse)
    return f.to(device=device, dtype=torch.float32)


# --------------------------------------------------------------------------------------------
# the augmenter
# --------------------------------------------------------------------------------------------

class GpuAugment:
    """Crop descriptors -> training batch on the descriptors' device (see the module doc).

    ``train=False`` draws nothing (validation); explicit ``affine [B, 3, 3]`` and ``elastic
    [B, 3, n, n, n]`` to ``__call__`` replace the random spatial draw (tests).
    ``device``: ``auto`` keeps tomograms on the batch's GPU when they fit (half the free memory),
    ``cuda`` always, ``cpu`` never (crops are read from the memory-mapped store)."""

    def __init__(self, stores: Sequence, n_classes: int, augmentation=None, chunk_size: int = 64,
                 zoom_aug_max: float = 1.0, band_max: float = 32.0, device: str = "auto", train: bool = True):
        from tomocpt.defaultConfigs.train_config import AugmentationConfig

        self.stores = list(stores)
        self.n_classes = int(n_classes)
        self.cfg = augmentation if augmentation is not None else AugmentationConfig()
        check_gpu_augmentation(self.cfg)
        self.n = int(chunk_size)
        self.zoom_aug_max = float(zoom_aug_max)
        self.band_max = float(band_max)
        self.device_mode = str(_ev(device))
        self.train = bool(train)
        self.mode = str(_ev(getattr(self.cfg, "rotation_mode", "affine")))
        self.label_interp = str(_ev(getattr(self.cfg, "label_interpolation", "nearest")))
        self.has_negative_masks = any(getattr(s, "has_negative_mask", False) for s in self.stores)
        self._pad: List[Optional[float]] = [None] * len(self.stores)
        self._res = {}      # (store index, device) -> dict(vol, neg, pts, segs)
        self._resident = {}  # device -> bool

    # ---------------------------------------------------------------- residency
    @property
    def pad_values(self) -> List[float]:
        for i, s in enumerate(self.stores):
            if self._pad[i] is None:
                self._pad[i] = float(np.median(np.asarray(s.volume).reshape(-1)))
        return list(self._pad)

    def _fits(self, device) -> bool:
        if device.type != "cuda" or self.device_mode == "cpu":
            return False
        if device not in self._resident:
            if self.device_mode == "cuda":
                ok = True
            else:
                need = sum(int(np.prod(s.shape)) * np.dtype(s.volume_dtype).itemsize * (1.0 + 1.0 / 16)
                           for s in self.stores)
                free, _ = torch.cuda.mem_get_info(device)
                ok = need <= 0.5 * free
            self._resident[device] = ok
        return self._resident[device]

    def _store(self, si: int, device):
        key = (si, device)
        if key not in self._res:
            s = self.stores[si]
            gpu = self._fits(device)
            vol = torch.from_numpy(np.ascontiguousarray(s.volume)).to(device) if gpu else None
            neg = None
            if s.has_negative_mask:
                from tomocpt.dataManager.tomostore import NEGATIVE_MASK_FNAME
                from pathlib import Path
                packed = np.load(Path(s.path) / NEGATIVE_MASK_FNAME, mmap_mode="r")
                neg = torch.from_numpy(np.ascontiguousarray(packed)).to(device) if gpu else packed
            pts = torch.as_tensor(np.asarray(s.points, np.float32)).to(device=device, dtype=torch.float64)
            segs = None
            if s.segments is not None and len(s.segments):
                segs = torch.as_tensor(np.asarray(s.segments, np.float32)).to(device=device, dtype=torch.float64)
            self._res[key] = dict(vol=vol, neg=neg, pts=pts, segs=segs)
        return self._res[key]

    def _box(self, si: int, lo, hi, device) -> torch.Tensor:
        """float32 ``[hi - lo]`` box of the padded volume, outside filled with its median."""
        s = self.stores[si]
        r = self._store(si, device)
        size = [h - l for l, h in zip(lo, hi)]
        out = torch.full(size, self.pad_values[si], dtype=torch.float32, device=device)
        a = [max(0, l) for l in lo]
        b = [min(d, h) for d, h in zip(s.shape, hi)]
        if all(bb > aa for aa, bb in zip(a, b)):
            dst = tuple(slice(aa - l, bb - l) for aa, bb, l in zip(a, b, lo))
            src = tuple(slice(aa, bb) for aa, bb in zip(a, b))
            if r["vol"] is not None:
                out[dst] = r["vol"][src].to(torch.float32)
            else:
                out[dst] = torch.from_numpy(np.asarray(s.volume[src], dtype=np.float32)).to(device)
        return out

    def _neg_at(self, si: int, idx: torch.Tensor, device) -> torch.Tensor:
        """bool mask values at integer positions ``idx [3, ...]`` (outside = False)."""
        s = self.stores[si]
        r = self._store(si, device)
        z, y, x = idx[0], idx[1], idx[2]
        inb = (z >= 0) & (z < s.shape[0]) & (y >= 0) & (y < s.shape[1]) & (x >= 0) & (x < s.shape[2])
        zc, yc, xc = z.clamp(0, s.shape[0] - 1), y.clamp(0, s.shape[1] - 1), x.clamp(0, s.shape[2] - 1)
        packed = r["neg"]
        if isinstance(packed, torch.Tensor):
            byte = packed[zc, yc, xc >> 3].to(torch.int64)
        else:
            byte = torch.from_numpy(np.asarray(packed[zc.cpu().numpy(), yc.cpu().numpy(),
                                                      (xc >> 3).cpu().numpy()])).to(device=device, dtype=torch.int64)
        bit = (byte >> (7 - (xc & 7))) & 1
        return (bit > 0) & inb

    # ---------------------------------------------------------------- random draws
    def sample_affines(self, b: int) -> torch.Tensor:
        """``[b, 3, 3]`` float64 random linear maps of the configured ``rotation_mode``."""
        if self.mode == "flips":
            p = float(getattr(self.cfg, "flip_p", 0.0))
            sign = torch.where(torch.rand(b, 3) < p, -1.0, 1.0).to(torch.float64)
            return torch.diag_embed(sign)
        return random_affine(b, degrees=float(self.cfg.affine_degrees), p=float(self.cfg.affine_p))

    def sample_elastic(self, b: int) -> Optional[torch.Tensor]:
        if self.mode == "flips" or b <= 0:
            return None
        return elastic_field(b, self.n)

    def _elastic_blur_choice(self, b: int):
        """Per sample (elastic, blur) booleans: the CPU ``OneOf`` with probability ``elastic_blur_p``."""
        p = float(self.cfg.elastic_blur_p)
        we = float(self.cfg.elastic_weight) if self.mode != "flips" else 0.0
        wb = float(self.cfg.blur_weight)
        fire = torch.rand(b) < p
        if we + wb <= 0:
            z = torch.zeros(b, dtype=torch.bool)
            return z, z.clone()
        pick_e = torch.rand(b) < we / (we + wb)
        return fire & pick_e, fire & ~pick_e

    # ---------------------------------------------------------------- intensity
    def intensity(self, x: torch.Tensor, blur: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Blur (where ``blur``; drawn from the elastic/blur choice when None), then contrast
        inversion about the sample mean at ``CONTRAST_INVERSION_PROB``. Input only."""
        b = x.shape[0]
        if blur is None:
            blur = self._elastic_blur_choice(b)[1]
        blur = blur.to(x.device)
        if bool(blur.any()):
            from kornia.filters import filter3d, get_gaussian_kernel3d

            idx = torch.nonzero(blur).flatten()
            std = float(self.cfg.blur_std)
            sig = (torch.rand(len(idx), 3, device=x.device) * std).clamp_min(1e-3)
            k = 2 * int(np.ceil(4.0 * std)) + 1
            kern = get_gaussian_kernel3d((k, k, k), sig.to(x.dtype))
            x = x.clone()
            x[idx] = filter3d(x[idx], kern, border_type="reflect")
        inv = (torch.rand(b, device=x.device) < constants.CONTRAST_INVERSION_PROB)
        if bool(inv.any()):
            m = x.mean(dim=tuple(range(1, x.dim())), keepdim=True)
            x = torch.where(inv.view(-1, *([1] * (x.dim() - 1))), 2.0 * m - x, x)
        return x

    # ---------------------------------------------------------------- the batch
    def __call__(self, batch: dict, affine: Optional[torch.Tensor] = None,
                 elastic: Optional[torch.Tensor] = None) -> dict:
        si = batch["store_index"].reshape(-1)
        corner = batch["corner"].reshape(-1, 3)
        device = corner.device
        B, n, C = len(si), self.n, self.n_classes
        rand = torch.ones(B, dtype=torch.bool) if "otf_train" not in batch else batch["otf_train"].reshape(-1).cpu()
        rand = rand & self.train & (affine is None)
        c = (n - 1) / 2.0

        A = torch.eye(3, dtype=torch.float64).repeat(B, 1, 1)
        if affine is not None:
            A = affine.detach().to("cpu", torch.float64).reshape(B, 3, 3).clone()
        elif bool(rand.any()):
            A[rand] = self.sample_affines(int(rand.sum()))
        zoom = torch.ones(B, dtype=torch.float64)
        t = torch.zeros(B, 3, dtype=torch.float64)
        if self.zoom_aug_max > 1.0 and bool(rand.any()):
            s = 1.0 + torch.rand(B, dtype=torch.float64) * (self.zoom_aug_max - 1.0)
            zoom = torch.where(rand, s, zoom)
            half = c * (1.0 - 1.0 / zoom)
            t = (torch.rand(B, 3, dtype=torch.float64) * 2 - 1) * half[:, None]
            A = A / zoom[:, None, None]
        fields: List[Optional[torch.Tensor]] = [None] * B
        blur = torch.zeros(B, dtype=torch.bool)
        if elastic is not None:
            for b in range(B):
                fields[b] = elastic[b].to(device=device, dtype=torch.float32)
        elif bool(rand.any()):
            do_e, blur = self._elastic_blur_choice(B)
            do_e, blur = do_e & rand, blur & rand
            if bool(do_e.any()):
                ef = self.sample_elastic(int(do_e.sum()))
                if ef is not None:
                    for j, b in enumerate(torch.nonzero(do_e).flatten().tolist()):
                        fields[b] = ef[j].to(device)

        vols, targets, negs = [], [], []
        pts_all, pbi, segs_all, sbi = [], [], [], []
        warped_targets = {}
        for b in range(B):
            k = int(si[b])
            r = self._store(k, device)
            cb = corner[b].to("cpu", torch.float64)
            Ab = A[b]
            Ai = torch.linalg.inv(Ab)
            off = cb + c + t[b]
            scale_d = float(abs(torch.linalg.det(Ab))) ** (-1.0 / 3.0)
            # input and negative mask
            diag = torch.diagonal(Ab)
            pure_flip = (fields[b] is None and torch.equal(Ab, torch.diag(diag)) and bool((diag.abs() == 1).all())
                         and bool((t[b] == 0).all()))
            if pure_flip:
                lo = [int(v) for v in corner[b].tolist()]
                box = self._box(k, lo, [l + n for l in lo], device)
                dims = [a for a in range(3) if diag[a] < 0]
                vols.append(box.flip(dims) if dims else box)
                if self.has_negative_masks:
                    idx = torch.stack(torch.meshgrid(*(torch.arange(n, device=device) + l for l in lo),
                                                     indexing="ij"))
                    m = self._neg_at(k, idx, device) if r["neg"] is not None else torch.zeros((n,) * 3, dtype=torch.bool,
                                                                                              device=device)
                    negs.append(m.flip(dims) if dims else m)
            else:
                src = self._source(Ab.to(device), off.to(device), fields[b], device)          # [3, n, n, n] float64
                lo = torch.floor(src.reshape(3, -1).min(1).values).long().tolist()
                hi = (torch.floor(src.reshape(3, -1).max(1).values).long() + 2).tolist()
                box = self._box(k, lo, hi, device)
                loc = (src - torch.tensor(lo, dtype=torch.float64, device=device).view(3, 1, 1, 1))
                vols.append(self._sample(box, loc, "bilinear"))
                if self.has_negative_masks:
                    if r["neg"] is not None:
                        negs.append(self._neg_at(k, torch.round(src).long(), device))
                    else:
                        negs.append(torch.zeros((n,) * 3, dtype=torch.bool, device=device))
            # targets: map points / segments into the affine frame
            Ai_d, off_d = Ai.to(device), off.to(device)
            margin = 0
            if fields[b] is not None:
                margin = int(np.ceil(float(fields[b].abs().max()))) + 1
            p = r["pts"]
            mp = None
            if len(p):
                q = c + (p[:, :3] - off_d) @ Ai_d.T + margin
                dv = p[:, 4:5] * scale_d
                Rr = dv / 2.0
                lim = n + 2 * margin
                sel = ((q >= -Rr) & (q < lim + Rr)).all(1)
                mp = torch.cat([q[sel], p[sel, 3:4], dv[sel]], 1)
            ms = None
            if r["segs"] is not None:
                sg = r["segs"]
                a0 = c + (sg[:, 0:3] - off_d) @ Ai_d.T + margin
                a1 = c + (sg[:, 3:6] - off_d) @ Ai_d.T + margin
                ms = torch.cat([a0, a1, sg[:, 6:7], sg[:, 7:8] * scale_d], 1)
                lim = n + 2 * margin
                R2 = ms[:, 7:8] / 2.0
                near = ((torch.maximum(a0, a1) + R2 >= 0) & (torch.minimum(a0, a1) - R2 <= lim - 1)).all(1)
                ms = ms[near]
            if fields[b] is None:
                if mp is not None and len(mp):
                    pts_all.append(mp)
                    pbi.append(torch.full((len(mp),), b, dtype=torch.int64, device=device))
                if ms is not None and len(ms):
                    segs_all.append(ms)
                    sbi.append(torch.full((len(ms),), b, dtype=torch.int64, device=device))
            else:
                shp = (n + 2 * margin,) * 3
                tg = render_targets_torch(mp if mp is not None else torch.zeros(0, 5, device=device), shp, C)
                if ms is not None and len(ms):
                    tg = torch.maximum(tg, render_tubes_torch(ms, shp, C))
                j = torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float64),) * 3,
                                               indexing="ij"))
                loc = j + fields[b].to(torch.float64) + margin
                mode = "bilinear" if self.label_interp == "linear" else "nearest"
                warped_targets[b] = self._sample(tg, loc, mode, pad=0.0)

        tgt = torch.zeros((B, C, n, n, n), dtype=torch.float32, device=device)
        if pts_all:
            tgt = torch.maximum(tgt, render_targets_torch(torch.cat(pts_all), (n,) * 3, C,
                                                          batch_index=torch.cat(pbi), batch_size=B))
        if segs_all:
            tgt = torch.maximum(tgt, render_tubes_torch(torch.cat(segs_all), (n,) * 3, C,
                                                        batch_index=torch.cat(sbi), batch_size=B))
        for b, tg in warped_targets.items():
            tgt[b] = tg

        x = torch.stack(vols)[:, None]
        if bool(rand.any()):
            x_r = self.intensity(x[rand.to(device)], blur=blur[rand])
            x = x.clone()
            x[rand.to(device)] = x_r

        out = {k: v for k, v in batch.items() if k not in ("store_index", "corner", "otf_train")}
        out["input_data"] = x.contiguous()
        out["target_data"] = tgt
        if self.has_negative_masks:
            out["negative_mask"] = torch.stack(negs)[:, None]
        if "class_d_vox" in out and bool((zoom != 1).any()):
            z = zoom.to(device=device, dtype=torch.float32)
            out["class_d_vox"] = out["class_d_vox"] * z[:, None]
            out["apix_work"] = out["apix_work"] / z[:, None]
            if "class_available" in out:
                out["class_available"] = out["class_available"] & (out["class_d_vox"] <= self.band_max * (1 + 1e-6))
        return out

    # ---------------------------------------------------------------- helpers
    def _source(self, A, off, field, device) -> torch.Tensor:
        n = self.n
        c = (n - 1) / 2.0
        j = torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float64),) * 3, indexing="ij"))
        if field is not None:
            j = j + field.to(torch.float64)
        q = (j.reshape(3, -1) - c)
        return (A @ q + off[:, None]).reshape(3, n, n, n)

    @staticmethod
    def _sample(vol: torch.Tensor, loc: torch.Tensor, mode: str, pad: Optional[float] = None) -> torch.Tensor:
        """Read ``vol [..., D, H, W]`` (3-D or 4-D) at voxel positions ``loc [3, n, n, n]``
        (z, y, x), ``align_corners=True``; outside: border (boxes already hold the padding) or
        zeros when ``pad == 0``."""
        squeeze = vol.dim() == 3
        v = vol[None, None] if squeeze else vol[None]
        dims = torch.tensor(vol.shape[-3:], dtype=torch.float64, device=loc.device).view(3, 1, 1, 1)
        g = (2.0 * loc / (dims - 1).clamp_min(1.0) - 1.0)
        grid = torch.stack([g[2], g[1], g[0]], dim=-1)[None].to(torch.float32)
        out = F.grid_sample(v.to(torch.float32), grid, mode=mode,
                            padding_mode="zeros" if pad == 0.0 else "border", align_corners=True)[0]
        return out[0] if squeeze else out
