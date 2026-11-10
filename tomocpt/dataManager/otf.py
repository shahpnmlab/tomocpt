"""On-the-fly crop sampling (Gate 8c.1): crops drawn every epoch from the tomogram store.

Candidates are the corners of the chunking grid (stride ``chunk_stride``). Each gets the weight
``w = keep * r + w_neg``: ``keep`` = 1 when the box holds an available object (the chunking
rule), else the keep probability ``1 - drop_p`` of an empty chunk (the ``ALPHA`` rule as an
expected value); ``r`` = the repeat factor of its classes (``chunk_sampling=repeat_factor``, the
class frequencies counted with the keep probabilities), else 1; ``w_neg`` =
``negative_mask_weight`` when the box holds negative-mask voxels. With the defaults the expected
mix per epoch equals the chunk baseline's. A draw picks a candidate by weight and moves its corner
by a uniform integer jitter in ``[0, stride)`` per axis (clipped to the volume).

Draws are seeded by ``(seed, epoch, global rank)``; each rank draws ``ceil(n / world_size)``.
The validation set is drawn once with ``val_seed`` and split over ranks by index.
The crops then go through the unchanged pipeline v2 sample code (``render_targets``, zoom,
TorchIO transforms in workers).
"""

import math
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from tomocpt.dataManager.dataloading import Data, VolumeDatsetIO, _enum_value
from tomocpt.dataManager.tomostore import TomogramStore, read_tomogram_store
from tomocpt.defaultConfigs.otf_config import OnTheFlyConfig
from tomocpt.defaultConfigs.train_config import ChunkSampling


def _seed_of(*parts) -> int:
    from tomocpt.dataManager.chunking import _stable_seed

    return int(_stable_seed(*parts) % (2 ** 63))


def crop_candidates(store: TomogramStore) -> np.ndarray:
    """int64 ``[K, 3]`` corners (z, y, x) of the chunking grid of one store entry, in the
    order of TorchIO's ``GridSampler``."""
    from tomocpt.predict.multiclass import grid_locations

    n, s = int(store.chunk_size), int(store.chunk_stride)
    return np.asarray(grid_locations(store.shape, n, n - s), dtype=np.int64)[:, :3].reshape(-1, 3)


def _candidate_info(store: TomogramStore, corners: np.ndarray, n_classes: int):
    """Per candidate: ``has_object`` (chunking's non-empty rule), the set of available classes
    centred inside the box (repeat factors, as ``VolumeDatsetIO.chunk_classes``) and whether the
    box holds negative-mask voxels."""
    from tomocpt.labels.filaments import segments_near_box

    n = int(store.chunk_size)
    avail = np.array([bool(a) for a in store.meta.class_available], dtype=bool)
    pts = store.points
    cls = pts[:, 3].astype(np.int64) if len(pts) else np.zeros(0, np.int64)
    r = (pts[:, 4].astype(np.float64) / 2.0)[:, None] if len(pts) else np.zeros((0, 1))
    segs = store.segments
    seg_cls = np.round(segs[:, 6]).astype(np.int64) if segs is not None else None
    has_obj = np.zeros(len(corners), dtype=bool)
    classes: List[Set[int]] = []
    has_neg = np.zeros(len(corners), dtype=bool)
    for i, c in enumerate(corners):
        local = pts[:, :3] - c.astype(np.float32)[None, :] if len(pts) else np.zeros((0, 3), np.float32)
        near = np.all((local >= -r) & (local < n + r), axis=1)
        obj = bool(avail[cls[near]].any()) if len(pts) else False
        inside = np.all((local >= 0) & (local < n), axis=1)
        here = {int(k) for k in cls[inside]}
        if segs is not None:
            ssel = segments_near_box(segs, [int(v) for v in c], n)
            if ssel.any():
                obj = obj or bool(avail[seg_cls[ssel]].any())
                ends = segs[:, :6].reshape(-1, 2, 3) - c.astype(np.float32)[None, None, :]
                hit = np.any(np.all((ends >= 0) & (ends < n), axis=2), axis=1)
                here |= {int(k) for k in seg_cls[hit]}
        has_obj[i] = obj
        classes.append({k for k in here if 0 <= k < n_classes and avail[k]})
        if store.has_negative_mask:
            has_neg[i] = bool(store.negative_crop(c, n).any())
    return has_obj, classes, has_neg


def candidate_weights(stores: Sequence[TomogramStore], n_classes: int,
                      chunk_sampling: str = "uniform", repeat_factor_threshold: float = 0.1,
                      negative_mask_weight: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """``(index [K, 4] int64 (store, z, y, x), weights [K] float64, expected)`` over every
    candidate of ``stores``. ``expected`` = the expected number of chunks (or repeat-factor draws)
    of the grid baseline, ``sum(keep * r)`` (the negative-mask weight not included)."""
    idx_parts, keeps, rs, negs, all_classes = [], [], [], [], []
    for si, st in enumerate(stores):
        corners = crop_candidates(st)
        has_obj, classes, has_neg = _candidate_info(st, corners, n_classes)
        idx_parts.append(np.concatenate([np.full((len(corners), 1), si, np.int64), corners], axis=1))
        keeps.append(np.where(has_obj, 1.0, 1.0 - float(st.drop_p)))
        negs.append(has_neg)
        all_classes.extend(classes)
    if not idx_parts:
        return np.zeros((0, 4), np.int64), np.zeros(0), 0.0
    index = np.concatenate(idx_parts)
    keep = np.concatenate(keeps)
    has_neg = np.concatenate(negs)
    r = np.ones(len(keep))
    if _enum_value(chunk_sampling) == "repeat_factor":
        member = np.zeros((len(keep), int(n_classes)), dtype=bool)
        for i, cl in enumerate(all_classes):
            for c in cl:
                member[i, c] = True
        f = (keep[:, None] * member).sum(0) / max(float(keep.sum()), 1e-12)
        r_c = np.ones(int(n_classes))
        nz = f > 0
        r_c[nz] = np.maximum(1.0, np.sqrt(float(repeat_factor_threshold) / f[nz]))
        r = np.where(member.any(1), np.max(np.where(member, r_c[None, :], 0.0), axis=1), 1.0)
    w = keep * r
    expected = float(w.sum())
    if negative_mask_weight > 0:
        w = w + float(negative_mask_weight) * has_neg
    return index, w, expected


def draw_crops(index: np.ndarray, weights: np.ndarray, n: int, seed: int, epoch=0, rank: int = 0,
               world_size: int = 1, jitter: bool = True, stride: int = 32,
               shapes: Optional[Sequence[Sequence[int]]] = None, size: int = 64) -> np.ndarray:
    """int64 ``[ceil(n / world_size), 4]`` crops (store, z, y, x) of one rank and epoch: draws by
    ``weights`` with replacement, then a uniform jitter in ``[0, stride)`` per axis clipped to
    ``shapes[store] - size``. Seeded by ``(seed, epoch, rank)`` only."""
    m = int(math.ceil(int(n) / max(1, int(world_size))))
    if m <= 0 or len(index) == 0:
        return np.zeros((0, 4), np.int64)
    g = torch.Generator().manual_seed(_seed_of("otf", seed, epoch, rank))
    pick = torch.multinomial(torch.as_tensor(weights, dtype=torch.double), m, replacement=True, generator=g).numpy()
    out = index[pick].copy()
    if jitter and stride > 1:
        off = torch.randint(0, int(stride), (m, 3), generator=g).numpy()
        hi = np.array([[int(s) - int(size) for s in shapes[int(k)]] for k in out[:, 0]], dtype=np.int64)
        out[:, 1:] = np.clip(out[:, 1:] + off, 0, hi)
    return out


def _rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class CropSampler(DistributedSampler):
    """Yields crop descriptors ``(store, z, y, x)`` for :class:`OnTheFlyDataset`. A
    ``DistributedSampler`` subclass so Lightning neither replaces nor wraps it under DDP; the rank
    and world size are read when iterated. ``fixed`` = a fixed list (validation), split by index."""

    def __init__(self, index=None, weights=None, n: int = 0, seed: int = 42, jitter: bool = True, stride: int = 32,
                 shapes=None, size: int = 64, fixed: Optional[np.ndarray] = None):
        # no DistributedSampler.__init__: it needs a process group and a sized dataset
        self.index, self.weights, self.n = index, weights, int(n if fixed is None else len(fixed))
        self.seed, self.jitter, self.stride, self.shapes, self.size = seed, jitter, stride, shapes, size
        self.fixed = fixed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def crops(self) -> np.ndarray:
        rank, world = _rank_world()
        if self.fixed is not None:
            m = int(math.ceil(len(self.fixed) / world))
            idx = [(rank + world * j) % max(1, len(self.fixed)) for j in range(m)] if len(self.fixed) else []
            return self.fixed[idx]
        return draw_crops(self.index, self.weights, self.n, self.seed, self.epoch, rank, world, self.jitter,
                          self.stride, self.shapes, self.size)

    def __iter__(self):
        for row in self.crops():
            yield tuple(int(v) for v in row)

    def __len__(self) -> int:
        return int(math.ceil(self.n / _rank_world()[1]))


class OnTheFlyDataset(VolumeDatsetIO):
    """Pipeline v2 samples cut from the tomogram store; the key is a crop ``(store, z, y, x)``.
    Everything after the crop (zoom, ``render_targets``, transforms, batch keys) is
    ``VolumeDatsetIO._getitem_v2``."""

    def __init__(self, stores: Sequence[TomogramStore], transform=None, n_classes: Optional[int] = None,
                 zoom_aug_max: float = 1.0, band_max: float = 32.0, chunk_size: int = 64):
        super().__init__([(None, None, None)], transform=transform, pipeline_version=2,
                         metas={s.path: s.meta for s in stores}, n_classes=n_classes, zoom_aug_max=zoom_aug_max,
                         band_max=band_max, chunk_size=chunk_size)
        self.stores = list(stores)
        self.has_negative_masks = any(s.has_negative_mask for s in self.stores)
        self._key = None

    def __len__(self):
        return sum(len(crop_candidates(s)) for s in self.stores)

    def __getitem__(self, key) -> dict:
        self._key = tuple(int(k) for k in key)
        return self._getitem_v2(0)

    def _store_corner(self):
        si, z, y, x = self._key
        return self.stores[si], (z, y, x)

    def _load_v2(self, index: int):
        st, corner = self._store_corner()
        vol = torch.from_numpy(st.crop(corner, self.chunk_size))[None]
        return vol, st.points_in_box(corner, self.chunk_size), None, st.meta

    def _load_segments(self, tgt_path) -> Optional[np.ndarray]:
        st, corner = self._store_corner()
        return st.segments_in_box(corner, self.chunk_size)

    def _load_negative_mask(self, vol_path) -> torch.Tensor:
        st, corner = self._store_corner()
        m = st.negative_crop(corner, self.chunk_size)
        n = self.chunk_size
        if m is None:
            return torch.zeros(1, n, n, n)
        return torch.from_numpy(m.astype(np.float32))[None]


class CropDescriptorDataset(torch.utils.data.Dataset):
    """GPU path (``augment_device=cuda``): a crop key ``(store, z, y, x)`` -> its descriptor and
    per-sample metadata; :class:`~tomocpt.dataManager.gpu_augment.GpuAugment` does the rest."""

    def __init__(self, stores: Sequence[TomogramStore], train: bool):
        self.stores = list(stores)
        self.train = bool(train)

    def __len__(self):
        return sum(len(crop_candidates(s)) for s in self.stores)

    def __getitem__(self, key) -> dict:
        si, z, y, x = (int(k) for k in key)
        meta = self.stores[si].meta
        return {"store_index": torch.tensor(si, dtype=torch.int64),
                "corner": torch.tensor([z, y, x], dtype=torch.int64),
                "otf_train": torch.tensor(self.train),
                "class_available": torch.tensor([bool(a) for a in meta.class_available], dtype=torch.bool),
                "class_d_vox": torch.tensor(meta.class_d_vox, dtype=torch.float32),
                "apix_work": torch.tensor([float(meta.apix_work)], dtype=torch.float32),
                "dataset_id": torch.tensor(int(meta.dataset_id), dtype=torch.int64)}


class OnTheFlyData(Data):
    """LightningDataModule of ``train.sampling=on_the_fly`` (pipeline v2, picking)."""

    def __init__(self, data_dir, return_labels, batch_size, workers_for_data, augmentation=None,
                 pipeline_version: int = 2, scale_cfg=None, chunk_sampling=ChunkSampling.uniform,
                 repeat_factor_threshold: float = 0.1, seed: Optional[int] = None, pin_memory: bool = False,
                 prefetch_factor: Optional[int] = None, otf: Optional[OnTheFlyConfig] = None):
        super().__init__(data_dir, return_labels, batch_size, workers_for_data, augmentation=augmentation,
                         pipeline_version=pipeline_version, scale_cfg=scale_cfg, chunk_sampling=chunk_sampling,
                         repeat_factor_threshold=repeat_factor_threshold, seed=seed, pin_memory=pin_memory,
                         prefetch_factor=prefetch_factor)
        self.otf = otf if otf is not None else OnTheFlyConfig()
        if self.pipeline_version != 2:
            raise ValueError("train.sampling=on_the_fly needs train.data_pipeline_version=2")
        self.augment_device = str(_enum_value(self.otf.augment_device))
        if self.augment_device not in ("cpu", "cuda"):
            raise ValueError(f"train.otf.augment_device must be 'cpu' or 'cuda', got '{self.augment_device}'")
        if self.augment_device == "cuda":
            from tomocpt.dataManager.gpu_augment import check_gpu_augmentation

            check_gpu_augmentation(self.augmentation)
        self.gpu_train = self.gpu_val = None

    def _setup_v2(self, layout: dict):
        from tomocpt.dataManager.dataloading import build_training_transforms
        from tomocpt.labels.multiclass import class_table_from_json

        if layout.get("storage") != "tomogram":
            raise ValueError(f"storage mismatch: {self.hparams.data_dir} holds chunk files but "
                             "train.sampling=on_the_fly needs a tomogram store. Use another chunks_dir, or set "
                             "train.sampling=chunks.")
        if not self.hparams.return_labels:
            raise ValueError("train.sampling=on_the_fly supports picking only")
        self.class_table = class_table_from_json(layout["class_table"])
        n_classes = len(self.class_table)
        self.chunk_size = int(layout.get("chunk_size", 64))
        self.chunk_stride = int(layout.get("chunk_stride", 32))
        self.stores_train = read_tomogram_store(self._get_split_dir("train"))
        self.stores_val = read_tomogram_store(self._get_split_dir("val"))
        if not self.stores_train:
            raise FileNotFoundError(f"No tomogram store in {self._get_split_dir('train')}")
        common = dict(n_classes=n_classes, band_max=float(self.scale_cfg.band_max_vox), chunk_size=self.chunk_size)
        if self.augment_device == "cuda":
            from tomocpt.dataManager.gpu_augment import GpuAugment

            dev = str(_enum_value(self.otf.device))
            self.dataset_training = CropDescriptorDataset(self.stores_train, train=True)
            self.dataset_val = CropDescriptorDataset(self.stores_val, train=False)
            self.gpu_train = GpuAugment(self.stores_train, augmentation=self.augmentation, device=dev, train=True,
                                        zoom_aug_max=float(self.scale_cfg.zoom_aug_max), **common)
            self.gpu_val = GpuAugment(self.stores_val, augmentation=self.augmentation, device=dev, train=False,
                                      **common)
        else:
            self.dataset_training = OnTheFlyDataset(
                self.stores_train, transform=build_training_transforms(self.augmentation),
                zoom_aug_max=float(self.scale_cfg.zoom_aug_max), **common)
            self.dataset_val = OnTheFlyDataset(self.stores_val, transform=None, zoom_aug_max=1.0, **common)
        kw = dict(n_classes=n_classes, chunk_sampling=_enum_value(self.chunk_sampling),
                  repeat_factor_threshold=self.repeat_factor_threshold,
                  negative_mask_weight=float(self.otf.negative_mask_weight))
        self.train_index, self.train_weights, expected = candidate_weights(self.stores_train, **kw)
        self.samples_per_epoch = int(self.otf.samples_per_epoch or max(1, round(expected)))
        if self.stores_val:
            v_index, v_weights, v_expected = candidate_weights(self.stores_val, **kw)
            kw_unif = dict(kw, chunk_sampling="uniform", negative_mask_weight=0.0)
            n_val = int(self.otf.val_samples or max(1, round(candidate_weights(self.stores_val, **kw_unif)[2])))
            self.val_crops = draw_crops(v_index, v_weights, n_val, int(self.otf.val_seed), epoch="val", rank=0,
                                        world_size=1, jitter=bool(self.otf.jitter), stride=self.chunk_stride,
                                        shapes=[s.shape for s in self.stores_val], size=self.chunk_size)
        else:
            self.val_crops = np.zeros((0, 4), np.int64)
        rank_zero_info(f"On-the-fly sampling: {len(self.stores_train)} training / {len(self.stores_val)} validation "
                       f"store entries, {len(self.train_index)} candidates, {self.samples_per_epoch} crops per epoch "
                       f"(grid expectation {expected:.1f}), {len(self.val_crops)} fixed validation crops")

    def train_sampler(self) -> CropSampler:
        return CropSampler(self.train_index, self.train_weights, self.samples_per_epoch, seed=int(self.otf.seed),
                           jitter=bool(self.otf.jitter), stride=self.chunk_stride,
                           shapes=[s.shape for s in self.stores_train], size=self.chunk_size)

    def val_sampler(self) -> CropSampler:
        return CropSampler(fixed=self.val_crops)

    def _loader(self, dataset, sampler):
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers_for_data,
                          sampler=sampler, persistent_workers=self._persistent_workers(), **self._loader_kwargs())

    def train_dataloader(self):
        return self._loader(self.dataset_training, self.train_sampler())

    def val_dataloader(self):
        return self._loader(self.dataset_val, self.val_sampler())

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """GPU path: crop descriptors -> the training batch, on the batch's device."""
        if isinstance(batch, dict) and "store_index" in batch and self.gpu_train is not None:
            train = bool(batch["otf_train"].reshape(-1)[0])
            return (self.gpu_train if train else self.gpu_val)(batch)
        return batch
