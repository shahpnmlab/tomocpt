import math
import os
from typing import Dict, List, Optional, Any, Sequence, Set
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
# No longer need to import is_main_process from utils, it's used inside the Lightning hook
from pytorch_lightning.utilities.rank_zero import rank_zero_info


from tomocpt import constants
from tomocpt.dataManager.fourier_augmentations import RandomFourierDegradation
from tomocpt.dataManager.intensity_augmentations import (
    RandomBrightness,
    RandomBrightnessGradient,
    RandomContrast,
    RandomContrastInversion,
    RandomLocalGamma,
)
from tomocpt.defaultConfigs.scale_config import ScaleConfig
from tomocpt.defaultConfigs.train_config import AugmentationConfig, ChunkSampling
from tomocpt.mainConfig import mainConfig
from tomocpt.logger import get_logger

logging = get_logger()


def _enum_value(v) -> str:
    return str(getattr(v, "value", v))


def build_training_transforms(cfg: Optional[AugmentationConfig] = None) -> tio.Compose:
    """Compose the training-time augmentation pipeline.

    Parameters
    ----------
    cfg : AugmentationConfig, optional
        Augmentation settings. Defaults to :class:`AugmentationConfig`, whose
        defaults reproduce tomocpt's historical pipeline exactly.

    Returns
    -------
    tio.Compose
        The composed pipeline.

    Notes
    -----
    Note that the returned pipeline is **not** identical to tomocpt's historical
    one even at default settings: contrast inversion is baked in unconditionally.
    Every other augmentation added here is off by default.

    Ordering matters and follows the reference pipeline:

    1. ``tio.RandomAffine`` - image and label rotated together.
    2. ``tio.OneOf({RandomElasticDeformation, RandomBlur})``.
    3. :class:`RandomFourierDegradation` - **after all spatial transforms**.
    4. Intensity transforms, contrast inversion first. Contrast inversion is
       always applied at :data:`tomocpt.constants.CONTRAST_INVERSION_PROB` and is not configurable.

    Step 3's position is the entire point of the augmentation: running after the
    rotations makes the missing wedge land at a random orientation relative to
    the specimen, while the label stays geometrically correct. Moving it earlier
    would silently destroy the augmentation while still training normally.
    """
    if cfg is None:
        cfg = AugmentationConfig()

    rotation_mode = _enum_value(getattr(cfg, "rotation_mode", "affine"))
    label_interpolation = _enum_value(getattr(cfg, "label_interpolation", "nearest"))
    if rotation_mode not in ("affine", "flips"):
        raise ValueError(f"rotation_mode must be 'affine' or 'flips', got '{rotation_mode}'")
    if label_interpolation not in ("nearest", "linear"):
        raise ValueError(f"label_interpolation must be 'nearest' or 'linear', got '{label_interpolation}'")
    # The target stays a tio.LabelMap (invariant 10); only its interpolation changes.
    label_kw = {"label_interpolation": "linear"} if label_interpolation == "linear" else {}

    if rotation_mode == "flips":
        # Axis flips keep the real missing wedge where it is. Elastic deformation is dropped
        # (it moves wedge power; human decision, 27 Oct 2026); the OneOf keeps the blur.
        transforms = [tio.RandomFlip(axes=(0, 1, 2), flip_probability=getattr(cfg, "flip_p", 0.0))]
        if cfg.blur_weight > 0:
            transforms.append(tio.OneOf({tio.RandomBlur(std=cfg.blur_std): cfg.blur_weight}, p=cfg.elastic_blur_p))
    else:
        transforms = [
            tio.RandomAffine(degrees=cfg.affine_degrees, default_pad_value="otsu", p=cfg.affine_p, **label_kw),
            tio.OneOf({tio.RandomElasticDeformation(**label_kw): cfg.elastic_weight,
                       tio.RandomBlur(std=cfg.blur_std): cfg.blur_weight}, p=cfg.elastic_blur_p),
        ]

    # --- Fourier degradation: strictly after every spatial transform ---
    if cfg.use_mw_aug or cfg.use_fourier_aug:
        transforms.append(
            RandomFourierDegradation(
                missing_wedge_aug=cfg.use_mw_aug,
                amplitude_aug=cfg.use_fourier_aug,
                missing_wedge_prob=cfg.missing_wedge_prob,
                amplitude_prob=cfg.amplitude_prob,
                sample_kernel_prob=cfg.sample_kernel_prob,
                keep_angle_range=(cfg.keep_angle_min, cfg.keep_angle_max),
                smooth_sigma_range=(cfg.smooth_sigma_min, cfg.smooth_sigma_max),
                step_sigma_range=(cfg.step_sigma_min, cfg.step_sigma_max),
                offset_range=(cfg.offset_min, cfg.offset_max),
                taper_width=cfg.taper_width,
                reclip=cfg.reclip,
                clip_value=cfg.clip_value,
            )
        )

    # --- Intensity transforms ---
    # Contrast inversion is always on, deliberately not configurable: tomogram
    # polarity is a property of the data rather than a tuning knob, and a picker
    # that is invariant to it is wanted in every run. It goes first because
    # polarity is established before any acquisition effect, and because
    # RandomLocalGamma is not symmetric about the mean.
    transforms.append(RandomContrastInversion(p=constants.CONTRAST_INVERSION_PROB))
    if cfg.brightness_gradient_p > 0:
        transforms.append(RandomBrightnessGradient(p=cfg.brightness_gradient_p))
    if cfg.local_gamma_p > 0:
        transforms.append(RandomLocalGamma(p=cfg.local_gamma_p))
    if cfg.brightness_p > 0:
        transforms.append(RandomBrightness(sigma=cfg.brightness_sigma, p=cfg.brightness_p))
    if cfg.contrast_p > 0:
        transforms.append(
            RandomContrast(
                contrast_range=(cfg.contrast_min, cfg.contrast_max),
                preserve_range=cfg.contrast_preserve_range,
                p=cfg.contrast_p,
            )
        )

    # --- TorchIO stand-ins for reference augmentations we did not port ---
    if cfg.anisotropy_p > 0:
        transforms.append(tio.RandomAnisotropy(p=cfg.anisotropy_p))
    if cfg.noise_p > 0:
        transforms.append(tio.RandomNoise(std=(0.0, cfg.noise_std), p=cfg.noise_p))
    if cfg.gamma_p > 0:
        transforms.append(tio.RandomGamma(p=cfg.gamma_p))
    if cfg.bias_field_p > 0:
        transforms.append(tio.RandomBiasField(p=cfg.bias_field_p))

    return tio.Compose(transforms)


class VolumeDatsetIO(Dataset):
    def __init__(self, filepath_tuples: list, transform: Optional[tio.Compose] = None,
                 pipeline_version: int = 1, metas: Optional[Dict[str, Any]] = None,
                 n_classes: Optional[int] = None, zoom_aug_max: float = 1.0, band_max: float = 32.0,
                 chunk_size: int = 64):
        """Pipeline v1: ``filepath_tuples`` are ``(volume, label)`` paths (unchanged).

        Pipeline v2: ``(volume, target, chunk_dir)`` where ``target`` is a points file (``.npy``,
        CLAUDE.md Section 5.2) or a dense label chunk, and ``metas[chunk_dir]`` is the
        directory's :class:`~tomocpt.dataManager.metadata.ChunkMeta`. The target is drawn in
        the worker with ``render_targets(points, (64, 64, 64), n_classes, "clip")`` after the
        optional zoom augmentation (``zoom_aug_max > 1``, training only) and before
        ``transform``; it stays a ``tio.LabelMap``.
        """
        self.filepath_tuples = filepath_tuples
        self.transform = transform
        self.pipeline_version = int(pipeline_version)
        self.metas = metas or {}
        self.n_classes = n_classes
        self.zoom_aug_max = float(zoom_aug_max)
        self.band_max = float(band_max)
        self.chunk_size = int(chunk_size)

    def __len__(self):
        return len(self.filepath_tuples)

    def __getitem__(self, index: int) -> dict:
        if self.pipeline_version == 2:
            return self._getitem_v2(index)
        """
        Generates one sample of data.
        1. Gets file paths for the given index.
        2. Creates a torchio.Subject to load and hold the data.
        3. Applies TorchIO transforms to the Subject.
        4. Extracts the underlying torch.Tensor from each image.
        5. Returns a simple dictionary of tensors, which the default DataLoader can collate.
        """
        vol_path, label_path = self.filepath_tuples[index]
        subject = tio.Subject(
            input_data=tio.ScalarImage(vol_path),
            target_data=tio.LabelMap(label_path)
        )

        if self.transform is not None:
            subject = self.transform(subject)

        # Return a dictionary of tensors, not a Subject object.
        # This allows PyTorch's default collate_fn to work correctly.
        return {
            'input_data': subject['input_data'].data,
            'target_data': subject['target_data'].data,
        }

    # ---------------------------------------------------------------- pipeline v2
    def _load_v2(self, index: int):
        vol_path, tgt_path, chunk_dir = self.filepath_tuples[index]
        meta = self.metas[chunk_dir]
        vol = tio.ScalarImage(vol_path).data.float()
        if str(tgt_path).endswith(".npy"):
            points = np.load(tgt_path).astype(np.float32).reshape(-1, 5)
            dense = None
        else:
            points = None
            dense = tio.ScalarImage(tgt_path).data.float()
        return vol, points, dense, meta

    def _zoom(self, vol: torch.Tensor, points: Optional[np.ndarray], dense: Optional[torch.Tensor], s: float):
        """Zoom by ``s`` (trilinear) and crop a random cube: output voxel ``j`` samples input
        position ``(j + o) / s`` (voxel 0 at the origin, as the points). Points map to
        ``p * s - o`` and their ``d_vox`` to ``d_vox * s``."""
        n = vol.shape[-1]
        max_off = (n - 1) * (s - 1.0)
        off = torch.rand(3) * max_off  # (z, y, x) offsets in zoomed voxels
        j = torch.arange(n, dtype=torch.float64)
        axes = [((j + float(off[a])) / s) * (2.0 / (n - 1)) - 1.0 for a in range(3)]
        gz, gy, gx = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0).float()  # grid_sample wants (x, y, z)

        def _sample(t):
            return torch.nn.functional.grid_sample(t.unsqueeze(0), grid, mode="bilinear", padding_mode="border",
                                                   align_corners=True)[0]

        vol = _sample(vol)
        if dense is not None:
            dense = _sample(dense)
        if points is not None and len(points):
            points = points.astype(np.float64).copy()
            points[:, :3] = points[:, :3] * s - off.double().numpy()[None, :]
            points[:, 4] = points[:, 4] * s
            points = points.astype(np.float32)
        return vol, points, dense

    def _getitem_v2(self, index: int) -> dict:
        vol, points, dense, meta = self._load_v2(index)
        n_classes = int(self.n_classes if self.n_classes is not None else len(meta.class_available))
        available = torch.tensor([bool(a) for a in meta.class_available], dtype=torch.bool)
        d_vox = torch.tensor(meta.class_d_vox, dtype=torch.float64)
        apix = float(meta.apix_work)

        if self.zoom_aug_max > 1.0:
            s = 1.0 + float(torch.rand(1)) * (self.zoom_aug_max - 1.0)
            if s > 1.0:
                vol, points, dense = self._zoom(vol, points, dense, s)
                d_vox = d_vox * s
                apix = apix / s
                # Zoom only enlarges: a class pushed above the band becomes unavailable.
                available = available & (d_vox <= self.band_max * (1.0 + 1e-6))

        if points is not None:
            from tomocpt.labels.helpers import render_targets
            size = tuple(vol.shape[-3:])
            target = torch.from_numpy(render_targets(points, size, n_classes, border_policy="clip"))
        else:
            target = dense

        subject = tio.Subject(input_data=tio.ScalarImage(tensor=vol), target_data=tio.LabelMap(tensor=target))
        if self.transform is not None:
            subject = self.transform(subject)
        return {
            "input_data": subject["input_data"].data,
            "target_data": subject["target_data"].data,
            "class_available": available,
            "class_d_vox": d_vox.float(),
            "apix_work": torch.tensor([apix], dtype=torch.float32),
            "dataset_id": torch.tensor(int(meta.dataset_id), dtype=torch.int64),
        }

    @staticmethod
    def _get_filepaths_v2(data_dir: str, target_storage: str = "points") -> tuple:
        """``(volume, target, chunk_dir)`` tuples and ``{chunk_dir: ChunkMeta}`` of one split."""
        from tomocpt.dataManager.metadata import CHUNK_META_FNAME, read_chunk_meta

        tuples, metas = [], {}
        root = Path(data_dir)
        if not root.is_dir():
            return tuples, metas
        labels_dirname = constants.LABELS_DIR_NAME_PREFIX % "_supervised"
        for chunk_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p / CHUNK_META_FNAME).is_file()):
            metas[str(chunk_dir)] = read_chunk_meta(chunk_dir)
            vol_root = chunk_dir / constants.VOLUMES_DIR_NAME_PREFIX
            if not vol_root.is_dir():
                continue
            for vol_path in sorted(vol_root.glob("*" + constants.CUBES_EXTENSION)):
                stem = vol_path.name[: -len(constants.CUBES_EXTENSION)]
                suffix = stem[len(constants.VOLUMES_DIR_NAME_PREFIX):]
                if target_storage == "dense":
                    tgt = chunk_dir / labels_dirname / f"{labels_dirname}{suffix}{constants.CUBES_EXTENSION}"
                else:
                    tgt = chunk_dir / "points" / f"points{suffix}.npy"
                if tgt.is_file():
                    tuples.append((str(vol_path), str(tgt), str(chunk_dir)))
        return tuples, metas

    def chunk_classes(self) -> List[Set[int]]:
        """Per chunk (pipeline v2): the classes that are available and have at least one point
        centred inside the chunk (dense targets: channel maximum > 0.5)."""
        out = []
        for vol_path, tgt_path, chunk_dir in self.filepath_tuples:
            meta = self.metas[chunk_dir]
            avail = [bool(a) for a in meta.class_available]
            if str(tgt_path).endswith(".npy"):
                pts = np.load(tgt_path).reshape(-1, 5)
                inside = np.all((pts[:, :3] >= 0) & (pts[:, :3] < self.chunk_size), axis=1)
                classes = {int(c) for c in pts[inside, 3]}
            else:
                lab = tio.ScalarImage(tgt_path).data
                classes = {c for c in range(lab.shape[0]) if float(lab[c].max()) > 0.5}
            out.append({c for c in classes if 0 <= c < len(avail) and avail[c]})
        return out

    @staticmethod
    def _get_filepaths(data_dir: str, return_labels: bool) -> list:
        names_list = []
        labels_dirname = constants.LABELS_DIR_NAME_PREFIX % ('_supervised' if return_labels else '_selfSup')
        for root, _, files in os.walk(data_dir):
            vol_root = Path(root)
            if vol_root.name != constants.VOLUMES_DIR_NAME_PREFIX: continue
            label_root = vol_root.parent / labels_dirname
            if not label_root.exists(): continue
            for fname in files:
                if fname.endswith(constants.CUBES_EXTENSION):
                    vol_path = vol_root / fname
                    label_fname = fname.replace(constants.VOLUMES_DIR_NAME_PREFIX, labels_dirname)
                    label_path = label_root / label_fname
                    if vol_path.is_file() and label_path.is_file():
                        names_list.append((str(vol_path), str(label_path)))
        return names_list

    @classmethod
    def get_dataset(cls, data_dir: str, is_training: bool, return_labels: bool,
                    augmentation: Optional[AugmentationConfig] = None):
        """Build a dataset for one split.

        ``augmentation`` is passed explicitly rather than read from the global
        ``mainConfig``: this is a classmethod called from more than one place,
        and implicit global state would make it untestable. Validation keeps
        ``transform=None``.
        """
        transform = build_training_transforms(augmentation) if is_training else None

        list_of_filepaths = cls._get_filepaths(data_dir, return_labels)
        if not list_of_filepaths: raise FileNotFoundError(f"No data in {data_dir}")
        
        return cls(list_of_filepaths, transform=transform)

    @classmethod
    def get_dataset_v2(cls, data_dir: str, is_training: bool, n_classes: int,
                       augmentation: Optional[AugmentationConfig] = None,
                       scale_cfg: Optional[ScaleConfig] = None, target_storage: str = "points",
                       chunk_size: int = 64):
        """Pipeline v2 dataset of one split (zoom augmentation and transforms for training only)."""
        scale_cfg = scale_cfg if scale_cfg is not None else ScaleConfig()
        transform = build_training_transforms(augmentation) if is_training else None
        tuples, metas = cls._get_filepaths_v2(data_dir, target_storage)
        if not tuples:
            raise FileNotFoundError(f"No pipeline v2 chunks in {data_dir}")
        return cls(tuples, transform=transform, pipeline_version=2, metas=metas, n_classes=n_classes,
                   zoom_aug_max=float(scale_cfg.zoom_aug_max) if is_training else 1.0,
                   band_max=float(scale_cfg.band_max_vox), chunk_size=chunk_size)


def repeat_factor_weights(chunk_classes: Sequence[Set[int]], n_classes: int, threshold: float) -> np.ndarray:
    """Repeat factors (Gupta et al., LVIS 2019) per chunk.

    ``f_c`` = fraction of chunks whose class set contains ``c``; ``r_c = max(1, sqrt(t / f_c))``;
    chunk factor ``r = max(r_c)`` over its classes, 1 for a chunk without classes.
    """
    n = len(chunk_classes)
    r_c = _class_repeat_factors(chunk_classes, n_classes, threshold)[1]
    r = np.ones(n, dtype=np.float64)
    for i, classes in enumerate(chunk_classes):
        vals = [r_c[c] for c in classes if 0 <= int(c) < n_classes]
        if vals:
            r[i] = max(vals)
    return r


def _class_repeat_factors(chunk_classes: Sequence[Set[int]], n_classes: int, threshold: float):
    n = max(1, len(chunk_classes))
    counts = np.zeros(int(n_classes), dtype=np.float64)
    for classes in chunk_classes:
        for c in classes:
            if 0 <= int(c) < n_classes:
                counts[int(c)] += 1
    f = counts / n
    r_c = np.ones(int(n_classes), dtype=np.float64)
    nz = f > 0
    r_c[nz] = np.maximum(1.0, np.sqrt(float(threshold) / f[nz]))
    return f, r_c


class Data(pl.LightningDataModule):
    def __init__(self, data_dir, return_labels, batch_size, workers_for_data,
                 augmentation: Optional[AugmentationConfig] = None,
                 pipeline_version: int = 1, scale_cfg: Optional[ScaleConfig] = None,
                 chunk_sampling: ChunkSampling = ChunkSampling.uniform,
                 repeat_factor_threshold: float = 0.1, seed: Optional[int] = None):
        """``pipeline_version``, ``scale_cfg``, ``chunk_sampling`` and
        ``repeat_factor_threshold`` come from the train config (``training/run.py``); their
        defaults give the legacy (pipeline v1) data module. ``seed`` seeds the repeat-factor
        sampler (default: ``torch.initial_seed()``)."""
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation", "pipeline_version", "scale_cfg", "chunk_sampling",
                                          "repeat_factor_threshold", "seed"])
        self.augmentation = augmentation
        self.pipeline_version = int(pipeline_version)
        self.scale_cfg = scale_cfg if scale_cfg is not None else ScaleConfig()
        self.chunk_sampling = ChunkSampling(_enum_value(chunk_sampling))
        self.repeat_factor_threshold = float(repeat_factor_threshold)
        self.seed = seed
        self.class_table = None
        self.sample_weights = None
        if self.pipeline_version not in (1, 2):
            raise ValueError(f"Unknown data pipeline version {pipeline_version} (expected 1 or 2)")
        if self.chunk_sampling == ChunkSampling.repeat_factor and self.pipeline_version != 2:
            raise ValueError("chunk_sampling='repeat_factor' needs data_pipeline_version=2 "
                             "(pipeline v1 chunks have no per-class information)")

    def setup(self, stage: Optional[str] = None):
        from tomocpt.dataManager.metadata import layout_pipeline_version, read_chunk_layout

        rank_zero_info(f"Loading data from {self.hparams.data_dir}")
        layout = read_chunk_layout(self.hparams.data_dir)
        found = layout_pipeline_version(layout)
        if found != self.pipeline_version:
            raise ValueError(
                f"chunk format mismatch: {self.hparams.data_dir} holds pipeline v{found} chunks but "
                f"data_pipeline_version={self.pipeline_version} was requested. Rebuild the chunks "
                "(delete the done file in chunks_dir) or change train.data_pipeline_version.")
        if self.pipeline_version == 2:
            self._setup_v2(layout)
            return
        self.dataset_training = VolumeDatsetIO.get_dataset(
            self._get_split_dir('train'), True, self.hparams.return_labels,
            augmentation=self.augmentation,
        )
        self.dataset_val = VolumeDatsetIO.get_dataset(
            self._get_split_dir('val'), False, self.hparams.return_labels,
        )

    def _setup_v2(self, layout: dict):
        from tomocpt.labels.multiclass import class_table_from_json

        if not self.hparams.return_labels:
            raise ValueError("data_pipeline_version=2 supports picking (labelled) chunks only")
        self.class_table = class_table_from_json(layout["class_table"])
        n_classes = len(self.class_table)
        storage = str(layout.get("target_storage", "points"))
        chunk_size = int(layout.get("chunk_size", 64))
        common = dict(n_classes=n_classes, scale_cfg=self.scale_cfg, target_storage=storage, chunk_size=chunk_size)
        self.dataset_training = VolumeDatsetIO.get_dataset_v2(
            self._get_split_dir('train'), True, augmentation=self.augmentation, **common)
        self.dataset_val = VolumeDatsetIO.get_dataset_v2(self._get_split_dir('val'), False, **common)
        if self.chunk_sampling == ChunkSampling.repeat_factor:
            sets = self.dataset_training.chunk_classes()
            f, r_c = _class_repeat_factors(sets, n_classes, self.repeat_factor_threshold)
            self.sample_weights = repeat_factor_weights(sets, n_classes, self.repeat_factor_threshold)
            for c, row in enumerate(self.class_table):
                rank_zero_info(f"Repeat-factor sampling: class {row.name} (#{row.number}) "
                               f"f_c = {f[c]:.4f}, r_c = {r_c[c]:.3f}")
            rank_zero_info(f"Repeat-factor sampling: {len(sets)} training chunks, "
                           f"{int(round(self.sample_weights.sum()))} draws per epoch")

    def _get_split_dir(self, split): 
        return os.path.join(self.hparams.data_dir, split)

    def _persistent_workers(self) -> bool:
        # Legacy rule (reads the global config, as before Gate 2).
        return self.hparams.workers_for_data > 0 and mainConfig.train.use_gpus

    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers_for_data, shuffle=shuffle,
                          persistent_workers=self._persistent_workers())

    def train_dataloader(self): 
        if self.chunk_sampling == ChunkSampling.repeat_factor:
            weights = self.sample_weights
            generator = torch.Generator()
            generator.manual_seed(int(self.seed) if self.seed is not None else int(torch.initial_seed()) % (2 ** 63))
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                            num_samples=int(round(float(np.sum(weights)))), replacement=True,
                                            generator=generator)
            return DataLoader(self.dataset_training, batch_size=self.hparams.batch_size,
                              num_workers=self.hparams.workers_for_data, sampler=sampler,
                              persistent_workers=self._persistent_workers())
        return self._get_dataloader(self.dataset_training, True)

    def val_dataloader(self): 
        return self._get_dataloader(self.dataset_val, False)
