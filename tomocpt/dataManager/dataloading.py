import os
from typing import Optional, Any
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset
# No longer need to import is_main_process from utils, it's used inside the Lightning hook
from pytorch_lightning.utilities.rank_zero import rank_zero_info


from tomocpt import constants
from tomocpt.dataManager.fourier_augmentations import RandomFourierDegradation
from tomocpt.dataManager.intensity_augmentations import (
    RandomBrightness,
    RandomBrightnessGradient,
    RandomContrast,
    RandomLocalGamma,
)
from tomocpt.defaultConfigs.train_config import AugmentationConfig
from tomocpt.mainConfig import mainConfig
from tomocpt.logger import get_logger

logging = get_logger()

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
    Ordering matters and follows the reference pipeline:

    1. ``tio.RandomAffine`` - image and label rotated together.
    2. ``tio.OneOf({RandomElasticDeformation, RandomBlur})``.
    3. :class:`RandomFourierDegradation` - **after all spatial transforms**.
    4. Intensity transforms.

    Step 3's position is the entire point of the augmentation: running after the
    rotations makes the missing wedge land at a random orientation relative to
    the specimen, while the label stays geometrically correct. Moving it earlier
    would silently destroy the augmentation while still training normally.
    """
    if cfg is None:
        cfg = AugmentationConfig()

    transforms = [
        tio.RandomAffine(degrees=cfg.affine_degrees, default_pad_value="otsu", p=cfg.affine_p),
        tio.OneOf({tio.RandomElasticDeformation(): cfg.elastic_weight,
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
    def __init__(self, filepath_tuples: list, transform: Optional[tio.Compose] = None):
        self.filepath_tuples = filepath_tuples
        self.transform = transform

    def __len__(self):
        return len(self.filepath_tuples)

    def __getitem__(self, index: int) -> dict:
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


class Data(pl.LightningDataModule):
    def __init__(self, data_dir, return_labels, batch_size, workers_for_data,
                 augmentation: Optional[AugmentationConfig] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])
        self.augmentation = augmentation

    def setup(self, stage: Optional[str] = None):
        rank_zero_info(f"Loading data from {self.hparams.data_dir}")
        self.dataset_training = VolumeDatsetIO.get_dataset(
            self._get_split_dir('train'), True, self.hparams.return_labels,
            augmentation=self.augmentation,
        )
        self.dataset_val = VolumeDatsetIO.get_dataset(
            self._get_split_dir('val'), False, self.hparams.return_labels,
        )

    def _get_split_dir(self, split): 
        return os.path.join(self.hparams.data_dir, split)

    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers_for_data, shuffle=shuffle,
                          persistent_workers=(self.hparams.workers_for_data > 0 and mainConfig.train.use_gpus))

    def train_dataloader(self): 
        return self._get_dataloader(self.dataset_training, True)

    def val_dataloader(self): 
        return self._get_dataloader(self.dataset_val, False)
