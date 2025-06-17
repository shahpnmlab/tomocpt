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
from tomocpt.mainConfig import mainConfig
from tomocpt.logger import get_logger

logging = get_logger()

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
    def get_dataset(cls, data_dir: str, is_training: bool, return_labels: bool):

        transform = tio.Compose([
            tio.RandomAffine(degrees=45, default_pad_value="otsu", p=0.8),
            tio.OneOf({tio.RandomElasticDeformation(): 0.1,
                       tio.RandomBlur(std=1): 0.1}, p=0.75),
        ]) if is_training else None
        
        list_of_filepaths = cls._get_filepaths(data_dir, return_labels)
        if not list_of_filepaths: raise FileNotFoundError(f"No data in {data_dir}")
        
        return cls(list_of_filepaths, transform=transform)


class Data(pl.LightningDataModule):
    def __init__(self, data_dir, return_labels, batch_size, workers_for_data):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        rank_zero_info(f"Loading data from {self.hparams.data_dir}")
        self.dataset_training = VolumeDatsetIO.get_dataset(self._get_split_dir('train'), True, self.hparams.return_labels)
        self.dataset_val = VolumeDatsetIO.get_dataset(self._get_split_dir('val'), False, self.hparams.return_labels)

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
