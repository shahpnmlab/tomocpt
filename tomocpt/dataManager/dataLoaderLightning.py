from typing import Optional, Dict, Any
import os
import torchio as tio
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tomocpt.mainConfig import mainConfig
from tomocpt.dataManager.datasetIO import VolumeDatsetIO

config = mainConfig.train


class Data(pl.LightningDataModule):
    def set_default_args(self, data_dir: str | None,
                         return_labels: bool | None,
                         batch_size: int | None,
                         workers_for_data: int | None):
        self.base_data_dir = data_dir if data_dir is not None else config.chunks_dir
        self.batch_size = batch_size if batch_size is not None else config.batch_size
        self.workers_for_data = workers_for_data if workers_for_data is not None else config.WORKERS_FOR_DATA
        self.return_labels = return_labels if return_labels is not None else False

    def __init__(self, data_dir: str | None = None, return_labels: bool | None = None,
                 batch_size: int | None = None, workers_for_data: int | None = None):
        super().__init__()
        self.set_default_args(data_dir=data_dir, return_labels=return_labels,
                              batch_size=batch_size, workers_for_data=workers_for_data)
        self.dataset_training = None
        self.dataset_val = None

    def _get_split_dir(self, split: str) -> str:
        """Get the directory for a specific data split."""
        return os.path.join(self.base_data_dir, split)

    def setup(self, stage: Optional[str] = None):
        train_dir = self._get_split_dir('train')
        val_dir = self._get_split_dir('val')

        print(f"Loading training data from: {train_dir}")
        print(f"Loading validation data from: {val_dir}")

        self.dataset_training = VolumeDatsetIO.get_dataset(
            data_dir=train_dir,
            isTraining=True,
            return_labels=self.return_labels
        )

        self.dataset_val = VolumeDatsetIO.get_dataset(
            data_dir=val_dir,
            isTraining=False,
            return_labels=self.return_labels
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        batch["input_data"] = batch["input_data"].data.to(device)
        batch["target_data"] = batch["target_data"].data.to(device)
        return batch

    def train_dataloader(self) -> DataLoader[Dict[str, tio.ScalarImage]]:
        return DataLoader(
            self.dataset_training,
            self.batch_size,
            num_workers=self.workers_for_data,
            shuffle=True,
            persistent_workers=True if (config.N_GPUS > 0 and
                                        config.use_cuda > 0 and
                                        self.workers_for_data > 0) else False
        )

    def val_dataloader(self) -> DataLoader[Dict[str, tio.ScalarImage]]:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.workers_for_data,
            shuffle=False,
            persistent_workers=True if (config.N_GPUS > 0 and
                                        config.use_cuda > 0 and
                                        self.workers_for_data > 0) else False
        )

    def test_dataloader(self):
        return NotImplemented

    def predict_dataloader(self):
        return NotImplemented


def _test():
    data = Data()
    data.setup()
    print("Testing training dataloader:")
    for batch in data.train_dataloader():
        print(f"Batch keys: {batch.keys()}")
        break  # Just test one batch


if __name__ == "__main__":
    _test()