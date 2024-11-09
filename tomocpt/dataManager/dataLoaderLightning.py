from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tomocpt import config
from tomocpt.dataManager.datasetIO import VolumeDatsetIO


class Data(pl.LightningDataModule):

    def set_default_args(self, data_dir: str | None, return_labels: bool | None,
                         batch_size: int | None, workers_for_data: int | None):
        self.data_dir = data_dir if data_dir is not None else config.DATA_CHUNKS_DIR
        self.batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
        self.workers_for_data = workers_for_data if workers_for_data is not None else config.WORKERS_FOR_DATA
        self.return_labels = return_labels if return_labels is not None else False

    def __init__(self, data_dir: str | None = None, return_labels: bool | None = None,
                 batch_size: int | None = None, workers_for_data: int | None = None):
        super().__init__()
        self.set_default_args(data_dir=data_dir, return_labels=return_labels,
                              batch_size=batch_size, workers_for_data=workers_for_data)

    def setup(self, stage: Optional[str] = None):
        self.dataset_training = VolumeDatsetIO.get_dataset(data_dir=self.data_dir, isTraining=True,
                                                           return_labels=self.return_labels)
        self.dataset_val = VolumeDatsetIO.get_dataset(data_dir=self.data_dir, isTraining=False,
                                                      return_labels=self.return_labels)

    def train_dataloader(self):
        return DataLoader(self.dataset_training, self.batch_size, num_workers=self.workers_for_data, shuffle=True,
                          persistent_workers=True if config.N_GPUS > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.workers_for_data,
                          shuffle=False, persistent_workers=True if config.N_GPUS > 0 else False)

    def test_dataloader(self):
        return NotImplemented

    def predict_dataloader(self):
        return NotImplemented

def _test():
    data = Data()
    data.setup()
    for batch in data.train_dataloader():
        print(batch.keys())

if __name__ == "__main__":
    _test()