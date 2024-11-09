import os
from pathlib import Path

import torch


def makedir(path: str):
    dirname = Path(path)
    if not dirname.is_dir():
        dirname.mkdir(parents=True)


def accelerator_selector(use_cuda=None):
    from pycotool import config
    if use_cuda is None:
        use_cuda = config.USE_CUDA
    if not use_cuda or not torch.cuda.is_available():
        accel = 'cpu'
        if config.NCPU is None:
            dev_count = os.cpu_count()
        else:
            dev_count = max(1, config.NCPU)
    else:
        accel = 'gpu'
        dev_count = torch.cuda.device_count()
    return accel, dev_count
