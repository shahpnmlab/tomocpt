import glob
import os
from pathlib import Path

import pandas as pd
import torch


def makedir(path: str):
    dirname = Path(path)
    if not dirname.is_dir():
        dirname.mkdir(parents=True)


def accelerator_selector(use_cuda=None, n_cpus=None):
    assert use_cuda is not None
    if not use_cuda or not torch.cuda.is_available():
        accel = "cpu"
        if n_cpus is None:
            dev_count = os.cpu_count()
        else:
            dev_count = max(1, n_cpus)
    else:
        accel = "gpu"
        dev_count = torch.cuda.device_count()
    return accel, dev_count


def read_particles_csvs(dirname):
    fnames = glob.glob(os.path.join(dirname, "*", "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fnames])
    return df


def is_main_process():
    """Determine if this is the main process (rank 0)."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True
