import os
import os.path as osp
from pathlib import Path

import torch
from ruamel.yaml import YAML


def update_config(modelFname):
    if modelFname and osp.isfile(modelFname):
        modelDirname = osp.dirname(osp.dirname(modelFname))
        modelYmlFname = osp.join(modelDirname, "hparams.yaml")
        with open(modelYmlFname, "r") as f:
            d = dict(YAML().load(f))
            confDict = d["config_dict"]
            constantsDict = d["constants_dict"]
        from pycotool import config, constants
        config.update(confDict)
        for k,v in constantsDict.items():
            setattr(constants, k, v)

def mkdir_if_not_exists(dirname):
    try:
        os.mkdir(dirname)
    except OSError:
        pass

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
