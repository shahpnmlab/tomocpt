import os
import os.path as osp
import shutil
import subprocess
import sys
from enum import Enum
from typing import Optional, Literal
import psutil
from tomocpt import config
from tomocpt.defaultConfigs import network_config

import torch

torch.set_float32_matmul_precision(network_config.TORCH_MATMUL_PRECISION)  # TODO: This should be config

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, LearningRateMonitor, \
    StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from tomocpt.dataManager.dataLoaderLightning import Data
from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
from tomocpt.utils import accelerator_selector

class ModelTypes(Enum):
    UNET = "UNET"
    unet = "unet"
    SwinUNETR = "SwinUNETR"
    swinunetr = "swinunetr"

class TrainingModes(Enum):
    selfSupervised = "selfSupervised"
    picking = "picking"

def train(chunks_dir: Optional[str]= None,
          model_dir: Optional[str] = None,
          experimentName: Optional[str] = None,
          epochs: Optional[str] = None,
          model_type: ModelTypes | None = None, # TODO: Define the available models in constants
          trainingMode: TrainingModes = "picking",
          learning_rate: Optional[float] = None,
          continueModelDir: Optional[str] = None,
          restoreFullStateWhenContinue: bool = True,
          compileModel: bool = False,
          batch_size: Optional[int] = None,
          use_cuda: bool = True,
          use_tensorboard: bool = True):
    """

    :param chunks_dir: the directory with the ready to train chunks.
    :param model_dir: the directory where the models will be saved
    :param experimentName: dir name to store checkpoints in
    :param epochs: number of epochs to train
    :param model_type: The model type name to be selected. Do not change it if using a previous chekpoint.
    :param trainingMode: Either for training a selfSupervised or supervised network
    :param learning_rate: learning_rate
    :param continueModelDir: The directory of a previously trained model to continue its training
    :param restoreFullStateWhenContinue: If True, the optimizer state, Learning rate, etc. from the last checkpoint will be restored. Otherwise, only the weights
    :param compileModel: If True, use pytorch 2.X compile
    :param batch_size: The batch size
    :param use_cuda: True to use cuda (if available)
    :param use_tensorboard: True to launch tensorboard
    :return:
    """

    chunks_dir = chunks_dir if chunks_dir is not None else config.chunks_dir
    model_dir = model_dir if model_dir is not None else config.model_dir
    experimentName = experimentName if experimentName is not None else config.EXPERIMENT_NAME
    epochs = epochs if epochs is not None else config.N_EPOCHS
    model_type = model_type if model_type is not None else network_config.MODEL_TYPE
    learning_rate = learning_rate if learning_rate is not None else network_config.LEARNING_RATE
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE

    config.MODEL_TYPE = model_type
    kwargs = dict(lr=learning_rate)

    if trainingMode == "picking":
        Model = BasePickingModel
        checkpointer = ModelCheckpoint(monitor='val_loss', filename='weights', verbose=True)
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            EarlyStopping(monitor='val_loss', patience=2 * network_config.COSINE_LR_SCHEDULE_N_EPOCHS, verbose=True),
            checkpointer,
            LearningRateMonitor(logging_interval='epoch'),
        ]
    # This will be consolidated to pre-train and supervised train
    elif trainingMode == "selfSupervised":
        Model = SelfSupervisedModel
        checkpointer = ModelCheckpoint(monitor='val_loss', filename=f'weights_{trainingMode}_{model_type}', verbose=True)
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            EarlyStopping(monitor='val_loss', patience=2 * network_config.COSINE_LR_SCHEDULE_N_EPOCHS, verbose=True),
            checkpointer,
            LearningRateMonitor(logging_interval='epoch'),
        ]
    else:
        raise ValueError("Error, trainingMode (%s) not valid" % trainingMode)

    if continueModelDir:
        resume_from_checkpoint = continueModelDir if restoreFullStateWhenContinue else None

        if trainingMode == "picking":
            try:
                pl_model = BasePickingModel.load_from_checkpoint(continueModelDir, **kwargs)
            except RuntimeError:
                pretrained_model = SelfSupervisedModel.load_from_checkpoint(continueModelDir)
                pl_model = BasePickingModel(**kwargs, model=pretrained_model)  # map_location="cuda:0"
                del pretrained_model
                resume_from_checkpoint = None
        else:
            pl_model = Model.load_from_checkpoint(continueModelDir, **kwargs)
    else:
        pl_model = Model(**kwargs)
        resume_from_checkpoint = None
    if compileModel:
        pl_model = torch.compile(pl_model)
    callbacks += [
        StochasticWeightAveraging(annealing_epochs=network_config.COSINE_LR_SCHEDULE_N_EPOCHS, swa_lrs=0.1 * pl_model.lr)]

    assert os.path.isdir(chunks_dir), f"Error, prepared_data_dir: {chunks_dir} does not exist "
    data = Data(data_dir=chunks_dir, return_labels=(trainingMode == "picking"),
                batch_size=batch_size,
                workers_for_data=config.WORKERS_FOR_DATA)
    data.setup()
    print(len(data.train_dataloader()))

    logger = TensorBoardLogger(save_dir=f'{model_dir}/{experimentName}', name='', version='')

    accel, dev_count = accelerator_selector(use_cuda=use_cuda)
    trainer = pl.Trainer(default_root_dir=model_dir, devices=f'{dev_count}', accelerator='auto',
                         max_epochs=epochs, callbacks=callbacks,
                         logger=logger,
                         overfit_batches= config.OVERFIT_N_BATCHES,
                         # limit_val_batches=constants.LIMIT_VALIDATION_BATCHES,
                         # val_check_interval=constants.VAL_CHECK_INTERVAL,
                         strategy="ddp_find_unused_parameters_false" if accel == "gpu" else None,
                         precision=network_config.TORCH_FLOAT_PRECISION)
    if trainer.is_global_zero:
        _copyCodeForReproducibility(trainer.log_dir)

    if use_tensorboard:
        subprocess.Popen(["tensorboard", "--logdir", model_dir], stdout=sys.stdout, stderr=open("/dev/null"))
        print("Use the url below to monitor training on tensorboard")
    print("Training starts")

    trainer.fit(pl_model, datamodule=data, ckpt_path=resume_from_checkpoint)

    # if trainer.is_global_zero:
    #     model = torch.load(checkpointer.best_model_path, weights_only=False)
    #     model_scripted = torch.jit.script(model)  # Export to TorchScript
    #     model_scripted.save(checkpointer.best_model_path+".scripted.pt")


def _copyCodeForReproducibility(logdir):
    """
    Copy the code to the logdir so that reproducibility is ensured

    Args:
        logdir: The directory were the code will be saved

    Returns:

    """
    copycodedir = osp.join(logdir, "code")
    os.makedirs(copycodedir, exist_ok=True)
    copycodedir = osp.join(copycodedir, "tomocpt")

    modulePath = osp.abspath(sys.modules[__name__].__file__)
    rootPath = osp.dirname(osp.dirname(modulePath))

    for root, dirs, files in os.walk(rootPath):
        # Iterate through all folders
        for directory in dirs:
            # Create the corresponding directory in the target path
            source_folder = osp.join(root, directory)
            target_folder = source_folder.replace(rootPath, copycodedir)
            os.makedirs(target_folder, exist_ok=True)
        # Iterate through all Python files
        for file in files:
            if file.endswith(".py") or file.endswith(".yaml"):
                # Copy the Python file to the corresponding directory in the target path
                source_file = os.path.join(root, file)
                target_file = source_file.replace(rootPath, copycodedir)
                shutil.copy2(source_file, target_file)

    fname = osp.join(logdir, "command.txt")
    with open(fname, "w") as f:
        f.write(" ".join(sys.argv))

    current_process = psutil.Process()
    # Get the parent process
    parent_process = current_process.parent()
    # Get the command line of the parent process
    parent_command = " ".join(["'" + x + "'" if x.startswith('{"') else x for x in parent_process.cmdline()])
    fname = osp.join(logdir, "parent_command.txt")
    with open(fname, "w") as f:
        f.write(parent_command)


if __name__ == '__main__':
    from argParseFromDoc import AutoArgumentParser

    parser = AutoArgumentParser()
    parser.add_args_from_function(train)
    train(**vars(parser.parse_args()))
    print(f"Training has ended. Cleaning up {config.chunks_dir}")
    shutil.rmtree(config.chunks_dir, ignore_errors=True)

"""

python -m tomocpt.training.train 

"""
