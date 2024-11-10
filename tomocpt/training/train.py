import os
import os.path as osp
import shutil
import subprocess
import sys
from typing import Annotated
import psutil
import typer
from omegaconf import DictConfig

from tomocpt.defaultConfigs.train_config import TrainingModes
from tomocpt.mainConfig import mainConfig

import torch

torch.set_float32_matmul_precision(mainConfig.network.TORCH_MATMUL_PRECISION)  # TODO: This should be config

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, LearningRateMonitor, \
    StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from tomocpt.dataManager.dataLoaderLightning import Data
from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
from tomocpt.utils import accelerator_selector

network_config = mainConfig.network
train_config = mainConfig.train
def train(train_experiment_name: Annotated[str, typer.Option(help="The name name")] = None,
          train_chunks_dir: Annotated[str, typer.Option(help="Where the data prepared for training is name")] = None,
          train_model_dir: Annotated[str, typer.Option(help="Model name")] = None,
          train_n_epochs: Annotated[int, typer.Option(help="Model name")] = None,
          train_mode: Annotated[TrainingModes, typer.Option(help="Executing mode, either picking or "
                                                                   "selfSupervised name")] = None,
          train_learning_rate: Annotated[float, typer.Option(help="Model name")] = None,
          continueModelDir: Annotated[str, typer.Option(help="Model name")] = None,
          restoreFullStateWhenContinue: Annotated[bool, typer.Option(help="Model name")] = True,
          compileModel: Annotated[bool, typer.Option(help="Model name")] = False,
          batch_size: Annotated[int, typer.Option(help="Model name")] = None,
          use_cuda: Annotated[bool, typer.Option(help="Model name")] = True,
          use_tensorboard: Annotated[bool, typer.Option(help="Model name")] = True,
          config: DictConfig = None
          ):
    kwargs = dict(lr=train_learning_rate)
    print(mainConfig)
    breakpoint()
    if train_mode == TrainingModes.picking:
        Model = BasePickingModel
        checkpointer = ModelCheckpoint(monitor='val_loss', filename='weights', verbose=True)
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            EarlyStopping(monitor='val_loss', patience=2 * network_config.COSINE_LR_SCHEDULE_N_EPOCHS, verbose=True),
            checkpointer,
            LearningRateMonitor(logging_interval='epoch'),
        ]
    # This will be consolidated to pre-train and supervised train
    elif train_mode == TrainingModes.selfSupervised:
        Model = SelfSupervisedModel
        checkpointer = ModelCheckpoint(monitor='val_loss',
                                       filename=f'weights_{train_mode}_{network_config.model_type}',
                                       verbose=True)
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            EarlyStopping(monitor='val_loss', patience=2 * network_config.COSINE_LR_SCHEDULE_N_EPOCHS, verbose=True),
            checkpointer,
            LearningRateMonitor(logging_interval='epoch'),
        ]
    else:
        raise ValueError("Error, mode (%s) not valid" % train_mode)

    if continueModelDir:
        resume_from_checkpoint = continueModelDir if restoreFullStateWhenContinue else None

        if train_mode == TrainingModes.picking:
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

    assert os.path.isdir(train_chunks_dir), f"Error, prepared_data_dir: {train_chunks_dir} does not exist "
    data = Data(data_dir=train_chunks_dir, return_labels=(train_mode == TrainingModes.picking),
                batch_size=batch_size,
                workers_for_data=train_config.WORKERS_FOR_DATA)
    data.setup()
    print(len(data.train_dataloader()))

    logger = TensorBoardLogger(save_dir=f'{train_model_dir}/{train_experiment_name}', name='', version='')

    accel, dev_count = accelerator_selector(use_cuda=use_cuda)
    trainer = pl.Trainer(default_root_dir=train_model_dir, devices=f'{dev_count}', accelerator='auto',
                         max_epochs=train_n_epochs, callbacks=callbacks,
                         logger=logger,
                         overfit_batches= train_config.OVERFIT_N_BATCHES,
                         # limit_val_batches=constants.LIMIT_VALIDATION_BATCHES,
                         # val_check_interval=constants.VAL_CHECK_INTERVAL,
                         strategy="ddp_find_unused_parameters_false" if accel == "gpu" else None,
                         precision=network_config.TORCH_FLOAT_PRECISION)
    if trainer.is_global_zero:
        _copyCodeForReproducibility(trainer.log_dir)

    if use_tensorboard:
        subprocess.Popen(["tensorboard", "--logdir", train_model_dir], stdout=sys.stdout, stderr=open("/dev/null"))
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

