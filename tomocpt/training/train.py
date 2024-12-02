import os
import os.path as osp
import shutil
import subprocess
import sys
import psutil
import typer
import torch

from omegaconf import DictConfig
from pathlib import Path
from typing import Annotated, Optional
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl




def train(
        compile_model: Annotated[bool, typer.Option(help="Path to pre-existing checkpoint file for fine-tuning with new data")] = None,
        continue_training: Annotated[Optional[Path], typer.Option(help="Path to pre-existing checkpoint file for fine-tuning with new data")] = None,
        launch_tensorboard: Annotated[bool, typer.Option(help="Launch tensorboard for evaluating training")] = True,
        config: DictConfig = None):

    from tomocpt.mainConfig import mainConfig
    torch.set_float32_matmul_precision(mainConfig.train.network.TORCH_MATMUL_PRECISION)
    from tomocpt.defaultConfigs.train_config import TrainingModes
    from tomocpt.dataManager.dataLoaderLightning import Data
    from tomocpt.networks.pickingModel import BasePickingModel
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
    from tomocpt.utils import accelerator_selector
    from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, LearningRateMonitor, \
        StochasticWeightAveraging  # TODO: Importing from pytorch_lightning.callbacks is launching a jit warning. Why?. Could it be version related

    kwargs = dict(lr=mainConfig.train.optimizer.lr)

    train__config = mainConfig.train
    network__config = mainConfig.train.network
    assert mainConfig.train.model_dir
    assert mainConfig.train.chunks_dir
    assert os.path.isdir(mainConfig.train.chunks_dir), f"Error, mainConfig.train.chunks_dir: {mainConfig.train.chunks_dir} does not exist "

    # print(mainConfig.train.learning_rate)
    # breakpoint()
    if mainConfig.train.mode == TrainingModes.picking:
        Model = BasePickingModel
        checkpointer = ModelCheckpoint(monitor='val_loss', filename='weights', verbose=True)
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            EarlyStopping(monitor='val_loss', patience=2 * train__config.COSINE_LR_SCHEDULE_N_EPOCHS, verbose=True),
            checkpointer,
            LearningRateMonitor(logging_interval='epoch'),
        ]
    # This will be consolidated to pre-train and supervised train
    elif mainConfig.train.mode == TrainingModes.selfSupervised:
        Model = SelfSupervisedModel
        checkpointer = ModelCheckpoint(monitor='val_loss',
                                       filename=f'weights_{mainConfig.train.mode}_{network__config.model_type}',
                                       verbose=True)
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            EarlyStopping(monitor='val_loss', patience=2 * train__config.COSINE_LR_SCHEDULE_N_EPOCHS, verbose=True),
            checkpointer,
            LearningRateMonitor(logging_interval='epoch'),
        ]
    else:
        raise ValueError("Error, mode (%s) not valid" % mainConfig.train.mode)

    if continue_training:
        resume_from_checkpoint = continue_training if mainConfig.train.restore_full_state else None

        if mainConfig.train.mode == TrainingModes.picking:
            try:
                pl_model = BasePickingModel.load_from_checkpoint(continue_training, **kwargs)
            except RuntimeError:
                pretrained_model = SelfSupervisedModel.load_from_checkpoint(continue_training)
                pl_model = BasePickingModel(**kwargs, model=pretrained_model)  # map_location="cuda:0"
                del pretrained_model
                resume_from_checkpoint = None
        else:
            pl_model = Model.load_from_checkpoint(continue_training, **kwargs)
    else:
        pl_model = Model(**kwargs)
        resume_from_checkpoint = None
    if compile_model:
        pl_model = torch.compile(pl_model)
    callbacks += [
        StochasticWeightAveraging(annealing_epochs=train__config.COSINE_LR_SCHEDULE_N_EPOCHS,
                                  swa_lrs=0.1 * pl_model.lr)]

    data = Data(data_dir=mainConfig.train.chunks_dir, return_labels=(mainConfig.train.mode == TrainingModes.picking),
                batch_size=mainConfig.train.batch_size,
                workers_for_data=mainConfig.train.WORKERS_FOR_DATA)
    data.setup()

    print(len(data.train_dataloader()))

    logger = TensorBoardLogger(save_dir=f'{mainConfig.train.model_dir}/{mainConfig.train.experiment_name}', name='', version='')

    accel, dev_count = accelerator_selector(use_cuda=mainConfig.train.use_cuda, n_cpus=mainConfig.train.N_CPUS_IF_NO_GPU)

    trainer = pl.Trainer(default_root_dir=mainConfig.train.model_dir, devices=f'{dev_count}',
                         accelerator='auto' if mainConfig.train.use_cuda else accel,
                         max_epochs=mainConfig.train.n_epochs, callbacks=callbacks,
                         logger=logger,
                         overfit_batches=mainConfig.train.OVERFIT_N_BATCHES if mainConfig.train.OVERFIT_N_BATCHES else 0,
                         # limit_val_batches=constants.LIMIT_VALIDATION_BATCHES,
                         # val_check_interval=constants.VAL_CHECK_INTERVAL,
                         strategy="ddp_find_unused_parameters_false" if accel == "gpu" else "auto",
                         precision=network__config.TORCH_FLOAT_PRECISION,
                         gradient_clip_val=1.0,
                         gradient_clip_algorithm='norm',
                         )

    if trainer.is_global_zero:
        _copyCodeForReproducibility(trainer.log_dir)

    if launch_tensorboard:
        subprocess.Popen(["tensorboard", "--logdir", f'{mainConfig.train.model_dir}/{mainConfig.train.experiment_name}'],
                         stdout=sys.stdout, stderr=open("/dev/null"))
        print("Use the url below to monitor training on tensorboard")
    print("Training starts")

    trainer.fit(pl_model, datamodule=data, ckpt_path=resume_from_checkpoint)

    # if trainer.is_global_zero: #TODO: monai models can be scripted. We need to try if they can be exported using dynamo or other backend
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
