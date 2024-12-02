import importlib
import logging

import hydra
import pytorch_lightning as pl
import torch

from tomocpt import constants
from tomocpt.configManager.configManager import update_config
from tomocpt.mainConfig import mainConfig

class BaseModel(pl.LightningModule):

    def set_default_args(self, lr: float | None, num_levels: int | None):
        self.lr = lr if lr is not None else self.hparams.config.train.optimizer.lr
        # self.num_levels = num_levels if num_levels is not None else self.hparams.config.train.network.NUM_LEVELS


    def __init__(self, lr: float | None = None,
                 num_levels: int | None = None,
                 config=None,
                 constants_dict=None):
        super(BaseModel, self).__init__()

        if config is None:
            config = mainConfig
        else:
            update_config(mainConfig.train, source=config.train) #We update all the training details acording to the checkpoint
        if constants_dict is None:
            constants_dict = {k: getattr(constants, k) for k in dir(constants) if not k.startswith("__")}
        else:
            for k, v in constants_dict.items():
                setattr(constants, k, v)

        self.save_hyperparameters(
            dict(constants_dict=constants_dict,
                 config=config
                 )
        )
        self.set_default_args(lr=lr, num_levels=num_levels)

        self.patch_size = self.hparams.config.train.CHUNK_SIZE
        self.DESIRED_PARTICLE_PIXELS = self.hparams.config.prepData.desired_particle_pixel_size


    def build_model(self):
        network_config = self.hparams.config.train.network
        model_name, model = network_config.build_model(img_size=self.patch_size)
        return model_name, model

    def forward_and_zero_edges(self, x):
        pred = self.forward(x)
        pred[:, :, 0:2, 0:2, 0:2] *= 0
        pred[:, :, -3:-1, -3:-1, -3:-1, ] *= 0
        return pred

    def forward(self, x):

        if isinstance(x, (tuple, list)):
            out = self.model(x[0])
        else:
            out = self.model(x)

        return out



    def configure_optimizers(self):
        train_config = self.hparams.config.train
        logging.info(f"optimizer:{train_config.optimizer}", )
        opt = hydra.utils.instantiate(train_config.optimizer, params=self.parameters()) #, decoupled_weight_decay=True
        # opt = torch.optim.RAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.hparams.mainConfig.train.WEIGHT_DECAY) #, decoupled_weight_decay=True)

        conf = {
            'optimizer': opt,
        }

        conf.update({
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True,
                                                                       factor=train_config.FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS,
                                                                       cooldown=max(1,
                                                                                    train_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS // 4),
                                                                       patience=int(
                                                                           1.5 * train_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS)),
            'monitor': 'val_loss'
        })
        return conf