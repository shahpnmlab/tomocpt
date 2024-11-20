import pytorch_lightning as pl

from tomocpt import constants
from tomocpt.configManager.configManager import update_config
from tomocpt.mainConfig import mainConfig

class BaseModel(pl.LightningModule):

    def set_default_args(self, lr: float | None, num_levels: int | None):
        self.lr = lr if lr is not None else self.hparams.config.train.optimizer.lr
        self.num_levels = num_levels if num_levels is not None else self.hparams.config.train.network.NUM_LEVELS


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

        self.patch_size = self.hparams.config.train.network.CHUNK_SIZE
        self.DESIRED_PARTICLE_PIXELS = self.hparams.config.prepData.DESIRED_PARTICLE_PIXELS

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
