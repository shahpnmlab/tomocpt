import pytorch_lightning as pl
from tomocpt import constants, network_config


class BaseModel(pl.LightningModule):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.save_hyperparameters(
            dict(constants_dict={k: getattr(constants, k) for k in dir(constants) if not k.startswith("__")},
                 config_dict=network_config)) #TODO: We need to serialize them to dicts

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
