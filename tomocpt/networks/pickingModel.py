import torch
import torchio as tio
import torchvision
from torch import nn

from tomocpt.mainConfig import mainConfig
network_config = mainConfig.network
train_config = mainConfig.train
from tomocpt.networks.baseModel import BaseModel
from tomocpt.networks.unet import Unet
from tomocpt.training.losses import gradient3d_loss

from monai.networks.nets import SwinUNETR

BETA_FOR_SOFTPLUS = 2


class BasePickingModel(BaseModel):# TODO: Change the name
    def set_default_args(self, lr: float | None,
                         num_levels: int | None):

        self.lr = lr if lr is not None else network_config.LEARNING_RATE
        self.num_levels = num_levels if num_levels is not None else network_config.NUM_LEVELS

    def __init__(self, lr: float | None = None,
                 num_levels: int | None = None,
                 model=None):
        self.set_default_args(lr=lr,
                              num_levels=num_levels)
        super(BasePickingModel, self).__init__()

        # TODO:change the model here to change the arch using monai.

        MODEL_TYPES = {
            "UNET": lambda: Unet(
                first_layer_out_channels=network_config.FIRST_LAYER_OUT_CHANNELS,
                last_activation="linear",
                stride_conv_instead_pooling=True,
                num_levels=num_levels
            ),
            "SWINUNETR": lambda: SwinUNETR(
                img_size=(network_config.CHUNK_SIZE, network_config.CHUNK_SIZE, network_config.CHUNK_SIZE),
                in_channels=1,
                out_channels=1,
                feature_size=network_config.SWINUNETR_FEAT_SIZE,
                use_v2=True,
                use_checkpoint=True,
                drop_rate=network_config.DROP_RATE,
                attn_drop_rate=network_config.ATTN_DROP_RATE,
                dropout_path_rate=network_config.DROPOUT_PATH_RATE,
            )
        }

        if model is None:
            self.model_name = str(network_config.model_type.value)
            model_constructor = MODEL_TYPES.get(self.model_name.upper())
            if model_constructor:
                self.model = model_constructor()
            else:
                raise ValueError(f"Unknown model type: {network_config.model_type.value}")
        else:
            self.model_name = model.model_name
            self.model = model.model

        print(f"PickingModel using {self.model_name}")
        self.lr = lr

    def loss(self, y_pred, y_true):
        eps = 1
        num_positive = torch.sum(y_true, dim=[1, 2, 3, 4])
        num_negative = y_true.numel() / y_true.shape[0] - num_positive

        # Compute weights
        # Adding eps to the denominator to avoid division by zero
        positive_weights = (y_true.numel() / y_true.shape[0]) / (num_positive + eps)
        negative_weights = (y_true.numel() / y_true.shape[0]) / (num_negative + eps)

        # Reshape weights to be broadcastable
        positive_weights = positive_weights.view(y_true.shape[0], 1, 1, 1, 1)
        negative_weights = negative_weights.view(y_true.shape[0], 1, 1, 1, 1)

        # Create importance mask
        importance_mask = torch.where(y_true > 0, positive_weights, 2 * negative_weights)

        # Compute MSE
        # loss = importance_mask * (y_pred - y_true) ** 2
        loss = importance_mask * nn.functional.huber_loss(y_pred, y_true, reduction="none")
        _loss = 10 * gradient3d_loss(y_pred, y_true).sum(2)
        loss += _loss
        loss = loss.mean()
        return loss

    def resolve_batch(self, batch):
        x = batch["input_data"][tio.DATA]
        y = batch["target_data"][tio.DATA]
        # y = (y>0).float()
        return x, y

    def forward(self, x):
        """

        :param x:
        :return: logits
        """
        out = super().forward(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)
        logits = self(x)
        y_pred = nn.functional.sigmoid(logits)
        # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
        loss = self.loss(y_pred, y)

        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])  # TODO: if on_step=True, reconsider sync_dist

        if batch_idx == 0:
            size = y_pred.shape[2]
            grid = torchvision.utils.make_grid(x[:, :, size // 2, :, :])
            tensorboard = self.logger.experiment
            tensorboard.add_image("input", grid.to(dtype=torch.float32), global_step=self.current_epoch)
            grid = torchvision.utils.make_grid(y[:, :, size // 2, :, :].to(dtype=torch.float32))
            tensorboard.add_image("label", grid.to(dtype=torch.float32), global_step=self.current_epoch)
            grid = torchvision.utils.make_grid(y_pred[:, :, size // 2, :, :].to(dtype=torch.float32))
            tensorboard.add_image("prediction", grid.to(dtype=torch.float32), global_step=self.current_epoch)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = self.resolve_batch(batch)
            logits = self(x)
            y_pred = nn.functional.sigmoid(logits)
            # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
            loss = self.loss(y_pred, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.shape[0])

        tensorboard = self.logger.experiment
        size = y_pred.shape[2]
        # print(x.shape)
        grid1 = torchvision.utils.make_grid(x[:, :, size // 2, :, :])
        tensorboard.add_image("input_val", grid1.to(dtype=torch.float32), global_step=self.current_epoch)
        grid2 = torchvision.utils.make_grid(y[:, :, size // 2, :, :])
        tensorboard.add_image("label_val", grid2.to(dtype=torch.float32), global_step=self.current_epoch)
        grid3 = torchvision.utils.make_grid(y_pred[:, :, size // 2, :, :].to(dtype=torch.float32))
        tensorboard.add_image("prediction_val", grid3.to(dtype=torch.float32), global_step=self.current_epoch)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
        y_pred = nn.functional.sigmoid(logits)
        return y_pred

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.99),
                                weight_decay=train_config.WEIGHT_DECAY) #, decoupled_weight_decay=True)

        conf = {
            'optimizer': opt,
        }

        conf.update({
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True,
                                                                       factor=network_config.FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS,
                                                                       cooldown=max(1,
                                                                                    network_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS // 4),
                                                                       patience=int(
                                                                           1.5 * network_config.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS)),
            'monitor': 'val_loss'
        })
        return conf


if __name__ == "__main__":
    model = BasePickingModel()
    batch_size = 3

    batch = dict(
        input_data=dict(data=torch.randn(batch_size, 1, *(network_config.CHUNK_SIZE,) * 3)),
        target_data=dict(data=torch.randn(batch_size, 1, *(network_config.CHUNK_SIZE,) * 3)),
    )

    out = model.predict_step(batch["input_data"]["data"], 0, dataloader_idx=0)
    out = model.training_step(batch, 0)
