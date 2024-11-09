import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
import torchvision
from torch import nn
from torch.nn.functional import binary_cross_entropy

from pycotool import constants, config
from pycotool.dataManager.dataUtils import resize_volume
from pycotool.networks.baseModel import BaseModel
from pycotool.networks.unet import Unet
from pycotool.training.losses import dice_loss, gradient3d_loss

from monai.networks.nets import SwinUNETR

BETA_FOR_SOFTPLUS = 2


class BasePickingModel(BaseModel):  # TODO: Change the name
    def __init__(self, lr=constants.LEARNING_RATE, num_levels=constants.NUM_LEVELS, model=None):
        super(BasePickingModel, self).__init__()
        '''
        change the model here to change the arch using monai.
        '''
        if model is None:
            self.model_name = config.MODEL_TYPE
            if config.MODEL_TYPE == "UNET" or config.MODEL_TYPE == "unet":
                self.model = Unet(first_layer_out_channels=constants.FIRST_LAYER_OUT_CHANNELS, last_activation="linear",
                                  stride_conv_instead_pooling=True, num_levels=num_levels)

            elif config.MODEL_TYPE == "SwinUNETR" or config.MODEL_TYPE == "swinunetr":
                self.model = SwinUNETR(img_size=(constants.CHUNK_SIZE, constants.CHUNK_SIZE, constants.CHUNK_SIZE),
                                       in_channels=1,
                                       out_channels=1,
                                       feature_size=constants.SWINUNETR_FEAT_SIZE,  # should be divisible by 12
                                       use_v2=True
                                       )
            else:
                raise ValueError(f"Not valid model type ({config.MODEL_TYPE}")
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

    # def loss(self, y_hat, y_true, logits):
    #
    #     eps = 1
    #     num_positive = torch.sum(y_true, dim=[1, 2, 3, 4])
    #     num_negative = y_true.numel() / y_true.shape[0] - num_positive
    #
    #     # Compute weights
    #     # Adding eps to the denominator to avoid division by zero
    #     positive_weights = (y_true.numel() / y_true.shape[0]) / (num_positive + eps)
    #     negative_weights = (y_true.numel() / y_true.shape[0]) / (num_negative + eps)
    #
    #     # Reshape weights to be broadcastable
    #     positive_weights = positive_weights.view(y_true.shape[0], 1, 1, 1, 1)
    #     negative_weights = negative_weights.view(y_true.shape[0], 1, 1, 1, 1)
    #
    #     # Compute the binary cross-entropy with logits
    #     max_val = torch.clamp(-logits, min=0)
    #     loss = (1 - y_true) * logits + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log()
    #
    #     # Apply the weights
    #     weighted_loss = negative_weights * loss * (1 - y_true) + positive_weights * loss * y_true
    #     # Average the loss over each image but keep it separate for each image in the batch
    #     average_loss_per_image = torch.mean(weighted_loss, dim=[1, 2, 3, 4])
    #     # Finally, average the loss over the batch
    #     bce = torch.mean(average_loss_per_image)
    #
    #     dice = dice_loss(y_hat, y_true)
    #     return bce+dice

    def resolve_batch(self, batch):
        x = batch["input_data"][tio.DATA]
        y = batch["target_data"][tio.DATA]
        # y = (y>0).float()
        return x, y

    def forward(self, x):
        out = super().forward(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)
        logits = self(x)
        y_hat = nn.functional.sigmoid(logits)
        y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
        loss = self.loss(y_pred, y)
        #loss = self.loss(y_hat, y, logits)
        with torch.no_grad():
            dice = dice_loss(y_hat, (y > 0).float()[:, :1, ...])

        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])  # TODO: if on_step=True, reconsider sync_dist

        self.log('dice', dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])  # TODO: if on_step=True, reconsider sync_dist

        if batch_idx == 0:
            size = y_hat.shape[2]
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
            y_hat = nn.functional.sigmoid(logits)
            y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
            loss = self.loss(y_pred, y)
            dice = dice_loss(y_hat, (y > 0).float()[:, :1, ...])
            #loss = self.loss(y_hat, y, logits)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.shape[0])
        self.log('dice', dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.shape[0])

        tensorboard = self.logger.experiment
        size = y_hat.shape[2]
        # print(x.shape)
        grid1 = torchvision.utils.make_grid(x[:, :, size // 2, :, :])
        tensorboard.add_image("input_val", grid1.to(dtype=torch.float32), global_step=self.current_epoch)
        grid2 = torchvision.utils.make_grid(y[:, :, size // 2, :, :])
        tensorboard.add_image("label_val", grid2.to(dtype=torch.float32), global_step=self.current_epoch)
        grid3 = torchvision.utils.make_grid(y_pred[:, :, size // 2, :, :].to(dtype=torch.float32))
        tensorboard.add_image("prediction_val", grid3.to(dtype=torch.float32), global_step=self.current_epoch)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        logits = self(x)
        y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
        return y_pred

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-10)

        conf = {
            'optimizer': opt,
        }

        conf.update({
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True,
                                                                       factor=constants.FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS,
                                                                       cooldown=max(1,
                                                                                    constants.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS // 4),
                                                                       patience=int(
                                                                           1.5 * constants.PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS)),
            'monitor': 'val_loss'
        })
        return conf


if __name__ == "__main__":
    model = BasePickingModel()
    batch_size = 3

    batch = dict(
        input_data=dict(data=torch.randn(batch_size, 1, *(constants.CHUNK_SIZE,) * 3)),
        target_data=dict(data=torch.randn(batch_size, 1, *(constants.CHUNK_SIZE,) * 3)),
    )

    out = model.predict_step(batch["input_data"]["data"], 0, dataloader_idx=0)
    out = model.training_step(batch, 0)
