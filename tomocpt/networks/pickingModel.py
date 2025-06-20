import torch
import torchvision
from torch import nn

from tomocpt.networks.baseModel import BaseModel
from tomocpt.training.losses import gradient3d_loss
from tomocpt.logger import get_logger

logging = get_logger()


class BasePickingModel(BaseModel):

    def __init__(
            self, lr: float | None = None, model=None, config_dict=None,
            ema_alpha: float = 0.3, *args, **kwargs
    ):
        # EMA Alpha (smoothing factor):
        # ema_alpha=0.3: 30% current loss + 70% historical average = quick adaptation
        # ema_alpha=0.1: 10% current loss + 90% historical average = heavy smoothing
        # Higher alpha = more responsive to recent validation losses
        # Lower alpha = more noise filtering, slower to detect real changes

        # Remove config_dict from kwargs before passing to parent
        if "config_dict" in kwargs:
            config_dict = kwargs.pop("config_dict")

        super(BasePickingModel, self).__init__(lr, *args, **kwargs)

        # Store config if needed
        self.config = config_dict
        self.ema_alpha = ema_alpha  # Smoothing factor for EMA

        if model is None:
            self.model_name, self.model = self.build_model()
        else:
            self.model_name = model.model_name
            self.model = model.model

        logging.info(f"PickingModel using {self.model_name}")
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
        importance_mask = torch.where(
            y_true > 0, positive_weights, 2 * negative_weights
        )

        # Compute MSE
        # loss = importance_mask * (y_pred - y_true) ** 2
        loss = importance_mask * nn.functional.huber_loss(
            y_pred, y_true, reduction="none"
        )
        _loss = 10 * gradient3d_loss(y_pred, y_true).sum(2)
        loss += _loss
        loss = loss.mean()
        return loss

    def resolve_batch(self, batch):
        x = batch["input_data"].data
        y = batch["target_data"].data
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
        loss = self.loss(y_pred, y)

        self.log(
            "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )  # if on_step=True, reconsider sync_dist

        if batch_idx == 0 and self.global_rank == 0:
            self._log_images(x, y, y_pred, "train")
        return loss


    def validation_step(self, batch, batch_idx):
            x, y = self.resolve_batch(batch)
            logits = self(x)
            y_pred = nn.functional.sigmoid(logits)
            loss = self.loss(y_pred, y)

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            if batch_idx == 0 and self.global_rank == 0:
                self._log_images(x, y, y_pred, "val")
                
            return loss

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute the smoothed loss.
        """
        current_val_loss = self.trainer.callback_metrics.get("val_loss")
        if current_val_loss is None:
            return

        # Calculate the EMA on the epoch's average loss
        if self.val_loss_ema is None:
            self.val_loss_ema = current_val_loss
        else:
            alpha = self.hparams.ema_alpha
            self.val_loss_ema = alpha * current_val_loss + (1 - alpha) * self.val_loss_ema
        
        # Log the smoothed metric for EarlyStopping and ModelCheckpoint
        self.log("val_loss_smooth", self.val_loss_ema, prog_bar=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
        y_pred = nn.functional.sigmoid(logits)
        return y_pred

    def _log_images(self, x, y, y_pred, prefix):
        tensorboard = self.logger.experiment
        size = y_pred.shape[2]
        grid_params = {
            "padding": 1,  # Smaller gap between images
            "pad_value": 1.0,  # White borders
        }
        grid1 = torchvision.utils.make_grid(x[:, :, size // 2, :, :], **grid_params)
        tensorboard.add_image(
            "input_val", grid1.to(dtype=torch.float32), global_step=self.current_epoch
        )
        grid2 = torchvision.utils.make_grid(y[:, :, size // 2, :, :], **grid_params)
        tensorboard.add_image(
            "label_val", grid2.to(dtype=torch.float32), global_step=self.current_epoch
        )
        grid3 = torchvision.utils.make_grid(
            y_pred[:, :, size // 2, :, :].to(dtype=torch.float32), **grid_params
        )
        tensorboard.add_image(
            "prediction_val",
            grid3.to(dtype=torch.float32),
            global_step=self.current_epoch,
        )
        return loss



if __name__ == "__main__":
    model = BasePickingModel()
    batch_size = 3
    from tomocpt.mainConfig import mainConfig

    train_config = mainConfig.train
    batch = dict(
        input_data=dict(
            data=torch.randn(batch_size, 1, *(train_config.CHUNK_SIZE,) * 3)
        ),
        target_data=dict(
            data=torch.randn(batch_size, 1, *(train_config.CHUNK_SIZE,) * 3)
        ),
    )

    out = model.predict_step(batch["input_data"]["data"], 0, dataloader_idx=0)
    out = model.training_step(batch, 0)
