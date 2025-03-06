import importlib
import torch
import torchvision
from torch import nn

from tomocpt.networks.baseModel import BaseModel
from tomocpt.training.losses import gradient3d_loss


class BasePickingModel(BaseModel):

    def __init__(
        self, lr: float | None = None, model=None, config_dict=None, *args, **kwargs
    ):

        super(BasePickingModel, self).__init__(lr, *args, **kwargs)

        # Store config if needed
        self.config = config_dict

        self.different_lrs = False
        if model is None:
            self.model_name, self.model = self.build_model()
        else:
            self.model_name = model.model_name
            self.model = model.model
            self.different_lrs = self.config.train.DIFFERENT_LRS_FOR_FINETUNE

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
        # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
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

        if batch_idx == 0:
            size = y_pred.shape[2]
            grid = torchvision.utils.make_grid(x[:, :, size // 2, :, :])
            tensorboard = self.logger.experiment
            tensorboard.add_image(
                "input", grid.to(dtype=torch.float32), global_step=self.current_epoch
            )
            grid = torchvision.utils.make_grid(
                y[:, :, size // 2, :, :].to(dtype=torch.float32)
            )
            tensorboard.add_image(
                "label", grid.to(dtype=torch.float32), global_step=self.current_epoch
            )
            grid = torchvision.utils.make_grid(
                y_pred[:, :, size // 2, :, :].to(dtype=torch.float32)
            )
            tensorboard.add_image(
                "prediction",
                grid.to(dtype=torch.float32),
                global_step=self.current_epoch,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = self.resolve_batch(batch)
            logits = self(x)
            y_pred = nn.functional.sigmoid(logits)
            # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
            loss = self.loss(y_pred, y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )

        tensorboard = self.logger.experiment
        size = y_pred.shape[2]
        # print(x.shape)
        grid1 = torchvision.utils.make_grid(x[:, :, size // 2, :, :])
        tensorboard.add_image(
            "input_val", grid1.to(dtype=torch.float32), global_step=self.current_epoch
        )
        grid2 = torchvision.utils.make_grid(y[:, :, size // 2, :, :])
        tensorboard.add_image(
            "label_val", grid2.to(dtype=torch.float32), global_step=self.current_epoch
        )
        grid3 = torchvision.utils.make_grid(
            y_pred[:, :, size // 2, :, :].to(dtype=torch.float32)
        )
        tensorboard.add_image(
            "prediction_val",
            grid3.to(dtype=torch.float32),
            global_step=self.current_epoch,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        y_pred = nn.functional.sigmoid(logits)
        return y_pred

    def _get_model_parameters_for_optimizer(self):
        #self.different_lrs = True #For debugging
        if hasattr(self.model, "_get_model_parameters_for_optimizer"):
            return self.model._get_model_parameters_for_optimizer(self.different_lrs)
        return self.parameters()


if __name__ == "__main__":
    model = BasePickingModel()

    model._get_model_parameters_for_optimizer()
    batch_size = 3
    from tomocpt.mainConfig import mainConfig

    train_config = mainConfig.train
    class DataWrapper():
        def __init__(self, data):
            self.data = data
        def __getattr__(self, name):
            if name == "data":
                return self.data
            else:
                return getattr(self.data, name)

    batch = dict(
        input_data=dict(
            data=DataWrapper(torch.randn(batch_size, 1, *(train_config.CHUNK_SIZE,) * 3))
        ),
        target_data=dict(
            data=DataWrapper(torch.randn(batch_size, 1, *(train_config.CHUNK_SIZE,) * 3))
        ),
    )

    out = model.predict_step(batch["input_data"]["data"], 0, dataloader_idx=0)
    out = model.training_step(batch, 0)
