import torch
import torchvision
from torch import nn

from tomocpt.configManager.configManager import safe_getattr
from tomocpt.networks.baseModel import BaseModel
from tomocpt.training.losses import gradient3d_loss
from tomocpt.training.multiclass_loss import multiclass_picking_loss
from tomocpt.logger import get_logger

logging = get_logger()


def parse_class_weights(text: str) -> dict:
    """Parse ``train.particles.class_weights`` (``"1:1.0,2:2.0"``) into ``{class number: weight}``."""
    weights = {}
    for entry in (e.strip() for e in str(text).split(",")):
        if not entry:
            continue
        key, sep, value = entry.partition(":")
        try:
            if not sep:
                raise ValueError
            number, weight = int(key.strip()), float(value.strip())
        except ValueError:
            raise ValueError(f"Malformed class_weights entry '{entry}': expected '<class number>:<weight>'")
        if weight < 0:
            raise ValueError(f"class_weights entry '{entry}': the weight must be >= 0")
        if number in weights:
            raise ValueError(f"Duplicate class {number} in class_weights")
        weights[number] = weight
    return weights


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
        self.val_loss_ema = None

        if model is None:
            self.model_name, self.model = self.build_model()
        else:
            self.model_name = model.model_name
            self.model = model.model

        self.lr = lr
    
    def setup(self, stage: str) -> None:
        """Called after the model is moved to the correct device."""
        if self.global_rank == 0:
            logging.info(f"PickingModel using {self.model_name}")

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

    def forward(self, x, apix=None):
        """
        :param x: input volume ``[B, 1, D, H, W]``
        :param apix: working voxel size in Å (float or ``[B, 1]`` tensor). A scale-conditioned
            network uses it; ``None`` runs the unconditioned computation. An unconditioned
            network ignores it (its output does not depend on the voxel size).
        :return: logits ``[B, C, D, H, W]``
        """
        if apix is None or not self.scale_conditioning():
            out = super().forward(x)
        else:
            out = self.model(x[0] if isinstance(x, (tuple, list)) else x, apix=apix)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def _batch_apix(self, batch):
        """``batch["apix_work"]`` for a scale-conditioned model, else ``None``."""
        if not self.scale_conditioning():
            return None
        if "apix_work" not in batch:
            raise ValueError(
                "train.network.scale_conditioning is on, but the batch has no 'apix_work' "
                "(scale conditioning needs data pipeline v2)"
            )
        return batch["apix_work"]

    @staticmethod
    def _is_v2_batch(batch) -> bool:
        return "class_available" in batch and "class_d_vox" in batch

    def _class_weights(self):
        """``train.particles.class_weights`` mapped to channels by class number (default 1.0),
        or ``None`` when unset."""
        particles = safe_getattr(self.hparams.config.train, "particles", None)
        text = safe_getattr(particles, "class_weights", None) if particles is not None else None
        if not text:
            return None
        if getattr(self, "_class_weights_cache", None) is None or self._class_weights_cache[0] != text:
            by_number = parse_class_weights(text)
            weights = [float(by_number.get(row.number, 1.0)) for row in self.class_table()]
            self._class_weights_cache = (text, weights)
        return self._class_weights_cache[1]

    def multiclass_loss(self, y_pred, y, batch):
        """Pipeline-v2 loss (CLAUDE.md Section 5.4) for a v2 batch."""
        particles = safe_getattr(self.hparams.config.train, "particles", None)
        use_checkpoint = bool(safe_getattr(particles, "loss_checkpoint", True)) if particles is not None else True
        return multiclass_picking_loss(
            y_pred, y, batch["class_available"], batch["class_d_vox"],
            class_weights=self._class_weights(), checkpoint=use_checkpoint,
            cross_class_neg_weight=float(safe_getattr(particles, "cross_class_neg_weight", 0.0))
            if particles is not None else 0.0,
        )

    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)
        logits = self(x, apix=self._batch_apix(batch))
        y_pred = nn.functional.sigmoid(logits)
        if self._is_v2_batch(batch):
            loss = self.multiclass_loss(y_pred, y, batch)
        else:
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
            logits = self(x, apix=self._batch_apix(batch))
            y_pred = nn.functional.sigmoid(logits)
            if self._is_v2_batch(batch):
                loss = self.multiclass_loss(y_pred, y, batch)
                self._log_per_class_val_loss(y_pred, y, batch)
            else:
                loss = self.loss(y_pred, y)

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            if batch_idx == 0 and self.global_rank == 0:
                self._log_images(x, y, y_pred, "val")
                
            return loss

    def _log_per_class_val_loss(self, y_pred, y, batch):
        """Log ``val_loss_<class name>`` for every class available in this batch."""
        available = batch["class_available"].to(torch.bool)
        d_vox = batch["class_d_vox"]
        for c, row in enumerate(self.class_table()):
            n_available = int(available[:, c].sum())
            if n_available == 0:
                continue
            loss_c = multiclass_picking_loss(
                y_pred[:, c:c + 1], y[:, c:c + 1], available[:, c:c + 1], d_vox[:, c:c + 1],
                checkpoint=False,
            )
            self.log(f"val_loss_{row.name}", loss_c, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=n_available)

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0, apix=None):
        logits = self(batch, apix=apix)
        # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
        y_pred = nn.functional.sigmoid(logits)
        return y_pred

    def _log_images(self, x, y, y_pred, prefix):
        """Central slices of input, target and prediction; one target/prediction panel per
        class (tags keep the legacy names for a single-class model)."""
        tensorboard = self.logger.experiment
        size = y_pred.shape[2]
        grid_params = {
            "padding": 1,  # Smaller gap between images
            "pad_value": 1.0,  # White borders
        }

        def add(tag, volume):
            grid = torchvision.utils.make_grid(
                volume[:, :, size // 2, :, :].to(dtype=torch.float32), **grid_params
            )
            tensorboard.add_image(tag, grid, global_step=self.current_epoch)

        add("input_val", x)
        n_classes = y_pred.shape[1]
        if n_classes == 1:
            add("label_val", y)
            add("prediction_val", y_pred)
            return
        names = [row.name for row in self.class_table()] if self.is_pipeline_v2() else []
        for c in range(n_classes):
            suffix = names[c] if c < len(names) else str(c)
            add(f"label_val_{suffix}", y[:, c:c + 1])
            add(f"prediction_val_{suffix}", y_pred[:, c:c + 1])


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
