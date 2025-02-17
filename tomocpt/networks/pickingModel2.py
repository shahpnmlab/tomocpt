import warnings
import torch
import torchvision
from torch import nn

import hydra
from tomocpt.networks.baseModel import BaseModel
from tomocpt.training.losses import gradient3d_loss

from swinuneter_continue import ContinualSwinUNETR


def find_last_conv3d(model):
    """Helper function to find the last Conv3d layer in a model"""
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            last_conv = module
    return last_conv


class BasePickingModel(BaseModel):
    def __init__(
        self,
        lr: float | None = None,
        model=None,
        config_dict=None,
        pretrained_weights: str = None,
        *args,
        **kwargs,
    ):
        super(BasePickingModel, self).__init__(lr, *args, **kwargs)

        self.config = config_dict

        if model is None:
            self.model_name, self.model = self.build_model()
        else:
            self.model_name = model.model_name
            self.model = model.model

        # If pretrained weights are provided, use ContinualSwinUNETR
        if pretrained_weights is not None:
            print(
                f"Initializing ContinualSwinUNETR with pretrained weights from {pretrained_weights}"
            )

            # Create new ContinualSwinUNETR with same architecture parameters
            if not isinstance(self.model, ContinualSwinUNETR):
                # Get input channels from first conv layer
                in_channels = self.model.swinViT.patch_embed.proj.in_channels

                # Find the last Conv3d layer to get output channels
                last_conv = find_last_conv3d(self.model)
                if last_conv is None:
                    raise ValueError("Could not find Conv3d layer in model")
                out_channels = last_conv.out_channels

                # Get feature size from hidden dimension
                feature_size = self.model.swinViT.embed_dim

                # Get patch size from SwinViT
                patch_size = self.model.swinViT.patch_embed.patch_size
                img_size = tuple(
                    patch_size[0] * 16 for _ in range(3)
                )  # Assuming cubic input

                # Check if using checkpointing
                use_checkpoint = getattr(self.model, "use_checkpoint", False)

                print(f"Creating ContinualSwinUNETR with parameters:")
                print(f"- img_size: {img_size}")
                print(f"- in_channels: {in_channels}")
                print(f"- out_channels: {out_channels}")
                print(f"- feature_size: {feature_size}")
                print(f"- use_checkpoint: {use_checkpoint}")

                # Store current state dict
                state_dict = self.model.state_dict()

                # Create new ContinualSwinUNETR
                self.model = ContinualSwinUNETR(
                    img_size=img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    use_checkpoint=use_checkpoint,
                )

                # Load the state dict
                self.model.load_state_dict(state_dict)

            # Load previous weights and setup for continual learning
            self.model.load_previous_weights(pretrained_weights)
            self.model.freeze_backbone(freeze_norm_layers=False)

        print(f"PickingModel using {self.model_name}")
        self.lr = lr

    def configure_optimizers(self):
        if isinstance(self.model, ContinualSwinUNETR):
            # Use layer-wise learning rates
            layer_groups = self.model.get_layer_groups()
            base_lr = self.lr or self.hparams.config.train.optimizer.lr

            # Create optimizer with different learning rates per group
            parameters = [
                {"params": group["params"], "lr": base_lr * group["lr_mult"]}
                for group in layer_groups
            ]

            opt = hydra.utils.instantiate(
                self.hparams.config.train.optimizer, params=parameters
            )
        else:
            # Use standard optimization for regular model
            opt = super().configure_optimizers()

        return opt

    def loss(self, y_pred, y_true, prev_logits=None):
        # Original picking loss computation from BasePickingModel
        eps = 1
        num_positive = torch.sum(y_true, dim=[1, 2, 3, 4])
        num_negative = y_true.numel() / y_true.shape[0] - num_positive

        # Compute weights
        positive_weights = (y_true.numel() / y_true.shape[0]) / (num_positive + eps)
        negative_weights = (y_true.numel() / y_true.shape[0]) / (num_negative + eps)

        # Reshape weights to be broadcastable
        positive_weights = positive_weights.view(y_true.shape[0], 1, 1, 1, 1)
        negative_weights = negative_weights.view(y_true.shape[0], 1, 1, 1, 1)

        # Create importance mask
        importance_mask = torch.where(
            y_true > 0, positive_weights, 2 * negative_weights
        )

        # Compute base loss
        base_loss = importance_mask * nn.functional.huber_loss(
            y_pred, y_true, reduction="none"
        )
        base_loss += 10 * gradient3d_loss(y_pred, y_true).sum(2)
        base_loss = base_loss.mean()

        # Add distillation loss if previous model exists
        if prev_logits is not None and isinstance(self.model, ContinualSwinUNETR):
            distill_loss = self.model.compute_distillation_loss(
                y_pred, prev_logits, alpha=0.5
            )
            return base_loss + distill_loss

        return base_loss

    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)

        # Handle different forward pass outputs
        if isinstance(self.model, ContinualSwinUNETR):
            logits, hidden, prev_logits = self.model(x)
            y_pred = nn.functional.sigmoid(logits)
            loss = self.loss(y_pred, y, prev_logits)
        else:
            logits = self(x)
            y_pred = nn.functional.sigmoid(logits)
            loss = self.loss(y_pred, y)

        # Only try to log if we're in a training context with a logger
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and self.trainer.logger is not None
        ):
            self.log(
                "loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )

            # Log images on first batch
            if batch_idx == 0:
                size = y_pred.shape[2]
                grid = torchvision.utils.make_grid(x[:, :, size // 2, :, :])
                self.trainer.logger.experiment.add_image(
                    "input",
                    grid.to(dtype=torch.float32),
                    global_step=self.current_epoch,
                )
                grid = torchvision.utils.make_grid(
                    y[:, :, size // 2, :, :].to(dtype=torch.float32)
                )
                self.trainer.logger.experiment.add_image(
                    "label",
                    grid.to(dtype=torch.float32),
                    global_step=self.current_epoch,
                )
                grid = torchvision.utils.make_grid(
                    y_pred[:, :, size // 2, :, :].to(dtype=torch.float32)
                )
                self.trainer.logger.experiment.add_image(
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
        # y_pred = nn.functional.softplus(logits, beta=BETA_FOR_SOFTPLUS)
        y_pred = nn.functional.sigmoid(logits)
        return y_pred

    def resolve_batch(self, batch):
        x = batch["input_data"]["data"]
        y = batch["target_data"]["data"]
        # y = (y>0).float()
        return x, y

    warnings.filterwarnings(
        "ignore", message=".*monai.networks.nets.swin_unetr.*", category=FutureWarning
    )


from torch.utils.data import DataLoader, Dataset


# Create a simple dataset class for our synthetic data
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, chunk_size):
        self.num_samples = num_samples
        self.chunk_size = chunk_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_data": {"data": torch.randn(1, *(self.chunk_size,) * 3)},
            "target_data": {"data": torch.randn(1, *(self.chunk_size,) * 3)},
        }


if __name__ == "__main__":
    import os
    from tomocpt.mainConfig import mainConfig
    import os
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    import torch
    from torch.utils.data import DataLoader, Dataset

    # Create directories if they don't exist
    os.makedirs("checkpoints", exist_ok=True)

    # Create synthetic dataset
    train_config = mainConfig.train
    dataset = SyntheticDataset(num_samples=4, chunk_size=train_config.CHUNK_SIZE)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Phase 1: Initial training with MySwinUNETR
    print("\n=== Phase 1: Initial Training ===")

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="myswinunetr-{epoch:02d}",
        save_top_k=1,
        monitor="loss",
        save_last=True,
    )

    # Initialize model and trainer
    initial_model = BasePickingModel()
    trainer = Trainer(
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        max_epochs=2,
        logger=False,
        accelerator="mps",  # Use MPS for Mac
        devices=1,
    )

    # Train the model properly using fit
    trainer.fit(initial_model, dataloader)

    # Get the path of the saved checkpoint
    checkpoint_path = checkpoint_callback.best_model_path
    if not checkpoint_path:
        checkpoint_path = os.path.join("checkpoints", "last.ckpt")
    print(f"\nSaved checkpoint to: {checkpoint_path}")

    # Phase 2: Fine-tuning with ContinualSwinUNETR
    print("\n=== Phase 2: Fine-tuning ===")

    try:
        # Initialize new model with pretrained weights
        fine_tune_model = BasePickingModel(
            lr=0.0001,  # Lower learning rate for fine-tuning
            pretrained_weights=checkpoint_path,
        )

        # Setup new trainer for fine-tuning
        fine_tune_trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=1,
            logger=False,
            accelerator="mps",
            devices=1,
        )

        # Verify model type and run one epoch
        print(f"Model type: {type(fine_tune_model.model).__name__}")

        # Train for one epoch using fit
        fine_tune_trainer.fit(fine_tune_model, dataloader)

    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise
    finally:
        # Cleanup checkpoints
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if os.path.exists(os.path.join("checkpoints", "last.ckpt")):
            os.remove(os.path.join("checkpoints", "last.ckpt"))
        os.rmdir("checkpoints")
