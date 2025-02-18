import os
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
import pytorch_lightning as pl
from tomocpt.networks.swinuneter_continue import IntegratedContinualSwinUNETR
# Suppress MONAI warnings
warnings.filterwarnings(
    "ignore", message=".*monai.networks.nets.swin_unetr.*", category=FutureWarning
)


class SyntheticDataset(Dataset):
    """Simple dataset for testing with proper normalization"""

    def __init__(self, num_samples, chunk_size):
        self.num_samples = num_samples
        self.chunk_size = chunk_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate input data (can be any range as it's feature input)
        input_data = torch.randn(1, *(self.chunk_size,) * 3)

        # Generate target data and ensure it's between 0 and 1
        target_data = torch.rand(1, *(self.chunk_size,) * 3)  # rand generates values in [0,1]

        return {
            "input_data": {"data": input_data},
            "target_data": {"data": target_data}
        }


class TestPickingModel(pl.LightningModule):
    """Lightning module for testing the integrated model"""

    def __init__(self, lr=0.001, model=None, pretrained_weights=None):
        super().__init__()
        self.lr = lr
        self.is_continual = pretrained_weights is not None

        # Initialize model
        if model is None:
            self.model = IntegratedContinualSwinUNETR(
                img_size=(64, 64, 64),
                in_channels=1,
                out_channels=1,
                feature_size=48,
                use_checkpoint=False
            )
        else:
            self.model = model

        # Load pretrained weights if provided
        if pretrained_weights is not None:
            print(f"Loading pretrained weights from {pretrained_weights}")
            self.model.load_previous_weights(pretrained_weights)
            self.model.freeze_backbone(freeze_norm_layers=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = self._get_inputs_targets(batch)

        # Handle different forward pass outputs
        outputs = self(x)

        if isinstance(outputs, tuple):
            if len(outputs) == 3:  # Continual learning mode
                logits, hidden, prev_logits = outputs
                y_pred = torch.sigmoid(logits)
                base_loss = self._compute_loss(y_pred, y)
                distill_loss = self.model.compute_distillation_loss(logits, prev_logits, alpha=0.5)
                loss = base_loss + distill_loss

                # Log components
                self.log('train_base_loss', base_loss, on_step=True, on_epoch=True)
                self.log('train_distill_loss', distill_loss, on_step=True, on_epoch=True)
            else:  # Initial training mode
                logits, hidden = outputs
                y_pred = torch.sigmoid(logits)
                loss = self._compute_loss(y_pred, y)
        else:  # Direct output mode
            y_pred = torch.sigmoid(outputs)
            loss = self._compute_loss(y_pred, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._get_inputs_targets(batch)
        outputs = self(x)
        # During validation, we always get logits directly
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        y_pred = torch.sigmoid(logits)
        loss = self._compute_loss(y_pred, y)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def _compute_loss(self, y_pred, y_true):
        """
        Compute binary cross entropy loss with safety checks
        """
        # Ensure predictions are between 0 and 1
        y_pred = torch.clamp(y_pred, 0, 1)

        # Ensure targets are between 0 and 1
        y_true = torch.clamp(y_true, 0, 1)

        return torch.nn.functional.binary_cross_entropy(y_pred, y_true)

    def _get_inputs_targets(self, batch):
        """Extract inputs and targets from batch dictionary"""
        return batch["input_data"]["data"], batch["target_data"]["data"]

    def configure_optimizers(self):
        if hasattr(self.model, 'get_layer_groups') and self.is_continual:
            # Use layer-wise learning rates for continual learning
            layer_groups = self.model.get_layer_groups()
            parameters = [
                {"params": group["params"], "lr": self.lr * group["lr_mult"]}
                for group in layer_groups
            ]
            optimizer = torch.optim.Adam(parameters)
        else:
            # Standard optimization for initial training
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer


def main():
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)

    # Create synthetic dataset
    CHUNK_SIZE = 64
    dataset = SyntheticDataset(num_samples=4, chunk_size=CHUNK_SIZE)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Phase 1: Initial training
    print("\n=== Phase 1: Initial Training ===")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="swinunetr-{epoch:02d}",
        save_top_k=1,
        monitor="train_loss",
        save_last=True,
    )

    initial_model = TestPickingModel()
    trainer = Trainer(
        max_epochs=2,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1,
        logger=False
    )

    try:
        # Train initial model
        trainer.fit(initial_model, dataloader)

        # Get checkpoint path
        checkpoint_path = checkpoint_callback.best_model_path
        if not checkpoint_path:
            checkpoint_path = os.path.join("checkpoints", "last.ckpt")
        print(f"\nSaved checkpoint to: {checkpoint_path}")

        # Phase 2: Continual learning
        print("\n=== Phase 2: Continual Learning ===")

        # Initialize new model with pretrained weights
        fine_tune_model = TestPickingModel(
            lr=0.0001,  # Lower learning rate for fine-tuning
            pretrained_weights=checkpoint_path
        )

        # Create new trainer for fine-tuning
        fine_tune_trainer = Trainer(
            max_epochs=1,
            accelerator="auto",
            devices=1,
            logger=False
        )

        # Verify model type
        print(f"Model type: {type(fine_tune_model.model).__name__}")

        # Train for one epoch
        fine_tune_trainer.fit(fine_tune_model, dataloader)

        # Test inference
        print("\n=== Testing Inference ===")
        sample_batch = next(iter(dataloader))
        with torch.no_grad():
            outputs = fine_tune_model(sample_batch["input_data"]["data"])
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get logits only
            print(f"Output shape: {outputs.shape}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if os.path.exists(os.path.join("checkpoints", "last.ckpt")):
            os.remove(os.path.join("checkpoints", "last.ckpt"))
        os.rmdir("checkpoints")


if __name__ == "__main__":
    main()