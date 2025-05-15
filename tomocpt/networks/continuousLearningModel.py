import copy
import random
import torch
import torchvision
from torch import nn

from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.training.losses import distillation_loss
from tomocpt.logger import get_logger

logging = get_logger()


class ContinuousLearningModel(BasePickingModel):
    """
    Enhanced picking model with continuous learning capabilities.
    Maintains a memory buffer of important samples and uses knowledge
    distillation to prevent catastrophic forgetting.
    """

    def __init__(
            self,
            memory_size=1000,
            replay_batch_size=8,
            replay_frequency=2,
            distill_weight=0.5,
            temperature=2.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.memory_buffer = []
        self.memory_size = memory_size
        self.replay_batch_size = replay_batch_size
        self.replay_frequency = replay_frequency
        self.distill_weight = distill_weight
        self.temperature = temperature
        self.task_id = 0
        self.teacher_model = None

        logging.info(f"Initialized ContinuousLearningModel with memory size {memory_size}")

    def update_memory(self, batch, loss_per_sample):
        """
        Update memory buffer with samples from the current batch,
        using loss as an importance metric for sample selection.
        """
        x, y = self.resolve_batch(batch)
        for i in range(x.shape[0]):
            self.memory_buffer.append({
                'input': x[i:i + 1].detach().clone(),
                'target': y[i:i + 1].detach().clone(),
                'importance': loss_per_sample[i].item(),
                'task_id': self.task_id
            })

        # Maintain buffer size by removing least important samples
        if len(self.memory_buffer) > self.memory_size:
            self.memory_buffer.sort(key=lambda x: x['importance'])
            self.memory_buffer = self.memory_buffer[-self.memory_size:]

    def training_step(self, batch, batch_idx):
        # Regular training on current batch
        x, y = self.resolve_batch(batch)
        logits = self(x)
        y_pred = nn.functional.sigmoid(logits)
        current_loss = self.loss(y_pred, y)

        # Get per-sample loss for memory update
        with torch.no_grad():
            sample_loss = nn.functional.huber_loss(y_pred, y, reduction='none').mean([1, 2, 3, 4])

        # Update memory with current batch samples
        self.update_memory(batch, sample_loss)

        # Calculate final loss
        final_loss = current_loss

        # Perform replay and distillation if needed
        if self.memory_buffer and self.current_epoch % self.replay_frequency == 0:
            replay_loss = self._compute_replay_loss()
            final_loss = 0.7 * current_loss + 0.3 * replay_loss

        # Log metrics
        self.log(
            "loss",
            final_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )

        # Visualization for tensorboard (same as original BasePickingModel)
        if batch_idx == 0:
            size = y_pred.shape[2]
            grid_params = {
                "padding": 1,
                "pad_value": 1.0,
            }
            grid = torchvision.utils.make_grid(x[:, :, size // 2, :, :], **grid_params)
            tensorboard = self.logger.experiment
            tensorboard.add_image(
                "input",
                grid.to(dtype=torch.float32),
                global_step=self.current_epoch,
            )
            grid = torchvision.utils.make_grid(
                y[:, :, size // 2, :, :].to(dtype=torch.float32), **grid_params
            )
            tensorboard.add_image(
                "label", grid.to(dtype=torch.float32), global_step=self.current_epoch
            )
            grid = torchvision.utils.make_grid(
                y_pred[:, :, size // 2, :, :].to(dtype=torch.float32), **grid_params
            )
            tensorboard.add_image(
                "prediction",
                grid.to(dtype=torch.float32),
                global_step=self.current_epoch,
            )

            # Additional logging for continuous learning
            self.log(
                "memory_buffer_size",
                len(self.memory_buffer),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            if hasattr(self, 'teacher_model') and self.teacher_model is not None:
                self.log(
                    "task_id",
                    self.task_id,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        return final_loss

    def _compute_replay_loss(self):
        """Compute loss on replay samples from memory buffer"""
        # Sample from memory buffer
        sample_indices = random.sample(
            range(len(self.memory_buffer)),
            min(self.replay_batch_size, len(self.memory_buffer))
        )
        memory_samples = [self.memory_buffer[i] for i in sample_indices]

        # Create replay batch
        replay_x = torch.cat([sample['input'] for sample in memory_samples], dim=0).to(self.device)
        replay_y = torch.cat([sample['target'] for sample in memory_samples], dim=0).to(self.device)

        # Forward pass on replay batch
        replay_logits = self(replay_x)
        replay_pred = nn.functional.sigmoid(replay_logits)

        # Regular task loss on replay samples
        replay_loss = self.loss(replay_pred, replay_y)

        # Add knowledge distillation if we have a teacher model
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(replay_x)

            # Distillation loss from teacher
            distill_loss = distillation_loss(
                replay_logits,
                teacher_logits,
                temperature=self.temperature
            )

            # Combine losses
            replay_loss = (1 - self.distill_weight) * replay_loss + self.distill_weight * distill_loss

            # Log distillation loss
            self.log(
                "distill_loss",
                distill_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        # Log replay loss
        self.log(
            "replay_loss",
            replay_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return replay_loss

    def prepare_new_task(self):
        """Prepare the model for a new task by creating a teacher model"""
        # Create a copy of the current model as teacher before adapting to new task
        self.teacher_model = copy.deepcopy(self.model)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

        # Increment task counter
        self.task_id += 1

        logging.info(f"Prepared model for new task {self.task_id}")
        logging.info(f"Memory buffer size: {len(self.memory_buffer)}")

        # Return parameters with different learning rates for optimizer
        # This leverages the existing SwinUNETR ability to use different learning rates
        return True

    def configure_optimizers(self):
        """Configure optimizer with different learning rates after task switch"""
        if self.task_id > 0:
            # We're in a new task - use different learning rates
            return super().configure_optimizers()
        else:
            # First task - use standard learning rate
            return super().configure_optimizers()

    def evaluate_all_tasks(self, datamodules):
        """Evaluate the model on all previous tasks"""
        results = []
        for task_id, datamodule in enumerate(datamodules):
            task_loss = 0
            n_batches = 0

            for batch in datamodule.val_dataloader():
                x, y = self.resolve_batch(batch)
                x, y = x.to(self.device), y.to(self.device)

                with torch.no_grad():
                    logits = self(x)
                    y_pred = nn.functional.sigmoid(logits)
                    loss = self.loss(y_pred, y)

                task_loss += loss.item()
                n_batches += 1

            avg_loss = task_loss / max(1, n_batches)
            results.append({'task_id': task_id, 'loss': avg_loss})

        return results

    def on_save_checkpoint(self, checkpoint):
        """Add memory buffer to checkpoint"""
        super().on_save_checkpoint(checkpoint)

        # Save memory buffer in checkpoint
        # Convert tensors to CPU to avoid issues when loading on different devices
        cpu_memory = []
        for item in self.memory_buffer:
            cpu_memory.append({
                'input': item['input'].cpu(),
                'target': item['target'].cpu(),
                'importance': item['importance'],
                'task_id': item['task_id']
            })

        checkpoint['memory_buffer'] = cpu_memory
        checkpoint['task_id'] = self.task_id

    def on_load_checkpoint(self, checkpoint):
        """Load memory buffer from checkpoint"""
        super().on_load_checkpoint(checkpoint)

        # Restore memory buffer if it exists in checkpoint
        if 'memory_buffer' in checkpoint:
            self.memory_buffer = checkpoint['memory_buffer']
            self.task_id = checkpoint.get('task_id', 0)
            logging.info(f"Restored memory buffer with {len(self.memory_buffer)} samples")