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
    A picking model enhanced with continuous learning capabilities using
    experience replay and knowledge distillation.
    """

    def __init__(
            self,
            memory_size=1000,
            replay_batch_size=8,
            distill_weight=0.5,
            temperature=2.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.memory_buffer = []
        self.memory_size = memory_size
        self.replay_batch_size = replay_batch_size
        self.distill_weight = distill_weight
        self.temperature = temperature
        self.teacher_model = None

        logging.info(f"Initialized ContinuousLearningModel with memory size {self.memory_size}")

    def update_memory(self, batch, loss_per_sample):
        """Update memory buffer with samples from the current batch."""
        x, y = self.resolve_batch(batch)
        for i in range(x.shape[0]):
            self.memory_buffer.append({
                'input': x[i:i + 1].cpu(),  # Store on CPU to save GPU memory
                'target': y[i:i + 1].cpu(),
                'importance': loss_per_sample[i].item()
            })

        # Prune buffer to maintain its size, keeping most important samples
        if len(self.memory_buffer) > self.memory_size:
            self.memory_buffer.sort(key=lambda x: x['importance'], reverse=True)
            self.memory_buffer = self.memory_buffer[:self.memory_size]

    def training_step(self, batch, batch_idx):
        # 1. Regular training on the current task's batch
        x, y = self.resolve_batch(batch)
        student_logits = self(x)
        y_pred = torch.sigmoid(student_logits)
        task_loss = self.loss(y_pred, y)

        # 2. Update memory buffer with important samples from current batch
        with torch.no_grad():
            sample_loss = nn.functional.huber_loss(y_pred, y, reduction='none').mean([1, 2, 3, 4])
        self.update_memory(batch, sample_loss)

        # 3. Perform experience replay and distillation if a teacher is available
        replay_loss = 0.0
        if self.teacher_model and self.memory_buffer:
            replay_loss = self._compute_replay_loss()

        # 4. Combine losses
        final_loss = task_loss + replay_loss

        # Logging
        self.log("loss", final_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.shape[0])
        self.log("loss/task", task_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])
        if replay_loss > 0:
            self.log("loss/replay", replay_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                     batch_size=x.shape[0])
        self.log("memory_buffer_size", float(len(self.memory_buffer)), on_step=False, on_epoch=True, prog_bar=False,
                 sync_dist=True)

        # Visualization (same as base model)
        if batch_idx == 0:
            self._log_images(x, y, y_pred)

        return final_loss

    def _compute_replay_loss(self):
        """Compute loss on a batch of replay samples from the memory buffer."""
        # Sample a mini-batch from the memory buffer
        replay_batch_size = min(self.replay_batch_size, len(self.memory_buffer))
        memory_samples = random.sample(self.memory_buffer, replay_batch_size)

        # Create replay batch and move to current device
        replay_x = torch.cat([s['input'] for s in memory_samples], dim=0).to(self.device)
        replay_y = torch.cat([s['target'] for s in memory_samples], dim=0).to(self.device)

        # Move teacher model to the current device
        self.teacher_model.to(self.device)

        # Forward pass on replay batch
        replay_logits = self(replay_x)
        replay_pred = torch.sigmoid(replay_logits)

        # Loss on replayed samples (how well we remember old tasks)
        memory_task_loss = self.loss(replay_pred, replay_y)

        # Distillation loss from teacher (prevents catastrophic forgetting)
        with torch.no_grad():
            teacher_logits = self.teacher_model(replay_logits.shape, device=self.device)  # Dummy input to get logits

        distill_loss = distillation_loss(
            replay_logits,
            teacher_logits,
            temperature=self.temperature
        )

        # Combine replay losses
        combined_replay_loss = (1 - self.distill_weight) * memory_task_loss + self.distill_weight * distill_loss
        return combined_replay_loss

    def on_train_epoch_start(self):
        """Called at the beginning of a training epoch."""
        # Check if we should start a new task
        if self.hparams.config.train.train_new_task:
            self._prepare_for_new_task()
            # Reset flag to avoid re-initializing on every epoch
            self.hparams.config.train.train_new_task = False

    def _prepare_for_new_task(self):
        """Freezes a copy of the current model to serve as the teacher."""
        logging.info("Preparing for new task. Creating teacher model from current student.")
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.freeze()
        self.teacher_model.eval()

    def _log_images(self, x, y, y_pred):
        """Logs images to TensorBoard."""
        size = y_pred.shape[2]
        grid_params = {"padding": 1, "pad_value": 1.0}
        tensorboard = self.logger.experiment

        grid_input = torchvision.utils.make_grid(x[:, :, size // 2, :, :], **grid_params)
        tensorboard.add_image("input", grid_input, global_step=self.current_epoch)

        grid_label = torchvision.utils.make_grid(y[:, :, size // 2, :, :], **grid_params)
        tensorboard.add_image("label", grid_label, global_step=self.current_epoch)

        grid_pred = torchvision.utils.make_grid(y_pred[:, :, size // 2, :, :], **grid_params)
        tensorboard.add_image("prediction", grid_pred, global_step=self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        """Save memory buffer to checkpoint."""
        checkpoint['memory_buffer'] = self.memory_buffer

    def on_load_checkpoint(self, checkpoint):
        """Load memory buffer from checkpoint."""
        if 'memory_buffer' in checkpoint:
            self.memory_buffer = checkpoint['memory_buffer']
            logging.info(f"Restored memory buffer with {len(self.memory_buffer)} samples.")