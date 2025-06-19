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
    A picking model enhanced with continual learning capabilities using
    experience replay and knowledge distillation from a dynamic teacher.

    When starting a new task, a copy of the current model is frozen to act
    as a 'teacher'. During training on the new task, the model (student) is
    trained on a combined loss:
    1. Task Loss: The standard loss on the new data.
    2. Replay Loss: A loss calculated on a small batch of 'replayed' samples
       from a memory buffer of previous tasks. This loss itself is a
       combination of a task loss (remembering) and a distillation loss
       (not forgetting how it learned).
    """

    def __init__(
            self,
            memory_size: int = 1000,
            replay_batch_size: int = 8,
            distill_weight: float = 0.5,
            temperature: float = 2.0,
            # Pass through base model arguments
            *args,
            **kwargs
    ):
        # Call parent constructor first to set up base model, optimizers, etc.
        super().__init__(*args, **kwargs)

        # Use save_hyperparameters to automatically save CL args and make them
        # accessible via self.hparams. This is the modern PyTorch Lightning way.
        self.save_hyperparameters('memory_size', 'replay_batch_size', 'distill_weight', 'temperature')

        # Initialize CL-specific components
        self.memory_buffer = []
        self.teacher_model = None  # The teacher is created dynamically, not loaded here

        logging.info(
            f"Initialized ContinuousLearningModel with memory_size={self.hparams.memory_size}, "
            f"replay_batch_size={self.hparams.replay_batch_size}, "
            f"distill_weight={self.hparams.distill_weight}, temperature={self.hparams.temperature}"
        )

    def _prepare_for_new_task(self):
        """
        Freezes a copy of the current model to serve as the teacher for the
        upcoming task. This is the core of the dynamic teacher mechanism.
        """
        logging.info("New task detected. Creating teacher model from current student state.")
        # Deepcopy the *internal model* (e.g., the SwinUNETR), not the entire LightningModule
        self.teacher_model = copy.deepcopy(self.model)

        # ======================== FIX IS HERE ========================
        # Manually freeze the teacher model's parameters.
        # This is the standard PyTorch way, since self.teacher_model is a torch.nn.Module
        # and does not have the .freeze() method from LightningModule.
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        # =============================================================

        self.teacher_model.eval()   # Set teacher to evaluation mode (disables dropout, etc.)
        logging.info("Teacher model created and frozen.")

    def on_train_epoch_start(self):
        """
        Hook called at the beginning of each training epoch.
        We use this to create the teacher model exactly once when a new task begins.
        """
        # The main 'train.py' script sets this flag to True when continuing from a
        # checkpoint for a new task. Use direct attribute access.
        if self.hparams.config.train.train_new_task:
            self._prepare_for_new_task()
            # Reset the flag so this only runs once per task, not on every epoch
            self.hparams.config.train.train_new_task = False
        # ========================================================================


    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)


        student_logits = self(x)
        student_pred = torch.sigmoid(student_logits)
        task_loss = self.loss(student_pred, y)

        with torch.no_grad():
            sample_loss = nn.functional.huber_loss(student_pred, y, reduction='none').mean(dim=(1, 2, 3, 4))
        self._update_memory(batch, sample_loss)

        replay_loss = torch.tensor(0.0, device=self.device)
        if self.teacher_model and self.memory_buffer:
            # Move teacher to the same device as the student, just in case
            self.teacher_model.to(self.device)
            replay_loss = self._compute_replay_loss()

        final_loss = task_loss + replay_loss

        self.log("loss", final_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.shape[0])
        self.log("loss/task", task_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=x.shape[0])
        if replay_loss > 0:
            self.log("loss/replay_total", replay_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("memory_buffer_size", float(len(self.memory_buffer)), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # 6. --- Visualization ---
        if batch_idx == 0:
            self._log_images(x, y, student_pred)

        return final_loss

    def _update_memory(self, batch, loss_per_sample):
        """Update memory buffer with important samples from the current batch."""
        x, y = self.resolve_batch(batch)
        for i in range(x.shape[0]):
            self.memory_buffer.append({
                'input': x[i:i + 1].cpu(),  # Store on CPU to save GPU memory
                'target': y[i:i + 1].cpu(),
                'importance': loss_per_sample[i].item()
            })

        # Prune buffer to maintain its size, keeping the most 'important' samples
        if len(self.memory_buffer) > self.hparams.memory_size:
            self.memory_buffer.sort(key=lambda item: item['importance'], reverse=True)
            self.memory_buffer = self.memory_buffer[:self.hparams.memory_size]

    def _compute_replay_loss(self):
        """
        Samples a batch from memory, computes a combined loss of remembering the
        old task and distilling knowledge from the old teacher.
        """
        # Sample a mini-batch from the memory buffer
        replay_batch_size = min(self.hparams.replay_batch_size, len(self.memory_buffer))
        memory_samples = random.sample(self.memory_buffer, replay_batch_size)

        # Create replay batch and move to current device
        replay_x = torch.cat([s['input'] for s in memory_samples], dim=0).to(self.device)
        replay_y = torch.cat([s['target'] for s in memory_samples], dim=0).to(self.device)

        # --- Get predictions from student and teacher on replayed data ---
        replay_student_logits = self(replay_x)
        with torch.no_grad():
            replay_teacher_logits = self.teacher_model(replay_x)

        # --- Calculate losses on replayed data ---
        # a) Loss on replayed samples (how well we remember old tasks)
        replay_pred = torch.sigmoid(replay_student_logits)
        memory_task_loss = self.loss(replay_pred, replay_y)

        # b) Distillation loss from teacher (prevents catastrophic forgetting)
        distill_loss = distillation_loss(
            replay_student_logits,
            replay_teacher_logits,
            temperature=self.hparams.temperature
        )

        # Combine replay losses using the distill_weight
        combined_replay_loss = (1 - self.hparams.distill_weight) * memory_task_loss + self.hparams.distill_weight * distill_loss

        # Log the components for better monitoring
        self.log("loss/replay_task", memory_task_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("loss/replay_distill", distill_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return combined_replay_loss

    def on_save_checkpoint(self, checkpoint):
        """Hook to add custom state to the checkpoint."""
        # This is critical for continual learning to work across runs
        checkpoint['memory_buffer'] = self.memory_buffer
        logging.info(f"Saving memory buffer with {len(self.memory_buffer)} samples to checkpoint.")

    def on_load_checkpoint(self, checkpoint):
        """Hook to load custom state from a checkpoint."""
        if 'memory_buffer' in checkpoint:
            self.memory_buffer = checkpoint['memory_buffer']
            logging.info(f"Restored memory buffer with {len(self.memory_buffer)} samples from checkpoint.")

    def _log_images(self, x, y, y_pred):
        """Logs images to TensorBoard (copied from BasePickingModel for consistency)."""
        if self.logger and self.logger.experiment:
            size = y_pred.shape[2]
            grid_params = {"padding": 1, "pad_value": 1.0}
            tensorboard = self.logger.experiment

            grid_input = torchvision.utils.make_grid(x[:, :, size // 2, :, :], **grid_params)
            tensorboard.add_image("input", grid_input, global_step=self.current_epoch)

            grid_label = torchvision.utils.make_grid(y[:, :, size // 2, :, :], **grid_params)
            tensorboard.add_image("label", grid_label, global_step=self.current_epoch)

            grid_pred = torchvision.utils.make_grid(y_pred[:, :, size // 2, :, :], **grid_params)
            tensorboard.add_image("prediction", grid_pred, global_step=self.current_epoch)