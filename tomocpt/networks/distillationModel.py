import torch
import torchvision
from torch.nn import functional as F

from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.logger import get_logger

logging = get_logger()


class DistillationPickingModel(BasePickingModel):
    """
    A picking model that uses feature-based knowledge distillation for fine-tuning.

    This model is trained with a combination of the standard task loss (e.g.,
    on new labeled data) and a feature-based distillation loss. The distillation
    loss encourages the student model's intermediate feature representations to
    match those of a frozen, pre-trained teacher model, effectively
    transferring knowledge and preventing catastrophic forgetting.
    """

    def __init__(
            self,
            teacher_checkpoint_path,
            feature_distill_weight=0.5,
            lr=None,
            model=None,
            config_dict=None,
            *args,
            **kwargs
    ):
        # Save distillation parameters before calling parent constructor
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.feature_distill_weight = feature_distill_weight

        # Continue with normal initialization
        super().__init__(lr, model, config_dict, *args, **kwargs)

        # Load teacher model from checkpoint and freeze it
        logging.info(f"Loading teacher model from {self.teacher_checkpoint_path}")
        self.teacher = BasePickingModel.load_from_checkpoint(self.teacher_checkpoint_path)
        self.teacher.freeze()  # Freeze all parameters of the teacher model
        self.teacher.eval()  # Set teacher to evaluation mode

        logging.info(
            f"Initialized DistillationPickingModel with feature_distill_weight={self.feature_distill_weight}")

    def forward(self, x):
        """
        Override the parent's forward to return the full model output (tuple),
        bypassing the logic that discards the hidden state.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)

        # Move teacher to the same device as student
        self.teacher.to(self.device)

        # Get student predictions (logits and hidden state)
        student_logits, student_hidden = self(x)

        # Get teacher predictions (logits and hidden state)
        with torch.no_grad():
            # We call the teacher's underlying model directly to get the tuple output
            teacher_logits, teacher_hidden = self.teacher.model(x)

        # 1. Calculate task loss (original loss function on sigmoid output)
        student_pred = torch.sigmoid(student_logits)
        task_loss = self.loss(student_pred, y)

        # 2. Calculate feature-based distillation loss (MSE on hidden states)
        feature_loss = F.mse_loss(student_hidden, teacher_hidden)

        # 3. Combined loss
        combined_loss = (1 - self.feature_distill_weight) * task_loss + self.feature_distill_weight * feature_loss

        # Logging
        self.log("loss/task", task_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])
        self.log("loss/feature", feature_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])
        self.log("loss", combined_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])

        # Visualization code (unchanged from BasePickingModel)
        if batch_idx == 0:
            size = student_pred.shape[2]
            grid_params = {"padding": 1, "pad_value": 1.0}

            tensorboard = self.logger.experiment

            grid_input = torchvision.utils.make_grid(x[:, :, size // 2, :, :], **grid_params)
            tensorboard.add_image("input", grid_input, global_step=self.current_epoch)

            grid_label = torchvision.utils.make_grid(y[:, :, size // 2, :, :], **grid_params)
            tensorboard.add_image("label", grid_label, global_step=self.current_epoch)

            grid_pred = torchvision.utils.make_grid(student_pred[:, :, size // 2, :, :], **grid_params)
            tensorboard.add_image("prediction", grid_pred, global_step=self.current_epoch)

            # Additionally visualize teacher predictions
            with torch.no_grad():
                teacher_pred = torch.sigmoid(teacher_logits)
                grid_teacher = torchvision.utils.make_grid(
                    teacher_pred[:, :, size // 2, :, :], **grid_params
                )
                tensorboard.add_image("teacher_prediction", grid_teacher, global_step=self.current_epoch)

        return combined_loss