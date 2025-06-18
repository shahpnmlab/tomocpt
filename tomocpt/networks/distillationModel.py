import torch
import torchvision

from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.training.losses import distillation_loss
from tomocpt.logger import get_logger

logging = get_logger()


class DistillationPickingModel(BasePickingModel):
    """
    A picking model that uses knowledge distillation for fine-tuning.

    This model is trained with a combination of the standard task loss (e.g.,
    on new labeled data) and a distillation loss. The distillation loss
    encourages the student model's logits to match the logits of a frozen,
    pre-trained teacher model, effectively transferring knowledge and
    preventing catastrophic forgetting.
    """

    def __init__(
            self,
            teacher_checkpoint_path,
            distill_weight=0.5,
            temperature=2.0,
            lr=None,
            model=None,
            config_dict=None,
            *args,
            **kwargs
    ):
        # Save distillation parameters before calling parent constructor
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.distill_weight = distill_weight
        self.temperature = temperature

        # Continue with normal initialization
        super().__init__(lr, model, config_dict, *args, **kwargs)

        # Load teacher model from checkpoint and freeze it
        logging.info(f"Loading teacher model from {self.teacher_checkpoint_path}")
        self.teacher = BasePickingModel.load_from_checkpoint(self.teacher_checkpoint_path)
        self.teacher.freeze()  # Freeze all parameters of the teacher model
        self.teacher.eval()  # Set teacher to evaluation mode

        logging.info(
            f"Initialized DistillationPickingModel with distill_weight={self.distill_weight}, temperature={self.temperature}")

    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)

        # Move teacher to the same device as student
        self.teacher.to(self.device)

        # Get student predictions (logits)
        student_logits = self(x)

        # Get teacher predictions (logits)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Calculate task loss (original loss function on sigmoid output)
        student_pred = torch.sigmoid(student_logits)
        task_loss = self.loss(student_pred, y)

        # Calculate distillation loss on raw logits
        distill_loss = distillation_loss(
            student_logits,
            teacher_logits,
            temperature=self.temperature
        )

        # Combined loss
        combined_loss = (1 - self.distill_weight) * task_loss + self.distill_weight * distill_loss

        # Logging
        self.log("loss/task", task_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=x.shape[0])
        self.log("loss/distill", distill_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
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