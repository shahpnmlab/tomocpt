import torch
from torch import nn
import torchvision

from tomocpt.networks.pickingModel import BasePickingModel
from tomocpt.training.losses import distillation_loss
from tomocpt.logger import get_logger

logging = get_logger()


class DistillationPickingModel(BasePickingModel):
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
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        logging.info(
            f"Initialized DistillationPickingModel with distill_weight={self.distill_weight}, temperature={self.temperature}")

    def training_step(self, batch, batch_idx):
        x, y = self.resolve_batch(batch)

        # Get student predictions
        student_logits = self(x)
        student_pred = nn.functional.sigmoid(student_logits)

        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Calculate task loss (original loss function)
        task_loss = self.loss(student_pred, y)

        # Calculate distillation loss
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

        # Visualization code (keeping the same as original BasePickingModel)
        if batch_idx == 0:
            size = student_pred.shape[2]
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
                student_pred[:, :, size // 2, :, :].to(dtype=torch.float32), **grid_params
            )
            tensorboard.add_image(
                "prediction",
                grid.to(dtype=torch.float32),
                global_step=self.current_epoch,
            )

            # Additionally visualize teacher predictions
            with torch.no_grad():
                teacher_pred = nn.functional.sigmoid(teacher_logits)
                grid = torchvision.utils.make_grid(
                    teacher_pred[:, :, size // 2, :, :].to(dtype=torch.float32), **grid_params
                )
                tensorboard.add_image(
                    "teacher_prediction",
                    grid.to(dtype=torch.float32),
                    global_step=self.current_epoch,
                )

        return combined_loss