from typing import Dict, Any

from tomocpt.logger import get_logger
from tomocpt.networks.pickingModel import BasePickingModel

logger = get_logger()


class InferenceModel(BasePickingModel):
    """
    A model wrapper for inference that automatically handles loading checkpoints
    from distillation training by filtering out teacher-related weights.
    """

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook called by PyTorch Lightning before loading the state dict.
        It removes teacher weights from the checkpoint in-place.
        """
        state_dict = checkpoint["state_dict"]
        teacher_keys = [k for k in state_dict if k.startswith("teacher.")]
        if teacher_keys:
            logger.info(
                f"Distillation checkpoint detected. Removing {len(teacher_keys)} teacher-related keys for inference."
            )
            for k in teacher_keys:
                del state_dict[k]
