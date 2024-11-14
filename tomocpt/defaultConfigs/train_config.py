from dataclasses import dataclass, fields, MISSING
from enum import Enum
from typing import Optional


class TrainingModes(Enum):
    selfSupervised = "selfSupervised"
    picking = "picking"

@dataclass
class TrainConfig:
    chunks_dir: Optional[str] = None #"/tmp/refactor/chunks/"
    model_dir: Optional[str] = None  #"/tmp/model"
    experiment_name: str = "unnamed"
    learning_rate: float = 4e-4
    n_epochs: int = 10
    mode: TrainingModes = TrainingModes.picking
    restore_full_state: bool = True
    batch_size: int = 2
    use_cuda: bool = True

    OVERFIT_N_BATCHES: Optional[int] = 10
    WORKERS_FOR_DATA: int = 0
    N_GPUS: int = 1
    N_CPUS_IF_NO_GPU: int = 1
    USE_CUDA_FOR_DATA: bool = False

    WEIGHT_DECAY: float = 1e-8

    def write_yaml(self) -> str: #TODO: This has to be moved to a baseclass
        """Convert the dataclass configuration into a YAML-formatted string.

        Returns:
            str: YAML-formatted string representation of the configuration
        """
        import yaml

        # Convert dataclass to dictionary
        config_dict = {}
        for field in fields(self):
            value = getattr(self, field.name)

            # Handle special cases
            if isinstance(value, TrainingModes):
                value = value.name  # Convert enum to string

            config_dict[field.name] = value

        # Convert to YAML string with proper indentation
        yaml_string = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        return yaml_string