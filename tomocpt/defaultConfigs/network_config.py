from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Annotated, Dict, Any, Type

import typer

from tomocpt.defaultConfigs.models.swinunetr_config import SwinUnetrConfig
from tomocpt.defaultConfigs.models.unet_config import UnetConfig


class ModelType(str, Enum):
    UNET = "UNET"
    SwinUNETR = "SwinUNETR"

    @property
    def target(self) -> str:
        targets = {
            ModelType.UNET: "tomocpt.networks.unet.Unet",
            ModelType.SwinUNETR: "tomocpt.networks.swinunetr.SwinUNETR"
        }
        return targets[self]

    def get_kwargs(self, parent_config: Any) -> Dict[str, Any]:
        return asdict(getattr(parent_config, self.value))


@dataclass
class NetworkConfig:

    model_type: Annotated[ModelType, typer.Option(help="The model type", case_sensitive=False)] = ModelType.SwinUNETR

    UNET: UnetConfig = field(default_factory=UnetConfig)
    SwinUNETR: SwinUnetrConfig = field(default_factory=SwinUnetrConfig)

    # THIS IS FOR PRETRAIN (self supervised).
    NOISE2NOISE_RAND_MASK_SIZE: int = 15
    CONTRAST_LOSS_L1_EMB_REGULARIZATION: float = 1e-4
    CONTRAST_LOSS_TEMPERATURE: float = 1e-1

    TORCH_MATMUL_PRECISION: str = "medium"  # "medium" "high" "highest"
    TORCH_FLOAT_PRECISION: str = '32'  # '32' # '16' 'bf16'

    SELF_SUPERVISED_EMBEDING_SIZE: int = 256

    def build_model(self, **kwargs):
        import importlib
        model_type = self.model_type
        model_name = str(model_type.value)
        config_kwargs = asdict(getattr(self, model_type.value))
        target_path = model_type.target
        module_path, class_name = target_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_cls = getattr(module, class_name)
        config_kwargs.update(kwargs)
        print("model config", config_kwargs)
        return model_name, target_cls(**config_kwargs)


network_config = NetworkConfig()
