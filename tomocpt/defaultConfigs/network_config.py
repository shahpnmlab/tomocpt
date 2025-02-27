from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Annotated, Dict, Any

import typer

from tomocpt.defaultConfigs.models.swinunetr_config import SwinUnetrConfig
from tomocpt.defaultConfigs.models.unet_config import UnetConfig


class ModelType(str, Enum):
    UNET = "Unet"
    SwinUNETR = "SwinUNETR"

    @property
    def module_path(self) -> str:
        targets = {
            ModelType.UNET: "tomocpt.networks.unet",
            ModelType.SwinUNETR: "tomocpt.networks.swinunetr",
        }
        return targets[self]

    def get_kwargs(self, parent_config: Any) -> Dict[str, Any]:
        return asdict(getattr(parent_config, self.value))


class PrecisionType(str, Enum):
    bf16 = "bf16-mixed"
    float = "32"
    half = "16"


@dataclass
class NetworkConfig:

    model_type: Annotated[
        ModelType, typer.Option(help="The model type", case_sensitive=False)
    ] = ModelType.SwinUNETR

    UNET: UnetConfig = field(default_factory=UnetConfig)
    SwinUNETR: SwinUnetrConfig = field(default_factory=SwinUnetrConfig)

    # THIS IS FOR PRETRAIN (self supervised).
    NOISE2NOISE_RAND_MASK_SIZE: int = 15
    CONTRAST_LOSS_L1_EMB_REGULARIZATION: float = 1e-4
    CONTRAST_LOSS_TEMPERATURE: float = 1e-1

    TORCH_MATMUL_PRECISION: str = "medium"  # "medium" "high" "highest"
    TORCH_FLOAT_PRECISION: Annotated[
        PrecisionType,
        typer.Option(help="The precision for gpu computing", case_sensitive=False),
    ] = PrecisionType.bf16

    SELF_SUPERVISED_EMBEDING_SIZE: int = 256

    def build_model(self, require_labels=False, **kwargs):
        import importlib

        model_type = self.model_type
        model_name = str(model_type.value)
        if model_name.startswith("SwinUNETR"):
            if require_labels:
                model_name = "MySwinUNETR"
            # if version.parse(monai.__version__) > version.parse("1.3"):
            #     kwargs.pop("img_size")

        config_kwargs = asdict(getattr(self, model_type.value))
        module_path = model_type.module_path
        module = importlib.import_module(module_path)
        target_cls = getattr(module, model_name)
        config_kwargs.update(kwargs)
        return model_name, target_cls(**config_kwargs)
