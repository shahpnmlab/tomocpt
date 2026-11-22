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
    NOISE2NOISE_RAND_MASK_SIZE: Annotated[int, typer.Option(help='Unused (kept for config compatibility)')] = 15
    CONTRAST_LOSS_L1_EMB_REGULARIZATION: Annotated[float, typer.Option(help='Self-supervised v1: weight of the L1 penalty on the contrastive embeddings')] = 1e-4
    CONTRAST_LOSS_TEMPERATURE: Annotated[float, typer.Option(help='Self-supervised: temperature of the NT-Xent contrastive loss')] = 1e-1

    TORCH_MATMUL_PRECISION: Annotated[str, typer.Option(help="torch.set_float32_matmul_precision: 'medium', 'high' or 'highest' (TF32 use on Ampere and newer GPUs)")] = "medium"  # "medium" "high" "highest"
    TORCH_FLOAT_PRECISION: Annotated[
        PrecisionType,
        typer.Option(help="The precision for gpu computing", case_sensitive=False),
    ] = PrecisionType.bf16

    SELF_SUPERVISED_EMBEDING_SIZE: Annotated[int, typer.Option(help='Self-supervised: size of the global contrastive projection')] = 256

    scale_conditioning: Annotated[
        bool,
        typer.Option(help="FiLM conditioning of the decoder on the working voxel size (SwinUNETR, pipeline v2)"),
    ] = False

    def build_model(self, require_labels=False, **kwargs):
        import importlib

        from tomocpt.configManager.configManager import safe_getattr

        model_type = self.model_type
        model_name = str(model_type.value)
        scale_conditioning = bool(safe_getattr(self, "scale_conditioning", False))
        if scale_conditioning and not model_name.startswith("SwinUNETR"):
            raise NotImplementedError("scale conditioning needs SwinUNETR")
        if model_name.startswith("SwinUNETR"):
            if require_labels:
                model_name = "ScaleConditionedSwinUNETR" if scale_conditioning else "MySwinUNETR"
            # if version.parse(monai.__version__) > version.parse("1.3"):
            #     kwargs.pop("img_size")

        config_kwargs = asdict(getattr(self, model_type.value))
        module_path = model_type.module_path
        module = importlib.import_module(module_path)
        target_cls = getattr(module, model_name)
        config_kwargs.update(kwargs)
        return model_name, target_cls(**config_kwargs)
