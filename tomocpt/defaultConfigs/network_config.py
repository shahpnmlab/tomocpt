
from dataclasses import dataclass, asdict, fields, field
from enum import Enum
from typing import Annotated, Union, Optional

import typer


class ModelType(str, Enum):
    UNET = "UNET"
    SwinUNETR = "SwinUNETR"

@dataclass
class ArtchConfig:
    model_type: Annotated[ModelType, typer.Option(help="Type of model")] = ModelType.UNET
    model_kwargs: Annotated[Optional[str], typer.Option(
        help='Additional model arguments as JSON string. Example: \'{"depth":5,"features":64}\''
    )] = None


@dataclass
class NetworkConfig:
    CHUNK_SIZE: Annotated[int, typer.Option(help="The patch size of the cubes")]  = 64
    CHUNK_STRIDE: int = 32 #TODO: Pranav. Is CHUNK STRIDE USED IN TRAINING? I though it was only in inference. If so, remove this
    RANDOM_FRACTION_TO_SAMPLE_TRAIN: float = -1.  # Train on all the chunks #TODO: Move this to train or to prep

    model_type: Annotated[ModelType, typer.Option(help="The model type", case_sensitive=False)] = ModelType.SwinUNETR
    # model_kwargs: Annotated[Optional[str], typer.Option(help='Additional model arguments as JSON string. Example: \'{"depth":5,"features":64}\'')] = None

    # model_arch: Annotated[Union[UNetConfig, ResNetConfig], typer.Option(help="Experimental model artchitecture")] = field(default_factory=UNetConfig)

    #### THIS IS CONFIG FOR U-NET #TODO: Should we want independent confi files for U-net and swin?
    IN_CHANNELS: int = 1
    FIRST_LAYER_OUT_CHANNELS: int = 32
    NUM_LEVELS: int = 5
    CHANNELS_INCREASE_FACTOR: int = 2
    KERNEL_SIZE: int = 5
    CONV_DILATION: int = 1
    LAST_ACTIVATION_LAYER: str = "linear" #or "sigmoid"
    STRIDE_CONV_INSTEAD_OF_POOLING: bool = False
    OUTPUT_DIM: str = "same"
    FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS:float = 0.5 #TODO: Move to train_config
    COSINE_LR_SCHEDULE_N_EPOCHS: int = 6 #TODO: Move to train_config
    PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS: int = 6 #TODO: Move to train_config

    ##### THIS IS CONFIG FOR SWINUNETR
    SWINUNETR_FEAT_SIZE: int = 12*3  # Should be multiple of 12
    DROP_RATE:float = 0.3
    ATTN_DROP_RATE:float = 0.3
    DROPOUT_PATH_RATE:float = 0.2

    # THIS IS FOR PRETRAIN
    NOISE2NOISE_RAND_MASK_SIZE: int = 15
    CONTRAST_LOSS_L1_EMB_REGULARIZATION:float = 1e-4
    CONTRAST_LOSS_TEMPERATURE: float = 1e-1

    #BCE Loss val
    BCE_EPS: int = 1 #TODO: Not used, should it be removed?

    #FASTER TRAINING ARGS
    LIMIT_VALIDATION_BATCHES: int = 1
    VAL_CHECK_INTERVAL: int = 1

    TORCH_MATMUL_PRECISION: str = "medium" # "medium" "high" "highest"
    TORCH_FLOAT_PRECISION: str = '32' # '32' # '16' 'bf16'

    SEED_FOR_TRAIN_VAL_SPLIT: int = 113


