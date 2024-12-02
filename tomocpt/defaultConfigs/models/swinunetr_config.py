from dataclasses import dataclass


@dataclass
class SwinUnetrConfig:
    ##### THIS IS CONFIG FOR SWINUNETR
    drop_rate:float = 0.3
    attn_drop_rate:float = 0.3
    dropout_path_rate:float = 0.2
    IN_CHANNELS: int = 1
    OUT_CHANNELS: int=1
    FEATURE_SIZE: int = 12*3  # Should be multiple of 12
    USE_V2:bool = True
    USE_CHECKPOINT:bool = True
