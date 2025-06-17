from dataclasses import dataclass


@dataclass
class SwinUnetrConfig:
    ##### THIS IS CONFIG FOR SWINUNETR
    drop_rate:float = 0.15
    attn_drop_rate:float = 0.15
    dropout_path_rate:float = 0.05
    in_channels: int = 1
    out_channels: int=1
    feature_size: int = 12 * 3  # Should be multiple of 12
    use_v2:bool = True
    use_checkpoint:bool = True
