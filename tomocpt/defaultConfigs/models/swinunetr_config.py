from dataclasses import dataclass
from typing import Annotated

import typer


@dataclass
class SwinUnetrConfig:
    ##### THIS IS CONFIG FOR SWINUNETR
    drop_rate: Annotated[float, typer.Option(help='SwinUNETR dropout rate')] = 0.3
    attn_drop_rate: Annotated[float, typer.Option(help='SwinUNETR attention dropout rate')] = 0.3
    dropout_path_rate: Annotated[float, typer.Option(help='SwinUNETR stochastic depth (drop path) rate')] = 0.2
    in_channels: Annotated[int, typer.Option(help='Input channels (one tomogram)')] = 1
    out_channels: Annotated[int, typer.Option(help='Output channels of a pipeline-v1 model; pipeline v2 sets one per class-table row')] = 1
    feature_size: Annotated[int, typer.Option(help='SwinUNETR width (multiple of 12); must match a checkpoint to fine-tune from')] = 12 * 3  # Should be multiple of 12
    use_v2: Annotated[bool, typer.Option(help='MONAI SwinUNETR v2 (extra residual convolution blocks in the encoder)')] = True
    use_checkpoint: Annotated[bool, typer.Option(help='Gradient checkpointing of the Swin blocks; with True it is switched off automatically when memory allows (picking on a GPU); False never checkpoints')] = True
