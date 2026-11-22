from dataclasses import dataclass
from typing import Annotated

import typer


@dataclass
class UnetConfig:  # help text only (Gate 10); not command-line flags (cli_options.CLI_FLAGS)
    in_channels: Annotated[int, typer.Option(help='Input channels (one tomogram)')] = 1
    first_layer_out_channels: Annotated[int, typer.Option(help='model_type=Unet: channels of the first level')] = 32
    num_levels: Annotated[int, typer.Option(help='model_type=Unet: number of levels (> 2)')] = 5
    channels_increase_factor: Annotated[int, typer.Option(help='model_type=Unet: channel multiplier from one level to the next')] = 2
    kernel_size: Annotated[int, typer.Option(help='model_type=Unet: convolution kernel size')] = 5
    convolution_dilation: Annotated[int, typer.Option(help='model_type=Unet: convolution dilation')] = 1
    last_activation: Annotated[str, typer.Option(help="model_type=Unet: last activation, 'linear' or 'sigmoid'")] = "linear" #or "sigmoid"
    stride_conv_instead_pooling: Annotated[bool, typer.Option(help='model_type=Unet: strided convolutions instead of pooling')] = False
    output_dim: Annotated[str, typer.Option(help="model_type=Unet: output size ('same' = input size)")] = "same"
