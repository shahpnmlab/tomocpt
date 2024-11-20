from dataclasses import dataclass


@dataclass
class UnetConfig: #I am not annotating any of the fields because I do not want to overcomplicate stuff to be exposed in CLI.
    in_channels: int = 1
    first_layer_out_channels: int = 32
    num_levels: int = 5
    channels_increase_factor: int = 2
    kernel_size: int = 5
    convolution_dilation: int = 1
    last_activation: str = "linear" #or "sigmoid"
    stride_conv_instead_pooling: bool = False
    output_dim: str = "same"
