from typing import Literal

import torch
from tomocpt import constants
import torch.nn as nn

NORM_LAYER = nn.InstanceNorm3d


# Calculate symmetric padding for vol convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class StridedConvDown(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=2):
        super().__init__()
        self.padding = [get_padding(kernel_size, stride, dilation=1)]*3
        self.conv = nn.Conv3d(channels, channels, kernel_size, stride=stride,
                              padding=self.padding,
                              bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1, perform_downsample=True,
                 stride_conv_instead_pooling=False):
        super(DownBlock, self).__init__()
        self.dilation = dilation

        self.perform_downsample = perform_downsample

        conv1 = nn.Conv3d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                          dilation=self.dilation,
                          padding="same")
        bn1 = NORM_LAYER(out_channels)
        actv1 = nn.PReLU()

        conv2 = nn.Conv3d(out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                          dilation=self.dilation,
                          padding="same")
        bn2 = NORM_LAYER(out_channels)
        actv2 = nn.PReLU()
        layers = [conv1, bn1, actv1, conv2, bn2, actv2]
        if perform_downsample:
            if stride_conv_instead_pooling:
                layers += [nn.AvgPool3d(2)]
            else:
                layers += [StridedConvDown(out_channels, kernel_size=kernel_size, stride=2)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x_ori = x
        x = self.block(x)
        return x_ori, x


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(UpBlock, self).__init__()
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        ups = nn.Upsample(size=None, scale_factor=2, mode='trilinear', align_corners=None, recompute_scale_factor=None)
        # Not needed, per se but it adds more parameters that could help.
        conv_pre = nn.Conv3d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same",
                          dilation=self.dilation)

        self.blueBlock = nn.Sequential(ups, conv_pre)

        # HERE YOU CONCATENATE, so you end up having 2*out_channels

        # half_channels = in_channels // 2 # You do outside
        # conv0 is the firs tof red blocks on the right hand side
        conv0 = nn.Conv3d(2 * out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same",
                          dilation=self.dilation)
        bn0 = NORM_LAYER(in_channels)
        actv0 = nn.PReLU()

        # conv1 is the second red block in the right arm
        conv1 = nn.Conv3d(out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same",
                          dilation=self.dilation)
        # in_channels = half_channels You do outside
        bn1 = NORM_LAYER(out_channels)
        actv1 = nn.PReLU()
        self.redBlock = nn.Sequential(conv0, bn0, actv0, conv1, bn1, actv1)

    def forward(self, x_prev, x_to_concat):
        x_prev = self.blueBlock(x_prev)
        x = torch.cat([x_to_concat, x_prev], 1)
        x = self.redBlock(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_channels:int=1, first_layer_out_channels:int= constants.FIRST_LAYER_OUT_CHANNELS,
                 num_levels=constants.NUM_LEVELS, channels_increase_factor=constants.CHANNELS_INCREASE_FACTOR,
                 kernel_size=constants.KERNEL_SIZE, last_activation:Literal["linear", "sigmoid"]= "linear",
                 convolution_dilation=constants.CONV_DILATION,
                 stride_conv_instead_pooling= False,
                 output_dim="same",
                 **kwargs):
        """

        :param in_channels: Number of channels of the input, probably 1
        :param first_layer_out_channels:
        :param num_levels: number of "blocks of layers"
        :param channels_increase_factor: change in the number of channels after moving to next level
        :param kernel_size: size of the conv kernel
        :param last_activation: The name of the activation layer
        :param convolution_dilation: dilation of the convolution
        :param output_dim: if same, same as input
        :param kwargs:
        """
        super(Unet, self).__init__()
        self.in_channels_ori = in_channels
        self.in_channels = in_channels
        self.out_channels = first_layer_out_channels
        self.num_blocks = num_levels
        assert num_levels > 2, f"Error, num_levels>2 required. Current {num_levels}"

        self.leftarm = nn.ModuleList()
        # The right arm of UNET does 3 operations in seq
        # Upsampling the previous block -> concatenation from the corrspnding left arm of the UNET
        # and finally vol convolution in which the channesl will be halved dimensions will be preserverd.
        # Thus the right arm of the UNET follows -

        self.upsampling_blocks = nn.ModuleList()

        self.first_layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=first_layer_out_channels, kernel_size=kernel_size, stride=1,
                    dilation=convolution_dilation, padding="same"),
            NORM_LAYER(first_layer_out_channels),
            nn.PReLU()
        )

        in_channels = first_layer_out_channels
        #Left arm of U-NET
        layers_in_channels = []
        layers_out_channels = []
        for level in range(num_levels):
            block = DownBlock(in_channels, first_layer_out_channels, kernel_size, dilation=convolution_dilation,
                              perform_downsample=level < num_levels - 1,
                              stride_conv_instead_pooling=stride_conv_instead_pooling)
            layers_in_channels.append(in_channels)
            layers_out_channels.append(first_layer_out_channels)
            self.leftarm.append(block)
            in_channels = first_layer_out_channels
            #Middle of UNET
            # The IF prevents the middle of the UNET being halved.
            if level < num_levels-2:
                first_layer_out_channels = int(first_layer_out_channels * channels_increase_factor)
            else:
                first_layer_out_channels = in_channels

        layers_in_channels = layers_in_channels[:-1][::-1]
        layers_out_channels = layers_out_channels[:-1][::-1]
        #Right arm of UNET
        for level in range(num_levels - 1): #We have N down 1 same and N-1 up
            block = UpBlock(layers_out_channels[level], layers_in_channels[level], kernel_size, dilation=convolution_dilation)
            self.upsampling_blocks.append(block)
        in_channels = layers_in_channels[-1]
        if output_dim == "same":
            out_channels = self.in_channels_ori
        else:
            out_channels = output_dim
        _last_layers=[nn.Conv3d(in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding="same"),
            nn.PReLU(),
            nn.Conv3d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding="same"),
            ]

        if last_activation == "linear":
            pass
        elif last_activation == "sigmoid":
            _last_layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"Wrong last_activation, {last_activation}")

        self.last_layer = nn.Sequential(*_last_layers)


    def forward(self, x):

        x = self.first_layer(x)
        last_layer_activations = []
        for i, block in enumerate(self.leftarm):
            xpre, xdown = block(x)
            x = xdown
            if i < self.num_blocks-1:
                last_layer_activations.append(xpre)
        hidden_repr = last_layer_activations[-1]
        last_layer_activations = last_layer_activations[::-1] # this is the same as reversed(last_layer_activations)
        for i, block in enumerate(self.upsampling_blocks):
            x = block(x, last_layer_activations[i])
        x = self.last_layer(x)
        return x, hidden_repr


if __name__ == "__main__":

    unet = Unet(in_channels=1, first_layer_out_channels=8, num_levels=4, kernel_size=3,
                stride_conv_instead_pooling=True)
    x = torch.randn(2,1,64,64,64)
    out = unet(x)
    #print(x.shape, out.shape)
    print(x.shape)