import torch
import torchaudio
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_seq_len, input_channels, n_impulse, out_impulse_len) -> None:
        super().__init__()
        self.d1_conv_block1 = self.grouped_1d_conv_block(input_channels, 32, 32)
        self.d1_conv_block2 = self.grouped_1d_conv_block(32, 64, 32)
        self.conv_block1 = self.grouped_conv_block(64, 64, 64, 3)
        self.conv_block2 = self.grouped_conv_block(64, 64, 128, 3)
        self.conv_block3 = self.grouped_conv_block(128, 128, 128, 3)
        self.conv_block4 = self.grouped_conv_block(128, 128, 128, 3)

        # input shape should be [[seq_length, n_speakers], [seq_length, n_mics]] ->
        # thus (C, H, W) = (2, seq_lenght, 3) for the phone

    def grouped_conv_block(self, in_channels, mid_channels, out_channels, kernelsize):
        """Apply grouped convolution and 1x1 convolution"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernelsize,
                groups=in_channels,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
        )

    def grouped_1d_conv_block(self, in_channels, out_channels, kernelsize):
        """apply grouped 1d convolutions, achived with 1d kernels in 2d space acros the sequence dimension"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernelsize, 1),
                groups=in_channels,
                padding=0,
                stride=16,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        return x
