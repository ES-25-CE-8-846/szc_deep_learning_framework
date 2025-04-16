from typing import Tuple
import torch
from torch.nn.modules import Conv2d, Linear, ReLU
import torchaudio
from torch import nn, prod, reshape, return_types
import torch.nn.functional as F


class ImpulseEstimatorModel(nn.Module):
    def __init__(self, input_channels, n_down_convs = 6) -> None:
        super().__init__()

        self.d1_conv_block1 = self.grouped_1d_conv_block(input_channels, 16, 32)
        self.d1_conv_block2 = self.grouped_1d_conv_block(16, 16, 32)
    
        input_channels = 16 
        self.down_conv_list = nn.ModuleList()
        
        for _ in range(n_down_convs):
            output_channels = input_channels*2
            self.down_conv_list.append(self.grouped_conv_block(input_channels,output_channels ,output_channels,3))
            input_channels = output_channels
        print(f"actual max channels {input_channels}")    
        n_up_convs = n_down_convs 
        self.up_convs_list = nn.ModuleList()
    
        for _ in range(n_up_convs):
            output_channels = int(input_channels // 2)
            self.up_convs_list.append(self.up_conv_block(input_channels, output_channels))
            input_channels = output_channels
        
        self.final_up_convs = nn.Sequential(
            nn.ConvTranspose2d(output_channels, 32, kernel_size=(64,9)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(64, 7))
        )

        # input shape should be [[seq_length, n_speakers], [seq_length, n_mics]] ->
        # thus (C, H, W) = (2, seq_lenght, 3) for the phone

    def grouped_conv_block(self, in_channels, mid_channels, out_channels, kernelsize):
        """apply grouped convolution and 1x1 convolution"""
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
                stride=(4,1),
            ),
            nn.ReLU(),
        )

    def up_conv_block(self, in_channels, out_channels):
        """apply transposed convolution to upsample"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(8, 3), stride=(3,1)),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU()
                )

    def forward(self, x):
        x = self.d1_conv_block1(x)
        x = self.d1_conv_block2(x)
        print(x.size())
        for down_conv in self.down_conv_list:
            x = down_conv(x)
        print(x.size())
        for up_conv in self.up_convs_list:
            x = up_conv(x)
        print(x.size())
        x = self.final_up_convs(x)
        return x
