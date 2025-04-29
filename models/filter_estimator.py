from typing import Tuple
import torch
from torch.nn.modules import Linear, ReLU
import torchaudio
from torch import nn, prod, reshape, return_types
import torch.nn.functional as F


class FilterEstimatorModel(nn.Module):
    def __init__(self, input_channels, output_shape: Tuple[int, int]) -> None:
        super().__init__()
        self.d1_conv_block1 = self.grouped_1d_conv_block(input_channels, 32, 64)
        self.d1_conv_block2 = self.grouped_1d_conv_block(32, 64, 32)
        self.d1_conv_block3 = self.grouped_1d_conv_block(64, 128, 32)
        self.d1_conv_block4 = self.grouped_1d_conv_block(128, 256, 32)
        self.d1_conv_block5 = self.grouped_1d_conv_block(256, 256, 32)

        self.conv_block1 = self.grouped_conv_block(256, 256, 256, 3)
        self.conv_block2 = self.grouped_conv_block(256, 512, 512, 3)
        self.conv_block3 = self.grouped_conv_block(512, 512, 512, 3)
        self.conv_block4 = self.grouped_conv_block(512, 512, 512, 3)

        self.output_shape = output_shape
        out_feat = output_shape[0] * output_shape[1]
        self.fl = self.final_linear(out_feat)

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
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            # nn.BatchNorm2d(out_channels),
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
                stride=(3, 1),
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def final_linear(self, output_features):
        return nn.Sequential(
            nn.LazyLinear(out_features=512), nn.ReLU(), nn.Linear(512, output_features, bias=False)
        )

    def forward(self, x):
        # print(f"input test {x[:,1,1,1]}")
        x = self.d1_conv_block1(x)
        # print(f"after convs down 1, sample 0 mean {x[0].mean().item()}, sample 1 mean {x[1].mean().item()}, sample 2 mean {x[2].mean().item()}")
        # print(x.size())
        x = self.d1_conv_block2(x)
        x = self.d1_conv_block3(x)
        x = self.d1_conv_block4(x)
        x = self.d1_conv_block5(x)
        # print(x.size())

        # print(f"after convs down, sample 0 mean {x[0].mean().item()}, sample 1 mean {x[1].mean().item()}, sample 2 mean {x[2].mean().item()}")
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        # print(f"after convs, sample 0 mean {x[0].mean().item()}, sample 1 mean {x[1].mean().item()}, sample 2 mean {x[2].mean().item()}")
        x = torch.flatten(x, start_dim=1)  # Keep batch dim intact
        x = self.fl(x)
        x = x.view(-1, *self.output_shape)  # reshape per batch

        return x
