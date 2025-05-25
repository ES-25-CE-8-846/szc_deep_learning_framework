
import torch
from torch import nn

class AudioFilterEstimatorFreq(nn.Module):
    """
    Estimates complex-valued filters in the frequency domain.
    Output: complex tensor [B, N, F] suitable for torch.fft.irfft
    """

    def __init__(self, input_channels=2, output_shape = (3, 2048)):
        super().__init__()
        num_filters = output_shape[0]
        filter_length = output_shape[1]
        self.output_filter_domain = "frequency"

        self.num_filters = num_filters
        self.freq_bins = filter_length // 2 + 1  # for rfft/irfft


        self.mlp = nn.Sequential(
            nn.Linear(512, num_filters * self.freq_bins * 2, bias = True),  # real + imag
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = self.mlp(torch.zeros(B,1,1,512, device=x.device))  # [B, N * F * 2]
        out = out.view(B, self.num_filters, self.freq_bins, 2)  # [B, N, F, 2]

        real = out[..., 0]
        imag = out[..., 1]

        complex_filters = torch.complex(real, imag)  # [B, N, F]
        return complex_filters

