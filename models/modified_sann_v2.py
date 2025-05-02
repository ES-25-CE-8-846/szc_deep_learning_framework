import torch
from torch import nn

class AudioFilterEstimator(nn.Module):
    """Model takes approximated ATF's as input and estimates filter coefficients in frequency domain."""
    def __init__(self, num_mics: int, output_dim: int = 1434):
        """
        Model takes speaker source audio + microphone inputs and estimates filter coefficients in frequency domain.

        Args:
            num_mics (int): _description_
            output_dim (int, optional): The number of coefficients the model should output. Usually calculated with: no_of_speakers * 2 (real + imag number) * no_freq_bins. Defaults to 1434.
        """
        super().__init__()
        self.num_mics = num_mics
    
        # block_size = delay (in samples) between information in, and audio out
        # Frequency bins: you’ll still likely focus on 100 – 1 500 Hz → that’s about
        # binlow=⌈100/(48 000/block_size)⌉,binhigh=⌊1500/(48 000/block_size)⌋ ##  highest_freq = (bin_high * sample_rate) / block_size
        #  ≈ bins 3…32 (≈ 30 bins). <-- with block_size = 1024
        #  ≈ bins 18…256 (≈ 239 bins). <-- with block_size = 8192

        input_size = num_mics * 3 * 2 * 239 # num_mics * num_speakers * 2 (real + imag) * num_bins

        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x_src: torch.Tensor, x_mics: torch.Tensor) -> torch.Tensor:
        # x_src: [B, 1, T], x_mics: [B, M, T]
        B, M, T = x_mics.shape
        src_feat = self.conv(x_src)       # [B, 64]
        mic_feats = [self.conv(x_mics[:, i:i+1]) for i in range(M)]  # list of [B, 64]
        all_feats = torch.cat([src_feat] + mic_feats, dim=1)         # [B, 64*(M+1)]
        return self.mlp(all_feats)        # [B, output_dim]