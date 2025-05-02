import torch
from torch import nn
import torchinfo

class AudioFilterEstimator(nn.Module):
    """Model takes speaker source audio + microphone inputs and estimates filter coefficients in frequency domain."""
    def __init__(self, num_mics: int, output_dim: int = 1434):
        super().__init__()
        self.num_mics = num_mics
        
        # Shared conv encoder for all channels (source + mics)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),  # [B, 32, T//2]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), # [B, 64, T//4]
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> [B, 64, 1]
            nn.Flatten()              # -> [B, 64]
        )
        
        total_inputs = 64 * (1 + num_mics)  # 1 source + M mics
        
        self.mlp = nn.Sequential(
            nn.Linear(total_inputs, 512),
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
    

if __name__ == "__main__":

    model = AudioFilterEstimator(num_mics=3, output_dim=1434)
    torchinfo.summary(model)