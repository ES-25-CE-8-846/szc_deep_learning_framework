from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import convolve

class AudioFilterDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.data_dir = Path(data_dir)
        self.file_paths = list(self.data_dir.rglob("*.npz"))
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load impulse responses
        rirs = np.load(self.file_paths[idx])
        
        bz_rirs = rirs["bz_rir"]
        dz_rirs = rirs["dz_rir"]
        
        # An item in this case should be a tuple of (source, microphones) so the rirs should be convolved with the source

        # Load source wav file
        source_path = "testing_scripts/relaxing-guitar-loop-v5-245859.wav"
        sampling_rate, source = wav.read(source_path)

        # Select 50 ms of the source
        samples = int(50/1000*sampling_rate)
        source = source[400000:400000+samples]

        # Convert source to float for better precision
        source = source.astype(np.float32)

        # Prepare arrays to store convolved signals
        mic_signals = []

        # Convolve source with each microphone RIR in bz_rirs
        for _, mic in enumerate(bz_rirs):
            mic_signal = convolve(source, mic, mode='same')
            mic_signals.append(mic_signal)
        
        # Convolve source with each microphone RIR in dz_rirs
        for _, mic in enumerate(dz_rirs):
            mic_signal = convolve(source, mic, mode='same')
            mic_signals.append(mic_signal)

        # Stack all microphone signals
        mic_signals = np.stack(mic_signals)

        # Convert to tensors
        x_src = torch.tensor(source, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        x_mics = torch.tensor(mic_signals, dtype=torch.float32)

        # Make a dictionary to hold the data
        data_dict = {
            'x_src': x_src,
            'x_mics': x_mics,
            'rirs': rirs
        }

        return data_dict
        


if __name__ == "__main__":
    # Example usage
    dataset = AudioFilterDataset(data_dir="/home/morten/GitHub/dataset/shoebox/alfredo-request")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for x_src, x_mics, y in dataloader:
        print(f"x_src shape: {x_src.shape}, x_mics shape: {x_mics.shape}, y shape: {y.shape}")
        # Add your training loop here