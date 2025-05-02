import torch 
import numpy as np
from models import modified_sann
import scipy


if __name__ == "__main__":

    # Define sample rate and window size
    window_size = 20 # ms

    # Load audio source
    sample_rate, source = scipy.io.wavfile.read("testing_scripts/relaxing-guitar-loop-v5-245859.wav")

    # Convert window size to samples
    window_size_samples = 960#int(sample_rate * window_size / 1000)

    # check for a chunk with actual audio
    index = 100000

    # Cut it into chunks of 960 samples
    src_window = source[index:index+window_size_samples]

    # Load simulated data for testing
    bz_rir = np.load("testing_scripts/bz_rir.npy")
    dz_rir = np.load("testing_scripts/dz_rir.npy")

    # Convolve with audio to obtain test signals from a couple of mics
    mic1 = scipy.signal.fftconvolve(src_window, bz_rir[0], mode='same').mean(axis=1)
    mic2 = scipy.signal.fftconvolve(src_window, bz_rir[1], mode='same').mean(axis=1)
    mic3 = scipy.signal.fftconvolve(src_window, bz_rir[2], mode='same').mean(axis=1)

    # Take

    # Stack the mic signals into a single tensor
    mic_test = np.stack((mic1, mic2, mic3), axis=0)

    src_test = torch.rand(1, 1, 960)
    mic_test2 = torch.rand(1, 3, 960)

    print(src_window.shape)
    print(src_test.shape)
    print(mic_test.shape)
    print(mic_test2.shape)

    model = modified_sann.AudioFilterEstimator(
        num_mics=3,
        output_dim=3824
    )

    print(model.forward(src_test,mic_test2).size())