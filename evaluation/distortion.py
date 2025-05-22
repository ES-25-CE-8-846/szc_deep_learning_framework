from matplotlib.pyplot import sci
import torch
import scipy.signal
from torch.functional import F
import numpy as np

import matplotlib.pyplot as plt

def normalize(x):
    return x / (np.max(np.abs(x), axis=-1, keepdims=True) + 1e-8)

def normalized_signal_distortion(evaluation_sound, filters, rirs, auralized_sound):
    """
    Computes signal distortion loss in the bright zone.
    L2 loss between the desired output (dry sound convolved with RIRs)
    and the actual CNN-produced bright zone input signal.
    """
    print(evaluation_sound.shape)
    dry_sound = evaluation_sound                     # (B, S, T)

    batch_size, num_speakers, dry_len = dry_sound.shape
    _, num_mics, _, rir_len = rirs.shape

    dry_sound = dry_sound[:,np.newaxis, ...]  # (B, 1, S, T)


    # Multiply dry_sound and RIRs (simulating dry signal through environment)
    desired_sound = np.sum(scipy.signal.fftconvolve(dry_sound, rirs[...,:512], axes=3), axis=2 )

    desired_sound = normalize(desired_sound)

    actual_sound = np.asarray(auralized_sound)
    actual_sound = normalize(actual_sound)

    # Truncate to original dry sound length
    desired_sound = desired_sound[..., :dry_len]
    actual_sound = np.asarray(actual_sound[..., :dry_len])

    # plt.plot(desired_sound[0,0,:], range(dry_len))
    # plt.plot(actual_sound[0,0,:], range(dry_len))
    # plt.show()

    # signal_distortion = np.mean((desired_sound - actual_sound)**2)/np.mean(desired_sound**2)

    numerator = np.sum(desired_sound**2)
    denominator = np.sum((desired_sound - actual_sound)**2) + 1e-10

    snr = 10 * np.log10(numerator / denominator)

    return snr


