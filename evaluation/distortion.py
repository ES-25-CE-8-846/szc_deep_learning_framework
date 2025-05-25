from matplotlib.pyplot import axes, axis, sci
import torch
import scipy.signal
from torch.functional import F
import numpy as np

import matplotlib.pyplot as plt

def normalize(x):
    return x / (np.max(np.abs(x), axis=-1, keepdims=True) + 1e-8)

def normalized_signal_distortion(evaluation_sound, filters, rirs):
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

    print(filters.numpy().shape)

    latency_filter = np.zeros(filters.numpy().shape)

    latency_filter_center_index = latency_filter.shape[-1] // 2

    latency_filter[:,:, latency_filter_center_index] = 1
    latency_filter = latency_filter[:,np.newaxis, :, :]

    # Multiply dry_sound and RIRs (simulating dry signal through environment)
    rirs = scipy.signal.fftconvolve(latency_filter, rirs, axes = 3)
    filterd_rirs = scipy.signal.fftconvolve(filters.detach().cpu().numpy()[:,np.newaxis,:,:], rirs, axes = 3)

    desired_sound = np.sum(scipy.signal.oaconvolve(dry_sound, rirs, axes=3), axis=2 )
    actual_sound = np.sum(scipy.signal.oaconvolve(dry_sound, filterd_rirs, axes=3), axis=2)

    desired_sound = normalize(desired_sound)

    actual_sound = normalize(actual_sound)

    # Truncate to original dry sound length
    desired_sound = desired_sound[..., :dry_len]
    actual_sound = np.asarray(actual_sound[..., :dry_len])

    # plt.plot(np.fft.rfft(desired_sound[0,0,:]), alpha= 0.5)
    # plt.plot(np.fft.rfft(actual_sound[0,0,:]), alpha=0.5)
    # plt.show()

    # signal_distortion = np.mean((desired_sound - actual_sound)**2)/np.mean(desired_sound**2)

    numerator = np.sum(np.fft.rfft(desired_sound, axis=-1)**2)
    denominator = np.sum((np.fft.rfft(desired_sound, axis=-1) - np.fft.rfft(actual_sound, axis=-1))**2) + 1e-10

    snr = 10 * np.log10(np.abs(numerator / denominator))

    return snr


