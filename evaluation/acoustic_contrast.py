from matplotlib.pyplot import axes
import torch
import scipy.signal
import scipy
import numpy as np


def bdr_evaluation(filters, bz_rirs, dz_rirs) -> np.ndarray:
    """Function to commpute the acoustic contrast given the filters and the room impulse responses
    Args:
        filters (torch.Tensor): the filters as a torch tensor shape (S, K) n speakers, filter length
        bz_rirs (torch.Tensor): the room impulse responses shape (S, M, L) n speakers, n microphones, impulse responses length
    Retruns:
        bdr (np.Ndarray): the bright to dark zone ratio across all frequenies in the rfftn, in db
    """

    filters = filters[:, None, :]

    # convolve filters with bz rirs
    filter_bz = np.sum(
        scipy.signal.fftconvolve(filters, bz_rirs, axes=2), axis=0, keepdims=True
    )
    filter_dz = np.sum(
        scipy.signal.fftconvolve(filters, dz_rirs, axes=2), axis=0, keepdims=True
    )

    # compute rfft for bz and dz
    h_b = scipy.fft.rfftn(filter_bz, axes=2)
    h_d = scipy.fft.rfftn(filter_dz, axes=2)

    bdr = 10 * np.log10(
        np.mean(np.abs(h_b) ** 2, axis=1) / np.mean(np.abs(h_d) ** 2, axis=1)
    )

    return bdr


def acc_evaluation(filters, bz_rirs, dz_rirs):
    """Function it compute the acc
    Args:
        filters (torch.Tensor): the filters as a torch tensor shape (S, K) n speakers, filter length
        bz_rirs (torch.Tensor): the bright zone room impulse responses shape (S, M, L)
            n speakers, n microphones, impulse responses length
        dz_rirs (torch.Tensor): the dark zone room impulse responses shape (S, M, L)
            n speakers, n microphones, impulse responses length
    Returns:
        acc (float): the acousitc contrast
    """
    filters = filters[:, None, :]

    # convolve filters with bz rirs
    filter_bz = np.sum(
        scipy.signal.fftconvolve(filters, bz_rirs, axes=2), axis=0, keepdims=True
    )
    filter_dz = np.sum(
        scipy.signal.fftconvolve(filters, dz_rirs, axes=2), axis=0, keepdims=True
    )

    # compute rfft for bz and dz
    h_b = scipy.fft.rfftn(filter_bz, axes=2)
    h_d = scipy.fft.rfftn(filter_dz, axes=2)

    m_b = h_b.shape[0]
    m_d = h_d.shape[0]

    # Compute total energy
    E_b = np.sum(np.abs(h_b) ** 2)
    E_d = np.sum(np.abs(h_d) ** 2)

    # Compute acoustic contrast
    acc = 10 * np.log10((m_b * E_b) / (m_d * E_d))

    # print(f"acc {acc}, acc_shape {acc.shape}, m_b {m_b}, m_d {m_d}")
    return acc
