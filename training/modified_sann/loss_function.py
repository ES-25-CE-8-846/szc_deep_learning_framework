import torch
import numpy as np



def sann_loss(model_output, H_B, H_D, p_TB, freq_bins, bz_mics, dz_mics):
    """
    model_output: [B, F, L] — predicted filters g(ω)
    H_B: [F, M_B, L] — ATFs for bright zone
    H_D: [F, M_D, L] — ATFs for dark zone
    p_TB: [F, M_B] — target pressure in bright zone
    """
    B, F, L = model_output.shape
    M_B = bz_mics
    M_D = dz_mics

    # Expand H_B and H_D to match batch dimension
    H_B_exp = H_B.unsqueeze(0).expand(B, -1, -1, -1)  # [B, F, M_B, L]
    H_D_exp = H_D.unsqueeze(0).expand(B, -1, -1, -1)  # [B, F, M_D, L]

    # model_output: [B, F, L] => add dim for matmul
    g = model_output.unsqueeze(-1)  # [B, F, L, 1]

    # Compute HB @ g for each batch and frequency
    HB_g = torch.matmul(H_B_exp, g).squeeze(-1)  # [B, F, M_B]
    HD_g = torch.matmul(H_D_exp, g).squeeze(-1)  # [B, F, M_D]

    # Magnitude difference in bright zone
    p_TB_exp = p_TB.unsqueeze(0).expand(B, -1, -1)  # [B, F, M_B]
    l1 = torch.mean((torch.abs(HB_g) - torch.abs(p_TB_exp)) ** 2)

    # Energy in dark zone
    #l2 = torch.mean(torch.abs(HD_g) ** 2)

    return l1 #+ l2


def sann_loss(model_output, source_sound, bz_mics, dz_mics, freq_bins, device=None):
    """
    Compute the loss for the modified SANN model.

    Args:
        model_output (torch.tensor): The filter coefficients estimated by the model.
        device (str, optional): The device to perform the computation on. Defaults to None.

    Returns:
        torch.Tensor: Scalar loss value
    """
    alpha, beta, gamma = 0.5, 0.5, 0.5

    # Convolve the source sound with the filter coefficients
    # Assuming model_output is of shape (B, M, F) and source_sound is of shape (B, S, T)
    # where B is batch size, M is number of microphones, S is number of speakers, and T is time, and F is the number of frequency bins
    # Convolution operation
    input_fft = torch.fft.rfft(source_sound, dim=-1)

    filtered_input = model_output * input_fft.unsqueeze(1)  # Broadcasting to match dimensions
    filtered_time = filtered_time = torch.fft.irfft(filtered_input, n=source_sound.shape[-1], dim=-1)  # [B, 3, T]

    # The first loss term
    l1 = 1/(freq_bins*bz_mics) * torch.sum(torch.abs(model_output[:, :bz_mics*freq_bins] - filtered_time[:, :bz_mics*freq_bins]))
   
    # The second loss term
    l2 = 1/(freq_bins*dz_mics) * torch.sum(torch.abs(model_output[:, bz_mics*freq_bins:] - filtered_time[:, bz_mics*freq_bins:]))

    # Apply filters to the source sound from each speaker to each microphone
    # Assuming filters is of shape (B, M, F) and src_sound is of shape (B, S, T)
    # where B is batch size, M is number of microphones, S is number of speakers, and T is time, and F is the number of frequency bins


    return alpha*l1 + (1-alpha)*l2
