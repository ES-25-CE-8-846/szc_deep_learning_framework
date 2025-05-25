from operator import matmul
import torch
import torch.nn.functional as F
import scipy.signal
import numpy as np
import soundfile


def _normalize(tensor, dim=-1, eps=1e-8):
    """
    Normalize a tensor along the specified dimension to have max absolute value of 1.

    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to normalize (default: last).
        eps (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    max_val = torch.amax(torch.abs(tensor), dim=dim, keepdim=True)
    return tensor / (max_val + eps)


def sound_loss(loss_data_dict, weights=None, device=None):
    """
    Spectral loss function that compares the FFT magnitude of the input and target sound.

    Args:
        loss_data_dict (dict): Dictionary with keys:
            'gt_sound': Ground truth dry sound [K]
            'bz_input': bright zone audio [B, C, T]
            'dz_input': dark zone audio [B, C, T]
        weights (optional): Not used for now, placeholder for later extensions
        device: compute device

    Returns:
        torch.Tensor: Scalar loss value
    """

    dry_sound = loss_data_dict["gt_sound"]

    # print(f"dry sound shape {dry_sound.size()}")

    dry_sound_len = dry_sound.size()[2]

    # cropping is needed to have the same dimensions in the fft output
    bz_input_sound = loss_data_dict["bz_input"][:, :, 0:dry_sound_len]
    dz_input_sound = loss_data_dict["dz_input"][:, :, 0:dry_sound_len]

    # create window for better fft
    hann_window = torch.windows.hann(dry_sound_len)

    if device is not None:
        dry_sound = dry_sound.to(device)
        hann_window = hann_window.to(device)
        bz_input_sound = bz_input_sound.to(device)
        dz_input_sound = dz_input_sound.to(device)

    # compute the desired frequency responses
    des_bz_h = torch.fft.rfft(dry_sound * hann_window)
    des_dz_h = torch.fft.rfft(torch.zeros_like(dry_sound))

    # compute the measured frequency responses
    bz_h = torch.fft.rfft(bz_input_sound * hann_window)
    dz_h = torch.fft.rfft(dz_input_sound * hann_window)

    n_bz_mics = bz_h.size()[1]
    n_dz_mics = dz_h.size()[1]

    # print(f"n bz mics {n_bz_mics}, n dz mics {n_dz_mics}")

    # compute the loss
    bz_loss = torch.mean(torch.norm(torch.abs(bz_h) - torch.abs(des_bz_h), p=2, dim=1))
    dz_loss = torch.mean(torch.norm(torch.abs(dz_h) - torch.abs(des_dz_h), p=2, dim=1))

    if weights is not None:
        loss = bz_loss * weights[0] + dz_loss * weights[1]
    else:
        loss = bz_loss + dz_loss

    loss_dict = {
        "bz_loss": bz_loss,
        "dz_loss": dz_loss,
        "loss": loss,
        "fft_bz": bz_h,
        "fft_dz": dz_h,
        "fft_bz_des": des_bz_h,
    }

    return loss_dict


def sann_loss(loss_data_dict, weights=None, device=None):
    """
    Loss function based on SANN-PSZ, including L1, L2, L3, and L4 terms.
    """

    bz_rirs_fft = torch.fft.rfft(loss_data_dict["data_dict"]["bz_rirs"])  # (B, S, M, F)
    dz_rirs_fft = torch.fft.rfft(loss_data_dict["data_dict"]["dz_rirs"])  # (B, S, M, F)
    sr = loss_data_dict["data_dict"]["sr"][0].item()

    # bz_rirs_fft = bz_rirs_fft.permute(0, 2, 1, 3)  # (B, M, S, F)
    # dz_rirs_fft = dz_rirs_fft.permute(0, 2, 1, 3)  # (B, M, S, F)de making the combe into a single sweep.[8] The garden, as it was originally laid out, influenced other designers and contributed to def

    complex_filters = loss_data_dict["filters_frq"].unsqueeze(2)  # (B, S, 1, F)
    filters_time = loss_data_dict["filters_time"]

    # === L1: Matching desired pressure in BZ ===
    predicted_l1 = torch.sum(
        bz_rirs_fft * complex_filters.permute(0, 2, 1, 3), dim=2
    )  # (B, M, F)
    ptb = torch.mean(bz_rirs_fft, dim=2)  # (B, M, F)
    l1 = torch.mean((torch.abs(ptb) - torch.abs(predicted_l1)) ** 2)

    # === L2: Suppress energy in DZ ===
    predicted_l2 = torch.sum(
        dz_rirs_fft * complex_filters.permute(0, 2, 1, 3), dim=2
    )  # (B, M, F)
    l2 = torch.mean(torch.abs(predicted_l2) ** 2)

    # === L3: Limit gain amplitude ===
    g_max = 1.0/8
    gain_mag = torch.abs(complex_filters)  # (B, S, F)
    excess = torch.clamp(gain_mag - g_max, min=0.0)
    l3 = torch.mean(excess**2)

    # === L4: Enforce time-domain compactness ===
    # Create window and dummy bandpass filter
    filter_len = filters_time.size()[-1]  # Choose based on inverse FFT target size
    time_filters = torch.fft.irfft(
        loss_data_dict["filters_frq"], n=filter_len
    )  # (B, S, T)

    # Window function (e.g. inverted Hann)
    w = 1.0 - torch.hann_window(filter_len, periodic=False).to(time_filters.device)
    w = w.view(1, 1, -1)

    # Filter specifications
    lowcut = 250     # Low cutoff frequency (Hz)
    highcut = 3500   # High cutoff frequency (Hz)

    # Design the Butterworth bandpass filter
    freqs = np.linspace(0, sr/2, int(filter_len/2 + 1)) #frequency bins for bandpass filter filter
    H = np.zeros(freqs.shape)
    H[(freqs >= lowcut) & (freqs <= highcut)] = 1.0  # Ideal rectangular bandpass
    # Use a transition width (e.g., 200 Hz) and a cosine ramp
    transition_width = 200

    # Lower transition
    start = np.logical_and(freqs >= (lowcut - transition_width), freqs < lowcut)
    H[start] = 0.5 * (1 + np.cos(np.pi * (freqs[start] - lowcut) / transition_width))

    # Upper transition
    end = np.logical_and(freqs > highcut, freqs <= (highcut + transition_width))
    H[end] = 0.5 * (1 + np.cos(np.pi * (freqs[end] - highcut) / transition_width))

    bandpass_filter_frq = torch.tensor(H, device=complex_filters.device)

    weighted_filter_frq = complex_filters * bandpass_filter_frq

    weighted_filter = torch.fft.irfft(weighted_filter_frq) * w

    # Energy of weighted signal
    l4 = torch.mean(weighted_filter**2)

    # === Combine Losses ===
    if weights is not None:
        loss = (
            weights[0] * l1 + (1 - weights[0]) * l2 + weights[1] * l3 + weights[2] * l4
        )
    else:
        loss = l1 + l2 + l3 + l4

    ##### the following is for plotting ######

    loss_dict = {
        "loss": loss,
        # "filter_len": filter_len,
        # "bandpass_filter": H,
        # "l1": l1,
        # "l2": l2,
        # "l3": l3,
        # "l4": l4,
    }

    return loss_dict


def acc_loss(loss_data_dict, weights=None, device=None):
    """Function to compute acoustic contrast loss, mimicking fftconvolve behavior."""
    bz_rirs = loss_data_dict["data_dict"]["bz_rirs"]  # shape: (B, M, S, L)
    dz_rirs = loss_data_dict["data_dict"]["dz_rirs"]
    filters = loss_data_dict["filters_time"]  # shape: (B, S, K)

    # print(f"---torch ---")
    filters = filters[:, None, :, :]  # shape: (B, 1, S, K)

    rir_len = bz_rirs.size(-1)
    filt_len = filters.size(-1)
    conv_len = rir_len + filt_len - 1  # fftconvolve output length

    # Pad filters and RIRs to match fftconvolve output length
    filters_padded = F.pad(filters, (0, conv_len - filt_len))
    bz_rirs_padded = F.pad(bz_rirs, (0, conv_len - rir_len))
    dz_rirs_padded = F.pad(dz_rirs, (0, conv_len - rir_len))

    # FFT along time axis
    filters_frq = torch.fft.rfft(filters_padded, n=conv_len, dim=-1)  # (B, 1, S, F)
    bz_frq = torch.fft.rfft(bz_rirs_padded, n=conv_len, dim=-1)  # (B, M, S, F)
    dz_frq = torch.fft.rfft(dz_rirs_padded, n=conv_len, dim=-1)

    # Multiply and sum over speakers
    filtered_bz = torch.sum(filters_frq * bz_frq, dim=2)  # shape: (B, M, F)
    filtered_dz = torch.sum(filters_frq * dz_frq, dim=2)  # shape: (B, M, F)

    # print(f"filtered_bz max abs {torch.max(torch.abs(filtered_bz))}")

    # Energy computation
    energy_bz = torch.sum(torch.abs(filtered_bz) ** 2)  # shape: (B, M)
    energy_dz = torch.sum(torch.abs(filtered_dz) ** 2)

    # print(f"energy_bz {energy_bz}")

    # Normalize by number of microphones
    M_b = bz_rirs.shape[1]
    M_d = dz_rirs.shape[1]

    contrast_ratio = (M_d * energy_bz) / (M_b * energy_dz + 1e-10)

    # print(f"contrast_ratio {contrast_ratio}")

    # Acoustic contrast (negative because loss is minimized)
    acc_loss_val = 10 * torch.log10(contrast_ratio + 1e-10)

    return {"loss": 1 / acc_loss_val}


signal_distortion_reference_sound, sr = soundfile.read(
    "./training/loss_function_reference_data/sd_reference.wav"
)
sd_reference = torch.tensor(signal_distortion_reference_sound)


def signal_distortion_loss(loss_data_dict, weights=None, device=None):
    """
    Computes signal distortion loss in the bright zone.
    L2 loss between the desired output (dry sound convolved with RIRs)
    and the actual CNN-produced bright zone input signal.
    """
    dry_sound = loss_data_dict["gt_sound"]  # (B, S, T)
    bz_rirs = loss_data_dict["data_dict"]["bz_rirs"]  # (B, M, S, L)

    B, M, S, L = bz_rirs.shape

    # dry_sound =   sd_reference
    # dry_sound = torch.ones((B, S, dry_sound.size()[-1])) * dry_sound

    dry_sound = dry_sound.to(bz_rirs.device)

    bz_rirs = loss_data_dict["data_dict"]["bz_rirs"]  # (B, M, S, L)
    filters = loss_data_dict["filters_time"]  # (B, S, K)
    # bz_input = loss_data_dict["bz_input"]  # (B, M, T)

    latency_filter = np.zeros(filters.detach().cpu().numpy().shape)

    latency_filter_center_index = latency_filter.shape[-1] // 2

    latency_filter[:,:, latency_filter_center_index] = 1
    latency_filter = latency_filter[:,np.newaxis, :, :]

    # Multiply dry_sound and RIRs (simulating dry signal through environment)
    bz_rirs = torch.tensor(scipy.signal.fftconvolve(latency_filter, bz_rirs.detach().cpu().numpy(), axes = 3), device = dry_sound.device)

    batch_size, num_speakers, dry_len = dry_sound.shape
    _, num_mics, _, rir_len = bz_rirs.shape
    _, _, filt_len = filters.shape

    conv_len = rir_len + filt_len - 1
    dry_sound = dry_sound.unsqueeze(1)  # (B, 1, S, T)

    # Pad signals and filters for FFT-based convolution
    dry_sound_padded = F.pad(dry_sound, (0, conv_len - dry_len))
    bz_rirs_padded = F.pad(bz_rirs, (0, conv_len - rir_len))
    filters_padded = F.pad(filters, (0, conv_len - filt_len)).unsqueeze(
        1
    )  # (B, 1, S, L)

    # FFT
    dry_fft = torch.fft.rfft(dry_sound_padded, n=conv_len, dim=-1)  # (B, 1, S, F)
    rirs_fft = torch.fft.rfft(bz_rirs_padded, n=conv_len, dim=-1)  # (B, M, S, F)
    filters_fft = torch.fft.rfft(filters_padded, n=conv_len, dim=-1)  # (B, 1, S, F)

    # Multiply dry_sound and RIRs (simulating dry signal through environment)
    desired_fft = dry_fft * rirs_fft  # (B, M, S, F)
    desired_fft = torch.sum(desired_fft, dim=2)  # sum over speakers → (B, M, F)

    # Multiply filters and RIRs (actual system output)
    actual_fft = filters_fft * rirs_fft  # (B, M, S, F)
    actual_fft = torch.sum(actual_fft, dim=2)  # (B, M, F)

    # IFFT to time domain
    desired = torch.fft.irfft(desired_fft, n=conv_len, dim=-1)  # (B, M, T)
    actual = torch.fft.irfft(actual_fft, n=conv_len, dim=-1)

    # Truncate to original dry sound length
    desired = _normalize(desired[..., :dry_len])
    actual = _normalize(actual[..., :dry_len])

    # L2 loss between desired and actual sound
    signal_distortion = torch.mean((desired - actual) ** 2)/torch.mean(desired**2)

    return {"loss": 1/signal_distortion}


def sd_acc_loss(loss_data_dict, weights=[10, 1], device=None):
    """Function to compute loss based on the acoustic contrast and signal distortion"""
    if weights is None:
        weights = [1, 1]

    return {
        "loss": acc_loss(loss_data_dict)["loss"] * weights[0]
        + signal_distortion_loss(loss_data_dict)["loss"] * weights[1]
    }


def zero_loss_functions(loss_data_dict, weights=None, device=None):
    """Loss function to debug gradient calculation dont use for actual training"""

    dry_sound = loss_data_dict["gt_sound"]

    # print(f"dry sound shape {dry_sound.size()}")

    dry_sound_len = dry_sound.size()[2]

    # cropping is needed to have the same dimensions in the fft output
    bz_input_sound = loss_data_dict["bz_input"][:, :, 0:dry_sound_len]
    dz_input_sound = loss_data_dict["dz_input"][:, :, 0:dry_sound_len]

    # create window for better fft
    hann_window = torch.windows.hann(dry_sound_len)

    if device is not None:
        dry_sound = dry_sound.to(device)
        hann_window = hann_window.to(device)
        bz_input_sound = bz_input_sound.to(device)
        dz_input_sound = dz_input_sound.to(device)

    # compute the desired frequency responses
    des_bz_h = torch.fft.rfft(dry_sound * hann_window)
    des_dz_h = torch.fft.rfft(torch.zeros_like(dry_sound))

    # compute the measured frequency responses
    bz_h = torch.fft.rfft(bz_input_sound * hann_window)
    dz_h = torch.fft.rfft(dz_input_sound * hann_window)

    n_bz_mics = bz_h.size()[1]
    n_dz_mics = dz_h.size()[1]

    # print(f"n bz mics {n_bz_mics}, n dz mics {n_dz_mics}")

    # compute the loss
    bz_loss = torch.mean(torch.norm(torch.abs(bz_h) - torch.abs(des_bz_h), p=2, dim=1))
    dz_loss = torch.mean(torch.norm(torch.abs(dz_h) - torch.abs(des_dz_h), p=2, dim=1))
    filters = loss_data_dict["filters"]
    target = torch.zeros_like(filters)

    loss = torch.mean(torch.sum(torch.pow(filters, 2), dim=1))

    loss_dict = {
        "bz_loss": bz_loss,
        "dz_loss": dz_loss,
        "loss": loss,
        "fft_bz": bz_h,
        "fft_dz": dz_h,
        "fft_bz_des": des_bz_h,
    }

    return loss_dict
