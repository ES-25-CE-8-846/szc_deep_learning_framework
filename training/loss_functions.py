from operator import matmul
import torch
import torch.nn.functional as F

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


def sann_loss(loss_data_dict, weights=None, device=None, bins=239):
    """
    Loss function based on SANN-PSZ, including L1, L2, L3, and L4 terms.
    """

    bz_rirs_fft = torch.fft.rfft(loss_data_dict['data_dict']['bz_rirs'])  # (B, S, M, F)
    dz_rirs_fft = torch.fft.rfft(loss_data_dict['data_dict']['dz_rirs'])  # (B, S, M, F)

    # bz_rirs_fft = bz_rirs_fft.permute(0, 2, 1, 3)  # (B, M, S, F)
    # dz_rirs_fft = dz_rirs_fft.permute(0, 2, 1, 3)  # (B, M, S, F)de making the combe into a single sweep.[8] The garden, as it was originally laid out, influenced other designers and contributed to def

    complex_filters = loss_data_dict['filters_frq'].unsqueeze(2)  # (B, S, 1, F)

    # === L1: Matching desired pressure in BZ ===
    predicted_l1 = torch.sum(bz_rirs_fft * complex_filters.permute(0, 2, 1, 3), dim=2)  # (B, M, F)
    ptb = torch.mean(bz_rirs_fft, dim=2)  # (B, M, F)
    l1 = torch.mean((torch.abs(ptb) - torch.abs(predicted_l1)) ** 2)

    # === L2: Suppress energy in DZ ===
    predicted_l2 = torch.sum(dz_rirs_fft * complex_filters.permute(0, 2, 1, 3), dim=2)  # (B, M, F)
    l2 = torch.mean(torch.abs(predicted_l2) ** 2)

    # === L3: Limit gain amplitude ===
    g_max = 1.0
    gain_mag = torch.abs(complex_filters)  # (B, S, F)
    excess = torch.clamp(gain_mag - g_max, min=0.0)
    l3 = torch.mean(excess ** 2)

    # === L4: Enforce time-domain compactness ===
    # Create window and dummy bandpass filter
    filter_len = 512  # Choose based on inverse FFT target size
    time_filters = torch.fft.irfft(loss_data_dict['filters_frq'], n=filter_len)  # (B, S, T)

    # Window function (e.g. inverted Hann)
    w = 1.0 - torch.hann_window(filter_len, periodic=False).to(time_filters.device)
    w = w.view(1, 1, -1)

    # Dummy bandpass filter - replace with real FIR filter
    bandpass_filter = torch.ones(1, 1, 33, device=time_filters.device) / 33  # simple lowpass

    # Convolve time filters with bandpass
    filt = F.conv1d(
        time_filters.view(-1, 1, filter_len),  # (B*S, 1, T)
        bandpass_filter,
        padding='same'
    ).view_as(time_filters)  # (B, S, T)

    # Apply window
    filt_weighted = filt * w

    # Energy of weighted signal
    l4 = torch.mean(filt_weighted ** 2)

    # === Combine Losses ===
    if weights is not None:
        loss = weights[0] * l1 + (1 - weights[0]) * l2 + weights[1] * l3 + weights[2] * l4
    else:
        loss = l1 + l2 + l3 + l4

    ##### the following is for plotting ######
    dry_sound = loss_data_dict["gt_sound"]

    # print(f"dry sound shape {dry_sound.size()}")

    dry_sound_len = dry_sound.size()[2]
    hann_window = torch.windows.hann(dry_sound_len)

    # cropping is needed to have the same dimensions in the fft output
    bz_input_sound = loss_data_dict["bz_input"][:, :, 0:dry_sound_len]
    dz_input_sound = loss_data_dict["dz_input"][:, :, 0:dry_sound_len]
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


    loss_dict = {'loss':loss,
                 'filter_len':filter_len,
                 'bandpass_filter':bandpass_filter,
                 'l1':l1,
                 'l2':l2,
                 'l3':l3,
                 'l4':l4,
                 "fft_bz": bz_h,
                 "fft_dz": dz_h,
                 "fft_bz_des": des_bz_h}

    return loss_dict





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
