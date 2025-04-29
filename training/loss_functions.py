import torch


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
