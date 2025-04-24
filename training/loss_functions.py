import torch 


def sound_loss(loss_data_dict, weights=None, device = None):
    """
    Spectral loss function that compares the FFT magnitude of the input and target sound.

    Args:
        loss_data_dict (dict): Dictionary with keys:
            - 'gt_sound': Ground truth dry sound [K]
            - 'bz_input': Binaural zone audio [B, C, T]
            - 'dz_input': Dead zone audio [B, C, T]
        weights (optional): Not used for now, placeholder for later extensions

    Returns:
        torch.Tensor: Scalar loss value
    """
    
    dry_sound = loss_data_dict['gt_sound']

    print(f"dry sound shape {dry_sound.size()}")
    
    dry_sound_len = dry_sound.size()[2]

    # cropping is needed ot have the same dimiensions in the fft output 
    bz_input_sound = loss_data_dict['bz_input'][:,:,0:dry_sound_len]
    dz_input_sound = loss_data_dict['dz_input'][:,:,0:dry_sound_len]

    hann_window = torch.windows.hann(dry_sound_len) 

    if device is not None:
        dry_sound = dry_sound.to(device)
        hann_window = hann_window.to(device)
        bz_input_sound = bz_input_sound.to(device)
        dz_input_sound = dz_input_sound.to(device)


    des_bz_h = torch.fft.rfft(dry_sound * hann_window)
    des_dz_h = torch.fft.rfft(torch.zeros_like(dry_sound)) 
    

    bz_h = torch.fft.rfft(bz_input_sound * hann_window)
    dz_h = torch.fft.rfft(dz_input_sound * hann_window)

    bz_loss = torch.norm(torch.abs(bz_h) - torch.abs(des_bz_h), p=2)
    dz_loss = torch.norm(torch.abs(dz_h) - torch.abs(des_dz_h), p=2)
    
    loss = bz_loss + dz_loss

    loss_dict = {'bz_loss':bz_loss,
                 'dz_loss':dz_loss,
                 'loss':loss}

    return loss_dict 
    


