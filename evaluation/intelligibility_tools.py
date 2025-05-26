import torch
import torchaudio
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import torchaudio.functional as F
import numpy as np


def evaluate_stoi_pesq(signal, dry_signal, sample_rate=16000):
    """
    Evaluates the STOI and PESQ metrics for the given signal against the dry signal.

    Input:
    signal: The signal to evaluate
    dry_signal: The dry reference signal
    sample_rate: The sample rate of the signals

    Returns:
    stoi: The STOI score
    pesq: The PESQ score
    """
    # Convert to numpy and flatten to 1D
    if isinstance(signal, torch.Tensor):
        signal = signal.squeeze().numpy()
    if isinstance(dry_signal, torch.Tensor):
        dry_signal = dry_signal.squeeze().numpy()

    # Make sure they're 1D
    signal = signal.flatten()
    dry_signal = dry_signal.flatten()

    # Make sure the lengths match
    min_len = min(len(signal), len(dry_signal))
    signal = signal[:min_len]
    dry_signal = dry_signal[:min_len]

    # Resample to 16KHz if needed
    if sample_rate != 16000:
        signal = F.resample(torch.tensor(signal), sample_rate, 16000).numpy()
        dry_signal = F.resample(torch.tensor(dry_signal), sample_rate, 16000).numpy()

    # Ensure they are 1D numpy arrays
    signal = signal.flatten()
    dry_signal = dry_signal.flatten()
    if len(signal.shape) > 1:
        signal = signal.squeeze()
    if len(dry_signal.shape) > 1:
        dry_signal = dry_signal.squeeze()


    # Calculate STOI and PESQ
    stoi_score = stoi(dry_signal, signal, fs_sig=16000)
    pesq_score = pesq(16000, dry_signal, signal)

    return stoi_score, pesq_score

def evaluate_mos(signal, dry_signal, device=None):
    """
    Evaluates the MOS score for the given signal against the dry signal.

    Input:
    signal: The signal to evaluate
    dry_signal: The dry reference signal

    Returns:
    mos: The MOS score
    """
    # Convert to numpy and flatten to 1D
    if isinstance(signal, torch.Tensor):
        signal = signal.squeeze().numpy()
    if isinstance(dry_signal, torch.Tensor):
        dry_signal = dry_signal.squeeze().numpy()


    # Make sure the lengths match
    min_len = min(len(signal), len(dry_signal))
    signal = signal[:min_len]
    dry_signal = dry_signal[:min_len]

    # Resample to 16KHz from the 44100 Hz in the source sound files
    signal = F.resample(torch.tensor(signal), 44100, 16000).numpy()
    dry_signal = F.resample(torch.tensor(dry_signal), 44100, 16000).numpy()

    # Ensure they are 1D numpy arrays
    signal = signal.flatten()
    dry_signal = dry_signal.flatten()
    if len(signal.shape) > 1:
        signal = signal.squeeze()
    if len(dry_signal.shape) > 1:
        dry_signal = dry_signal.squeeze()

    signal = signal[np.newaxis, :]
    dry_signal = dry_signal[np.newaxis, :]

    # Make sure they're [1, k]D
    if device is not None:
        print(f"evaluating mos on {device}")
        signal = torch.tensor(signal[0:1,:]).float().to(device)
        dry_signal = torch.tensor(dry_signal[0:1, :]).float().to(device)

        subjective_model = SQUIM_SUBJECTIVE.get_model().to(device)
        mos_score = subjective_model(signal, dry_signal)
    else:

        signal = torch.tensor(signal[0:1,:]).float()
        dry_signal = torch.tensor(dry_signal[0:1, :]).float()

        subjective_model = SQUIM_SUBJECTIVE.get_model()
        mos_score = subjective_model(signal, dry_signal)
    return mos_score[0].item()

def evaluate_stoi(signal, dry_signal, sample_rate=16000):
    """
    Evaluates the STOI score for the given signal against the dry signal.

    Input:
    signal: The signal to evaluate
    dry_signal: The dry reference signal
    sample_rate: The sample rate of the signals

    Returns:
    stoi: The STOI score
    """
    # Convert to numpy and flatten to 1D
    if isinstance(signal, torch.Tensor):
        signal = signal.squeeze().numpy()
    if isinstance(dry_signal, torch.Tensor):
        dry_signal = dry_signal.squeeze().numpy()

    # Make sure they're 1D
    signal = signal.flatten()
    dry_signal = dry_signal.flatten()

    # Calculate STOI
    stoi_score = stoi(dry_signal, signal, fs_sig=sample_rate)

    return stoi_score

def evaluate_pesq(signal, dry_signal, sample_rate=16000):
    """
    Evaluates the PESQ score for the given signal against the dry signal.

    Input:
    signal: The signal to evaluate
    dry_signal: The dry reference signal
    sample_rate: The sample rate of the signals

    Returns:
    pesq: The PESQ score
    """
    # Convert to numpy and flatten to 1D
    if isinstance(signal, torch.Tensor):
        signal = signal.squeeze().numpy()
    if isinstance(dry_signal, torch.Tensor):
        dry_signal = dry_signal.squeeze().numpy()

    # Make sure they're 1D
    signal = signal.flatten()
    dry_signal = dry_signal.flatten()

    # Calculate PESQ
    pesq_score = pesq(sample_rate, dry_signal, signal)

    return pesq_score

def evaluate_intelligibility(signal, dry_signal, sample_rate=16000):
    """
    Evaluates the MOS, STOI and PESQ metrics for the given signal against the dry signal.

    Input:
    signal: The signal to evaluate
    dry_signal: The dry reference signal
    sample_rate: The sample rate of the signals

    Returns:
    mos: The MOS score
    stoi: The STOI score
    pesq: The PESQ score
    """
    # Convert to numpy and flatten to 1D
    if isinstance(signal, torch.Tensor):
        signal = signal.squeeze().numpy()
    if isinstance(dry_signal, torch.Tensor):
        dry_signal = dry_signal.squeeze().numpy()

    # Make sure they're 1D
    signal = signal.flatten()
    dry_signal = dry_signal.flatten()

    # Calculate STOI and PESQ
    stoi_score = stoi(dry_signal, signal, fs_sig=sample_rate)
    pesq_score = pesq(sample_rate, dry_signal, signal)

    subjective_model = SQUIM_SUBJECTIVE.get_model()
    mos_score = subjective_model(signal, dry_signal)

    return mos_score, stoi_score, pesq_score

def evaluate_signals(Dry_sound, SAMPLE_BZ, SAMPLE_DZ, Original_BZ, Original_DZ):
    """
    Generates the estimated metrics (STOI, PESQ, MOS) for the original and filtered signals using the SQUIM model.

    Input:
    Dry_sound: Path to the dry sound file
    SAMPLE_BZ: Path to the bright mic filtered sound file
    SAMPLE_DZ: Path to the dark mic filtered sound file
    Original_BZ: Path to the bright mic original sound file
    Original_DZ: Path to the dark mic original sound file

    Returns:
    A dictionary containing the metrics for the original and filtered signals.
    """
    WAVEFORM_DRY, SAMPLE_RATE_DRY = torchaudio.load(Dry_sound)
    WAVEFORM_BZ, SAMPLE_RATE_BZ = torchaudio.load(SAMPLE_BZ)
    WAVEFORM_DZ, SAMPLE_RATE_DZ = torchaudio.load(SAMPLE_DZ)
    WAVEFORM_BZ_ORIGINAL, SAMPLE_RATE_BZ_ORIGINAL = torchaudio.load(Original_BZ)
    WAVEFORM_DZ_ORIGINAL, SAMPLE_RATE_DZ_ORIGINAL = torchaudio.load(Original_DZ)
    # Convert to mono if needed
    WAVEFORM_DRY = WAVEFORM_DRY[0:1, :]
    WAVEFORM_DZ = WAVEFORM_DZ[0:1, :]
    WAVEFORM_BZ = WAVEFORM_BZ[0:1, :]
    Original_DZ = WAVEFORM_DZ_ORIGINAL[0:1, :]
    Original_BZ = WAVEFORM_BZ_ORIGINAL[0:1, :]

    # Resample to 16kHz
    if SAMPLE_RATE_DRY != 16000:
        WAVEFORM_DRY = F.resample(WAVEFORM_DRY, SAMPLE_RATE_DRY, 16000)
    if SAMPLE_RATE_BZ != 16000:
        WAVEFORM_BZ = F.resample(WAVEFORM_BZ, SAMPLE_RATE_BZ, 16000)
    if SAMPLE_RATE_DZ != 16000:
        WAVEFORM_DZ = F.resample(WAVEFORM_DZ, SAMPLE_RATE_DZ, 16000)
    if SAMPLE_RATE_BZ_ORIGINAL != 16000:
        Original_BZ = F.resample(Original_BZ, SAMPLE_RATE_BZ_ORIGINAL, 16000)
    if SAMPLE_RATE_DZ_ORIGINAL != 16000:
        Original_DZ = F.resample(Original_DZ, SAMPLE_RATE_DZ_ORIGINAL, 16000)

    # Trim to shortest length
    min_len = min(WAVEFORM_DRY.shape[1], WAVEFORM_BZ.shape[1], WAVEFORM_DZ.shape[1], Original_BZ.shape[1], Original_DZ.shape[1])
    WAVEFORM_DRY = WAVEFORM_DRY[:, :min_len]
    WAVEFORM_BZ = WAVEFORM_BZ[:, :min_len]
    WAVEFORM_DZ = WAVEFORM_DZ[:, :min_len]
    Original_BZ = Original_BZ[:, :min_len]
    Original_DZ = Original_DZ[:, :min_len]

    objective_model = SQUIM_OBJECTIVE.get_model()


    # Original DZ
    squim_og_stoi_dz, squim_og_pesq_dz, squim_og_si_sdr_dz = objective_model(Original_DZ[0:1, :])
    squim_og_dz_mos = evaluate_mos(Original_DZ, WAVEFORM_DRY)
    og_stoi_dz, og_pesq_dz = evaluate_stoi_pesq(Original_DZ, WAVEFORM_DRY)

    # Filtered DZ
    squim_filtered_stoi_dz, squim_filtered_pesq_dz, squim_filtered_si_sdr_dz = objective_model(WAVEFORM_DZ[0:1, :])
    squim_filtered_dz_mos = evaluate_mos(WAVEFORM_DZ, WAVEFORM_DRY)
    filtered_stoi_dz, filtered_pesq_dz = evaluate_stoi_pesq(WAVEFORM_DZ, WAVEFORM_DRY)

    # Original BZ
    squim_og_stoi_bz, squim_og_pesq_bz, squim_og_si_sdr_bz = objective_model(Original_BZ[0:1, :])
    squim_og_bz_mos = evaluate_mos(Original_BZ, WAVEFORM_DRY)
    og_stoi_bz, og_pesq_bz = evaluate_stoi_pesq(Original_BZ, WAVEFORM_DRY)


    # Filtered BZ
    squim_filtered_stoi_bz, squim_filtered_pesq_bz, squim_filtered_si_sdr_bz = objective_model(WAVEFORM_BZ[0:1, :]) # TODO: Ensure this works
    squim_filtered_bz_mos = evaluate_mos(WAVEFORM_BZ, WAVEFORM_DRY)
    filtered_stoi_bz, filtered_pesq_bz = evaluate_stoi_pesq(WAVEFORM_BZ, WAVEFORM_DRY)

    return {
        "Original_BZ": {
            "Squim_STOI": squim_og_stoi_bz,
            "Squim_PESQ": squim_og_pesq_bz,
            "STOI": og_stoi_bz,
            "PESQ": og_pesq_bz,
            "MOS": squim_og_bz_mos,
        },
        "Filtered_BZ": {
            "Squim_STOI": squim_filtered_stoi_bz,
            "Squim_PESQ": squim_filtered_pesq_bz,
            "STOI": filtered_stoi_bz,
            "PESQ": filtered_pesq_bz,
            "MOS": squim_filtered_bz_mos,
        },
        "Original_DZ": {
            "Squim_STOI": squim_og_stoi_dz,
            "Squim_PESQ": squim_og_pesq_dz,
            "STOI": og_stoi_dz,
            "PESQ": og_pesq_dz,
            "MOS": squim_og_dz_mos,
        },
        "Filtered_DZ": {
            "Squim_STOI": squim_filtered_stoi_dz,
            "Squim_PESQ": squim_filtered_pesq_dz,
            "STOI": filtered_stoi_dz,
            "PESQ": filtered_pesq_dz,
            "MOS": squim_filtered_dz_mos,
        },
    }


# Example usage
if __name__ == "__main__":
    # Load the audio files
    Dry_sound = r"relaxing-guitar-loop-v5-245859.wav"  # Replace with the path to your dry sound file
    SAMPLE_BZ = r"bright_mic_filtered.wav"  # Replace with the path to your bright mic filtered sound file
    SAMPLE_DZ = r"dark_mic_filtered.wav"  # Replace with the path to your dark mic filtered sound file
    Original_BZ = r"bright_mic_original.wav"  # Replace with the path to your bright mic original sound file
    Original_DZ = r"dark_mic_original.wav"  # Replace with the path to your dark mic original sound file

    # Evaluate the signals
    print(evaluate_signals(Dry_sound, SAMPLE_BZ, SAMPLE_DZ, Original_BZ, Original_DZ))