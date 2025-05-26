import torch
from pathlib import Path
import torchaudio
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import torchaudio.functional as F
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf
from tqdm import tqdm

from intelligibility_tools import evaluate_stoi_pesq, evaluate_mos


def extract_combined_filters(filters_path: Path):
    """
    Extracts filters from a .npz file and returns them as a dictionary.
    Args:
        filters_path (Path): Path to the .npz file containing the filters.
    Returns:
        q_acc: Array of filters for acoustic contrast control.
        q_vast: Array of filters for vast.
        q_pm: Array of filters for pressure matching.
    """
    if not filters_path.exists():
        raise FileNotFoundError(f"Filters file not found at {filters_path}")
    if not filters_path.suffix == '.npz':
        raise ValueError(f"Expected a .npz file, but got {filters_path.suffix}")
    
    # Load the filters from the .npz file
    filters_npz = np.load(filters_path, allow_pickle=True)

    # Extract all arrays from the npz file
    filters = {}
    for file_name in filters_npz.files:
        filters[file_name] = filters_npz[file_name]
        print(f"Extracted {file_name}, shape: {filters[file_name].shape if hasattr(filters[file_name], 'shape') else 'No shape'}")
    
    # Close the npz file after extraction
    filters_npz.close()

    return filters


def auralize_audio(dry_sound, impulse_responses):
    """
    Auralizes the dry sound using the provided impulse responses.
    
    Args:
        dry_sound (Tensor): The dry sound audio signal.
        impulse_responses (Tensor): Tensor with impulse response tensors from each speaker.
    
    Returns:
        Tensor: Auralized audio signal.
    """
    # Ensure dry_sound is a 1D tensor
    if dry_sound.ndim > 1:
        dry_sound = dry_sound.mean(dim=0)

    # Convolve the dry sound with each impulse response
    auralized_signal = sum(
        fftconvolve(dry_sound.numpy(), ir, mode='full')
        for ir in impulse_responses
    )

    # Convert back to tensor and normalize
    auralized_signal = torch.tensor(auralized_signal)
    auralized_signal /= torch.max(torch.abs(auralized_signal))

    return auralized_signal

def apply_filters(dry_sound: torch.Tensor, filters: np.ndarray) -> torch.Tensor:
    """
    Apply a set of time-domain filters to a dry mono audio signal.

    Args:
        dry_sound (Tensor): 1D audio tensor (mono).
        filters (ndarray): 2D array of shape [num_filters, filter_length].

    Returns:
        Tensor: Summed signal after filtering.
    """
    if dry_sound.ndim > 1:
        dry_sound = dry_sound.mean(dim=0)

    # Convolve dry signal with each filter
    filtered_signals = [
        fftconvolve(dry_sound, filt, mode='full')
        for filt in filters
    ]

    # Sum all filtered signals
    output = sum(filtered_signals)

    # Normalize
    output = output / np.max(np.abs(output))

    return torch.tensor(output, dtype=torch.float32)



def create_soundfiles(output_path, q_acc, q_vast, q_pm, dry_sound, bz_rir, dz_rir):
    """
    Create sound files for the different filters applied to the dry sound.
    Args:
        output_path (Path): Path to save the sound files.
        q_acc (np.ndarray): Filters for acoustic contrast control.
        q_vast (np.ndarray): Filters for vast.
        q_pm (np.ndarray): Filters for pressure matching.
        dry_sound (torch.Tensor): The dry sound audio signal.
        bz_rir (list): List of impulse responses for the bright mic.
        dz_rir (list): List of impulse responses for the dark mic.
    """

    # Auralize the dry sound using the provided impulse responses
    original_bz = auralize_audio(dry_sound, bz_rir[0])
    original_dz = auralize_audio(dry_sound, dz_rir[-1])

    # Apply the filters to the auralized audio
    filtered_acc_bz = apply_filters(original_bz.numpy(), q_acc)
    filtered_acc_dz = apply_filters(original_dz.numpy(), q_acc)

    filtered_vast_bz = apply_filters(original_bz.numpy(), q_vast)
    filtered_vast_dz = apply_filters(original_dz.numpy(), q_vast)

    filtered_pm_bz = apply_filters(original_bz.numpy(), q_pm)
    filtered_pm_dz = apply_filters(original_dz.numpy(), q_pm)

    # Save sounds with the dataset
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    sf.write(output_path / "original_bz.wav", original_bz.numpy(), 44100)
    sf.write(output_path / "original_dz.wav", original_dz.numpy(), 44100)
    sf.write(output_path / "filtered_acc_bz.wav", filtered_acc_bz.numpy(), 44100)
    sf.write(output_path / "filtered_acc_dz.wav", filtered_acc_dz.numpy(), 44100)
    sf.write(output_path / "filtered_vast_bz.wav", filtered_vast_bz.numpy(), 44100)
    sf.write(output_path / "filtered_vast_dz.wav", filtered_vast_dz.numpy(), 44100)
    sf.write(output_path / "filtered_pm_bz.wav", filtered_pm_bz.numpy(), 44100)
    sf.write(output_path / "filtered_pm_dz.wav", filtered_pm_dz.numpy(), 44100)


def evaluate_position_from_variable(position_path: Path, dry_sound, original_bz, original_dz, filtered_acc_bz, filtered_acc_dz, filtered_vast_bz, filtered_vast_dz, filtered_pm_bz, filtered_pm_dz, filters):
    """Evaluate the intelligibility of the sound files at the test position.
    """

    # Calculate metrics for the original signals
    og_stoi_bz, og_pesq_bz = evaluate_stoi_pesq(original_bz, dry_sound, sample_rate=44100)
    og_stoi_dz, og_pesq_dz = evaluate_stoi_pesq(original_dz, dry_sound, sample_rate=44100)
    squim_og_bz_mos = evaluate_mos(original_bz, dry_sound)
    squim_og_dz_mos = evaluate_mos(original_dz, dry_sound)

    # Calculate metrics for the filtered signals
    acc_filtered_stoi_bz, acc_filtered_pesq_bz = evaluate_stoi_pesq(filtered_acc_bz, dry_sound, sample_rate=44100)
    acc_filtered_stoi_dz, acc_filtered_pesq_dz = evaluate_stoi_pesq(filtered_acc_dz, dry_sound, sample_rate=44100)
    acc_squim_filtered_bz_mos = evaluate_mos(filtered_acc_bz, dry_sound)
    acc_squim_filtered_dz_mos = evaluate_mos(filtered_acc_dz, dry_sound)
    
    # Calculate metrics for the filtered signals
    vast_filtered_stoi_bz, vast_filtered_pesq_bz = evaluate_stoi_pesq(filtered_vast_bz, dry_sound, sample_rate=44100)
    vast_filtered_stoi_dz, vast_filtered_pesq_dz = evaluate_stoi_pesq(filtered_vast_dz, dry_sound, sample_rate=44100)
    vast_squim_filtered_bz_mos = evaluate_mos(filtered_vast_bz, dry_sound)
    vast_squim_filtered_dz_mos = evaluate_mos(filtered_vast_dz, dry_sound)

    # Calculate metrics for the filtered signals
    pm_filtered_stoi_bz, pm_filtered_pesq_bz = evaluate_stoi_pesq(filtered_pm_bz, dry_sound, sample_rate=44100)
    pm_filtered_stoi_dz, pm_filtered_pesq_dz = evaluate_stoi_pesq(filtered_pm_dz, dry_sound, sample_rate=44100)
    pm_squim_filtered_bz_mos = evaluate_mos(filtered_pm_bz, dry_sound)
    pm_squim_filtered_dz_mos = evaluate_mos(filtered_pm_dz, dry_sound)

    # Save the results in a text file
    position_path = Path(position_path)
    position_path.mkdir(parents=True, exist_ok=True)

    results_path = position_path / "evaluation_results.txt"

    with open(results_path, 'w') as f:
        f.write(f"Evaluation results for position: {position_path.name}\n")
        f.write(f"Original signal:\n")
        f.write(f"Original BZ - STOI: {og_stoi_bz}, PESQ: {og_pesq_bz}, MOS: {squim_og_bz_mos}\n")
        f.write(f"Original DZ - STOI: {og_stoi_dz}, PESQ: {og_pesq_dz}, MOS: {squim_og_dz_mos}\n")

        f.write("\nResults for ACC Filters:\n")
        f.write(f"Filtered BZ - STOI: {acc_filtered_stoi_bz}, PESQ: {acc_filtered_pesq_bz}, MOS: {acc_squim_filtered_bz_mos}\n")
        f.write(f"Filtered DZ - STOI: {acc_filtered_stoi_dz}, PESQ: {acc_filtered_pesq_dz}, MOS: {acc_squim_filtered_dz_mos}\n")

        f.write("\nResults for VAST Filters:\n")
        f.write(f"Filtered BZ - STOI: {vast_filtered_stoi_bz}, PESQ: {vast_filtered_pesq_bz}, MOS: {vast_squim_filtered_bz_mos}\n")
        f.write(f"Filtered DZ - STOI: {vast_filtered_stoi_dz}, PESQ: {vast_filtered_pesq_dz}, MOS: {vast_squim_filtered_dz_mos}\n")

        f.write("\nResults for PM Filters:\n")
        f.write(f"Filtered BZ - STOI: {pm_filtered_stoi_bz}, PESQ: {pm_filtered_pesq_bz}, MOS: {pm_squim_filtered_bz_mos}\n")
        f.write(f"Filtered DZ - STOI: {pm_filtered_stoi_dz}, PESQ: {pm_filtered_pesq_dz}, MOS: {pm_squim_filtered_dz_mos}\n")

    print(f"Evaluation results from {position.name} saved to {results_path}")


def run_test_for_position(position_path: Path, dry_sound_path: Path, filters):
    """
    Run the evaluation for a specific position.
    
    Args:
        position_path (Path): Path to the .npz file containing impulse responses for a position.
        dry_sound_path (Path): Path to the dry sound file.
    """

    # Load the dry sound
    dry_sound, sample_rate = torchaudio.load(dry_sound_path)
    if dry_sound.shape[0] > 1:
        dry_sound = dry_sound.mean(dim=0, keepdim=True)  # Convert to mono if stereo
        dry_sound = dry_sound.squeeze(0)  # Remove channel dimension

    # Load the RIRs from a position
    rirs = np.load(f"{position_path}", allow_pickle=True)
    dz_rir, bz_rir = rirs['dz_rir'], rirs['bz_rir']


    # Create auralized audio for the dry sound using the RIRs
    original_bz = auralize_audio(dry_sound, bz_rir[0])
    original_dz = auralize_audio(dry_sound, dz_rir[-1])

    # Apply the filters to the auralized audio
    filtered_acc_bz = apply_filters(original_bz.numpy(), filters['q_acc'])
    filtered_acc_dz = apply_filters(original_dz.numpy(), filters['q_acc'])

    filtered_vast_bz = apply_filters(original_bz.numpy(), filters['q_vast'])
    filtered_vast_dz = apply_filters(original_dz.numpy(), filters['q_vast'])

    filtered_pm_bz = apply_filters(original_bz.numpy(), filters['q_pm'])
    filtered_pm_dz = apply_filters(original_dz.numpy(), filters['q_pm'])

    # Path to the evaluation of the position
    save_path = Path(f"{position_path.parent.parent}/evaluation/{position_path.parent.name}/{position_path.stem}")


    evaluate_position_from_variable(save_path, dry_sound, original_bz, original_dz, filtered_acc_bz, filtered_acc_dz, filtered_vast_bz, filtered_vast_dz, filtered_pm_bz, filtered_pm_dz, filters)

    # Path to the evaluation of the position
    #save_path = Path(f"{position_path.parent.parent}/evaluation/{position_path.parent.name}/{position_path.stem}")

    # Create the sound files
    #create_soundfiles(save_path, filters['q_acc'], filters['q_vast'], filters['q_pm'], dry_sound, bz_rir, dz_rir)

    # Evaluate the sound files
    #evaluate_position(save_path, dry_sound_path)

if __name__ == "__main__":
    # Paths to dataset split, filters, and dry sound
    dataset_split_path = Path("/home/ubuntu/shoebox/run2/test")
    filters_path = Path("/home/ubuntu/combined_filters.npz")
    dry_sound_path = Path("/home/ubuntu/concatenated_test_audio_44100.wav")

    # Load the filters
    filters = extract_combined_filters(filters_path)

    # Test a room
    for room in tqdm(sorted(dataset_split_path.glob("room_*"))):
        print(f"Running tests for room: {room.name}")
        position = room / "0000.npz"
        run_test_for_position(position, dry_sound_path, filters)

        # for position in tqdm(sorted(room.glob("*.npz"))):
        #     print(f"  - Position: {position.name}")
            # Run the test for each position
            
    
