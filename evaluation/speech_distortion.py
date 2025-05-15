import torch
import torchaudio

from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import torchaudio.functional as F


def speech_distortion(sound):
    """Function to compute the speech distortion based on torchaudio squim
    Args:
        sound: should be filtered and auralized sound of a person speaking
    """
