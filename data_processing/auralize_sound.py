import numpy as np 
from scipy.signal import fftconvolve
from torch._subclasses.functional_tensor import return_and_correct_aliasing


class AuraFarmer:
    def __init__(self, max_length, rirs):
        self.max_length = max_length 
        self.rirs = rirs

    def auralize(self, sound):
        sound = np.ones_like(self.rirs) * sound
        auralized_sound = fftconvolve(sound, self.rirs, axes=0)
        return auralized_sound
