import torch
import torchaudio


class DefaultDataset:
    def __init__(self, dataset_root, transforms):
        self.dataroot = dataset_root
        self.transforms = transforms

    def __getitem__(self):
        data_dict = {}

        # get impulse responses

        # get audio

        return data_dict
