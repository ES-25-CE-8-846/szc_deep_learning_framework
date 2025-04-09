import torch
import torchaudio
import wandb


class Trainer:
    def __init__(self, model, dataloader, args) -> None:
        self.model = model
        self.dataloader = dataloader

    def run_epoch(self):
        for data_dict in self.dataloader:
            pass
