from torch._subclasses.fake_tensor import torch_decomp_decompositions
from training import dataloader
import torch
from tqdm import tqdm
if __name__ == "__main__":
    test_dataloader = dataloader.DefaultDataset(sound_dataset_root='/home/ai/datasets/audio/LibriSpeech/train-clean-100/',
                                                rir_dataset_root='/home/ai/datasets/audio/test_rirs/dataset/shoebox/alfredo-request/test/')
    print(len(test_dataloader))
    data_dict = test_dataloader[2300]

    print(data_dict)

    torch_dataloader = torch.utils.data.DataLoader(dataset=test_dataloader, batch_size=16)

    for data_dict in tqdm(torch_dataloader):
        sound = data_dict["sound"]
        bz_rirs = data_dict["bz_rirs"]
        dz_rirs = data_dict["dz_rirs"]
        
        print(f'sound shape {sound.size()}')
        print(f'bz rirs {bz_rirs.size()}')
        print(f'dz rirs {dz_rirs.size()}')
        

        

