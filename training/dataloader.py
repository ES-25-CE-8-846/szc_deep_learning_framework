import torch
import torchaudio
import os
import pyflac
import warnings
import soundfile
from tqdm import tqdm
import numpy as np 
from torch.utils.data import Dataset

class DefaultDataset(Dataset):
    def __init__(self, 
                 sound_dataset_root, 
                 rir_dataset_root, 
                 transforms = None,
                 sound_snip_len = 200, 
                 sound_snip_save_path = '/tmp/sound_snips',
                 limit_used_soundclips = 1000,
                 rir_fixed_length = 4096):
        
        self.sound_dataroot = sound_dataset_root
        self.rir_dataroot = rir_dataset_root
        self.transforms = transforms
        self.sound_snip_save_path = sound_snip_save_path
        self.limit_used_soundclips = limit_used_soundclips  
        
        self.bz_rirs_shape = None
        self.dz_rirs_shape = None
        self.rir_fixed_length = rir_fixed_length

        ### sort sound dataset 
        
        if not os.path.exists(sound_snip_save_path):
            os.makedirs(sound_snip_save_path)

            sound_flac_files = [] 
            for root, directory, files in os.walk(self.sound_dataroot):
                for file in files:
                    file_path = os.path.join(root, file) 
                    file_extension = file.split(".")[-1]
         
                    if file_extension == "flac":
                        sound_flac_files.append(file_path)
                    else:
                        warnings.warn(f"file type .{file_extension} is not yet supported", UserWarning)
            snip_counter = 0
            for sound_file in tqdm(sorted(sound_flac_files)):
                data, sr = soundfile.read(sound_file)
                
                slen = int((sound_snip_len/1000) * sr)
                n_snips = len(data) // slen
                for snip in range(n_snips):
                    data_snip = data[snip*slen : (snip + 1) * slen]
                    fn = f'{snip_counter}'.rjust(10,'0') + '.wav'
                    soundfile.write(file=os.path.join(sound_snip_save_path, fn), data=data_snip, samplerate=sr)
                    snip_counter += 1 

            print(snip_counter)
            self.n_sound_snips = snip_counter
        else:
            warnings.warn('No new files added, remove the existing dataset or change the sound_snip_save_path', UserWarning)
            self.n_sound_snips = len(os.listdir(sound_snip_save_path))

        ### something rir 
        self.rir_files = []
        for root, directory, files in os.walk(self.rir_dataroot):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = file.split(".")[-1]
                if file_extension == "npz":
                    self.rir_files.append(file_path)

        self.n_rirs = len(self.rir_files)  

        if limit_used_soundclips:
            self.n_sound_snips = limit_used_soundclips

    def __len__(self):
        return self.n_rirs * self.n_sound_snips 


    def __getitem__(self, index):

        grid_index = np.unravel_index(index, (self.n_rirs, self.n_sound_snips)) 
        
        rir_index = grid_index[0]
        sound_index = grid_index[1]
        
        sound_fp = os.path.join(self.sound_snip_save_path, f'{sound_index}'.rjust(10,'0') + '.wav')
        sound, sr= torchaudio.load(sound_fp)

        # get impulse responses
        rirs = np.load(self.rir_files[rir_index])
        
        bz_rirs = rirs["bz_rir"]
        dz_rirs = rirs["dz_rir"]

        ## the following is to ensure that the length of each impulese response is the same 
        if self.rir_fixed_length:
            if (not self.bz_rirs_shape):
                self.bz_rirs_shape = (bz_rirs.shape[0],
                                      bz_rirs.shape[1],
                                      self.rir_fixed_length)

                self.dz_rirs_shape = (dz_rirs.shape[0],
                                      dz_rirs.shape[1],
                                      self.rir_fixed_length)
                print(self.dz_rirs_shape)
            
            if bz_rirs.shape[2] > self.rir_fixed_length:
                bz_rirs_ext = bz_rirs[:,:,:self.rir_fixed_length]
            else:
                bz_rirs_ext = np.zeros(self.bz_rirs_shape)
                bz_rirs_ext[:,:,:bz_rirs.shape[2]] = bz_rirs
            
            if dz_rirs.shape[2] > self.rir_fixed_length:
                dz_rirs_ext = dz_rirs[:,:,:self.rir_fixed_length]
            else:
                dz_rirs_ext = np.zeros(self.dz_rirs_shape)
                dz_rirs_ext[:,:,:dz_rirs.shape[2]] = dz_rirs
            
            data_dict = {'sound':sound, 'bz_rirs':bz_rirs_ext, 'dz_rirs':dz_rirs_ext}
        else:
            data_dict = {'sound':sound, 'bz_rirs':bz_rirs, 'dz_rirs':dz_rirs}

        return data_dict
