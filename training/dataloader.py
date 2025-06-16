from re import split
from typing_extensions import override
import torch
import torchaudio
import os
import pyflac
import warnings
import soundfile
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import shutil

class DefaultDataset(Dataset):
    def __init__(self,
                 sound_dataset_root,
                 rir_dataset_root,
                 transforms = None,
                 sound_snip_len = 200,
                 sound_snip_save_path = '/tmp/sound_snips',
                 limit_used_soundclips = 1000,
                 rir_fixed_length = 4096,
                 override_existing = False,
                 filter_by_std = None,
                 filter_by_mean = None,
                 load_pre_computed_filters = False,
                 filter_dirname = "filters_4096_1.0_1e-05",
                 return_filter_save_path = False,
                 sound_samplerate=44100):

        self.sound_dataroot = sound_dataset_root
        self.rir_dataroot = rir_dataset_root
        self.transforms = transforms
        self.sound_snip_save_path = sound_snip_save_path
        self.limit_used_soundclips = limit_used_soundclips
        self.sound_snip_len = sound_snip_len
        self.sound_samplerate = sound_samplerate

        self.bz_rirs_shape = None
        self.dz_rirs_shape = None
        self.rir_fixed_length = rir_fixed_length

        self.filter_by_std = filter_by_std
        self.filter_by_mean = filter_by_mean

        self.load_pre_computed_filters = load_pre_computed_filters
        self.filter_dirname = filter_dirname

        self.return_filter_save_path = return_filter_save_path

        if override_existing:
            shutil.rmtree(sound_snip_save_path)


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
            removed_snip_counter = 0

            # for sound_file in tqdm(sorted(sound_flac_files)):
                # data, sr = soundfile.read(sound_file)
                #
                # slen = int((sound_snip_len/1000) * sr)
                # n_snips = len(data) // slen
                # for snip in range(n_snips):
                #     data_snip = data[snip*slen : (snip + 1) * slen]
                #     data_sound_tensor = torch.tensor(data_snip)
                #
                #     if self.filter_snippets(filter_by_std=filter_by_std, filter_by_mean=filter_by_mean, sound_tensor=data_sound_tensor):
                #         fn = f'{snip_counter}'.rjust(10,'0') + '.wav'
                #         soundfile.write(file=os.path.join(sound_snip_save_path, fn), data=data_snip, samplerate=sr)
                #         snip_counter += 1
                #     else:
                #         removed_snip_counter += 1
                #
                # if limit_used_soundclips and snip_counter > limit_used_soundclips:
                #     break

            for sound_file in tqdm(sorted(sound_flac_files)):
                data, sr = soundfile.read(sound_file)

                # Resample only if needed
                if sr != self.sound_samplerate:
                    data_tensor = torch.tensor(data, dtype=torch.float32)
                    if data_tensor.ndim == 1:
                        data_tensor = data_tensor.unsqueeze(0)  # [1, time]
                    else:
                        data_tensor = data_tensor.T  # [channels, time]

                    data_tensor = torchaudio.functional.resample(data_tensor, orig_freq=sr, new_freq=self.sound_samplerate)
                    data = data_tensor.squeeze().T.numpy()
                    sr = self.sound_samplerate

                slen = int((sound_snip_len / 1000) * sr)
                n_snips = len(data) // slen
                for snip in range(n_snips):
                    data_snip = data[snip * slen : (snip + 1) * slen]
                    data_sound_tensor = torch.tensor(data_snip)

                    if self.filter_snippets(filter_by_std=filter_by_std, filter_by_mean=filter_by_mean, sound_tensor=data_sound_tensor):
                        fn = f'{snip_counter}'.rjust(10, '0') + '.wav'
                        soundfile.write(file=os.path.join(sound_snip_save_path, fn), data=data_snip, samplerate=sr)
                        snip_counter += 1
                    else:
                        removed_snip_counter += 1

                if limit_used_soundclips and snip_counter > limit_used_soundclips:
                    break

            print(snip_counter, removed_snip_counter)
            self.n_sound_snips = snip_counter
        else:
            warnings.warn('No new files added, remove the existing dataset or change the sound_snip_save_path', UserWarning)
            self.n_sound_snips = len(os.listdir(sound_snip_save_path))

        ### something rir
        self.rir_files = []
        for root, directory, files in os.walk(self.rir_dataroot):
            # print(directory)
            if directory != self.filter_dirname:
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = file.split(".")[-1]
                    file_dir = root.split('/')[-1]
                    if file_extension == "npz" and file_dir != self.filter_dirname:
                        self.rir_files.append(file_path)
                        # print(file_path)






        self.n_rirs = len(self.rir_files)

        if limit_used_soundclips:
            if limit_used_soundclips < self.n_sound_snips:
                self.n_sound_snips = limit_used_soundclips

        self.sound_clip_inicies = np.arange(self.n_sound_snips)

        print(f"n rirs {self.n_rirs}")

    def filter_snippets(self, filter_by_std, filter_by_mean, sound_tensor):
        std = torch.std(sound_tensor)
        mean = torch.mean(torch.abs(sound_tensor))

        # print(f"mean {mean.item()}, std {std.item()}" )


        is_good = False

        if filter_by_std is not None and filter_by_mean is not None:
            is_good = std > filter_by_std and mean > filter_by_mean
        elif filter_by_std is not None:
            is_good = std > filter_by_std
        elif filter_by_mean is not None:
            is_good = mean > filter_by_mean
        else:
            is_good = True

        return is_good



    def __len__(self):
        return self.n_rirs -1


    def __getitem__(self, index):


        rir_index = index
        sound_index = np.random.choice(self.n_sound_snips)

        sound_fp = os.path.join(self.sound_snip_save_path, f'{sound_index}'.rjust(10,'0') + '.wav')
        sound, sr = torchaudio.load(sound_fp)

        sound_len = sr * self.sound_snip_len
        sound = sound[:sound_len]

        # get impulse responses
        rirs = np.load(self.rir_files[rir_index])

        # get filters
        if self.load_pre_computed_filters:
            split_path = self.rir_files[index].split('/')
            split_path.insert(-1, self.filter_dirname)
            filter_path = "/" + os.path.join(*split_path)
            filters = np.load(filter_path)


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

            data_dict = {'sound':sound, 'bz_rirs':bz_rirs_ext, 'dz_rirs':dz_rirs_ext, 'sr':sr}
        else:
            data_dict = {'sound':sound, 'bz_rirs':bz_rirs, 'dz_rirs':dz_rirs, 'sr':sr}


        if self.load_pre_computed_filters:
            data_dict['precomp_filters'] = filters

        if self.return_filter_save_path:
            data_dict['filter_save_path'] = filter_path

        return data_dict
