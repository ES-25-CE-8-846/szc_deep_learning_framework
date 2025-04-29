from ctypes.util import test
from cffi.api import model
from torch._subclasses.fake_tensor import torch_decomp_decompositions
from torch.nn.modules import loss
from training import dataloader
import torch
from tqdm import tqdm
import unittest
import shutil
from training import trainer
from training import loss_functions
import numpy as np
import models
import soundfile
import torchinfo


class TestDataloading(unittest.TestCase):
    def test_parsing(self):

        sound_snips_len_ms = 1000

        self.test_dataloader = dataloader.DefaultDataset(
            sound_dataset_root="./testing_data/audio_raw/",
            rir_dataset_root="./testing_data/rirs/test_rirs/dataset/shoebox/alfredo-request/test/",
            sound_snip_len=sound_snips_len_ms,
            override_existing=True,
        )

        self.assertTrue(self.test_dataloader.__len__() > 0)
        data_dict = self.test_dataloader[0]

        self.assertIn("sound", data_dict.keys())
        self.assertIn("dz_rirs", data_dict.keys())
        self.assertIn("bz_rirs", data_dict.keys())
        self.assertIn("sr", data_dict.keys())

        self.assertTrue(
            self.test_dataloader[0]["sound"].size()[1]
            == self.test_dataloader[0]["sr"] * (sound_snips_len_ms / 1000)
        )

    def test_dataloader_torch_integrarion(self):

        sound_snips_len_ms = 1000
        self.test_dataloader = dataloader.DefaultDataset(
            sound_dataset_root="./testing_data/audio_raw/",
            rir_dataset_root="./testing_data/rirs/test_rirs/dataset/shoebox/alfredo-request/test/",
            sound_snip_len=sound_snips_len_ms,
            override_existing=True,
        )

        torch_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_dataloader, batch_size=16
        )

        for data_dict in tqdm(torch_dataloader):
            self.assertIn("sound", data_dict.keys())
            self.assertIn("dz_rirs", data_dict.keys())
            self.assertIn("bz_rirs", data_dict.keys())
            self.assertIn("sr", data_dict.keys())

    def test_auralizer(self):
        sound_tensor = torch.rand((16, 3, 16000))
        rirs = torch.rand((16, 3, 3, 4096))

        sound_snips_len_ms = 1000
        self.test_dataloader = dataloader.DefaultDataset(
            sound_dataset_root="./testing_data/audio_raw/",
            rir_dataset_root="./testing_data/rirs/test_rirs/dataset/shoebox/alfredo-request/test/",
            sound_snip_len=sound_snips_len_ms,
            override_existing=True,
        )

        torch_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_dataloader, batch_size=16
        )

        test_trainer = trainer.Trainer(
            dataloader=torch_dataloader, loss_function=None, model=None
        )

        test_trainer.auralizer(sound_tensor, rirs)

    def test_filter_apply(self):

        sound_snips_len_ms = 1000
        self.test_dataloader = dataloader.DefaultDataset(
            sound_dataset_root="./testing_data/audio_raw/",
            rir_dataset_root="./testing_data/rirs/test_rirs/dataset/shoebox/alfredo-request/test/",
            sound_snip_len=sound_snips_len_ms,
            override_existing=True,
        )

        torch_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_dataloader, batch_size=16
        )
        test_trainer = trainer.Trainer(
            dataloader=torch_dataloader, loss_function=None, model=None
        )

        for i, data_dict in enumerate(torch_dataloader):
            fl = test_trainer.filter_length

            filters = torch.ones((16, 3, fl))
            sound = data_dict["sound"]

            filtered_sound = test_trainer.apply_filter(sound, filters)
            print(f"filtered sound shape {filtered_sound.size()} ")
            print(f"input sound shape {sound.size()}")

            if i > 2:
                break

    def test_filter_apply_and_auralize(self):

        sound_snips_len_ms = 1000
        self.test_dataloader = dataloader.DefaultDataset(
            sound_dataset_root="./testing_data/audio_raw/",
            rir_dataset_root="./testing_data/rirs/test_rirs/dataset/shoebox/alfredo-request/test/",
            sound_snip_len=sound_snips_len_ms,
            override_existing=True,
        )

        torch_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_dataloader, batch_size=16
        )
        test_trainer = trainer.Trainer(
            dataloader=torch_dataloader, loss_function=None, model=None
        )

        for i, data_dict in enumerate(torch_dataloader):
            fl = test_trainer.filter_length

            filters = torch.ones((16, 3, fl))
            sound = data_dict["sound"]

            bz_rirs = data_dict["bz_rirs"]
            dz_rirs = data_dict["dz_rirs"]

            filtered_sound = test_trainer.apply_filter(sound, filters)

            auralized_sound_bz = test_trainer.auralizer(filtered_sound, bz_rirs)
            auralized_sound_dz = test_trainer.auralizer(filtered_sound, dz_rirs)

            print(f"input sound shape {sound.size()}")
            print(f"input rir shapes {bz_rirs.size()} bz, {dz_rirs.size()} dz")
            print(f"filtered sound shape {filtered_sound.size()} ")
            print(
                f"auralized sound shape {auralized_sound_bz.size()} bz, {auralized_sound_dz.size()} dz "
            )

            if i > 2:
                break

    def test_sound_filter(self):
        sound_snips_len_ms = 500
        test_dataloader = dataloader.DefaultDataset(
            sound_dataset_root="./testing_data/audio_raw/",
            rir_dataset_root="./testing_data/rirs/test_rirs/dataset/shoebox/alfredo-request/test/",
            sound_snip_len=sound_snips_len_ms,
            override_existing=True,
            filter_by_std=0.02,
            filter_by_mean=0.02,
        )

        test_sound_1 = torch.tensor(np.random.normal(loc=0, scale=2, size=100))
        test_sound_2 = torch.tensor(np.random.normal(loc=2, scale=0.5, size=100))
        test_sound_3 = torch.tensor(np.random.normal(loc=0.1, scale=0.1, size=100))

        self.assertTrue(
            test_dataloader.filter_snippets(
                filter_by_std=1.0, filter_by_mean=1.0, sound_tensor=test_sound_1
            )
        )
        self.assertTrue(
            test_dataloader.filter_snippets(
                filter_by_std=None, filter_by_mean=1.0, sound_tensor=test_sound_1
            )
        )
        self.assertTrue(
            test_dataloader.filter_snippets(
                filter_by_std=1.0, filter_by_mean=None, sound_tensor=test_sound_1
            )
        )

        self.assertFalse(
            test_dataloader.filter_snippets(
                filter_by_std=1.0, filter_by_mean=1.0, sound_tensor=test_sound_3
            )
        )
        self.assertFalse(
            test_dataloader.filter_snippets(
                filter_by_std=None, filter_by_mean=1.0, sound_tensor=test_sound_3
            )
        )
        self.assertFalse(
            test_dataloader.filter_snippets(
                filter_by_std=1.0, filter_by_mean=None, sound_tensor=test_sound_3
            )
        )

        self.assertFalse(
            test_dataloader.filter_snippets(
                filter_by_std=1.0, filter_by_mean=1.0, sound_tensor=test_sound_2
            )
        )
        self.assertFalse(
            test_dataloader.filter_snippets(
                filter_by_std=1.0, filter_by_mean=None, sound_tensor=test_sound_2
            )
        )
        self.assertTrue(
            test_dataloader.filter_snippets(
                filter_by_std=None, filter_by_mean=1.0, sound_tensor=test_sound_2
            )
        )

        for sound_i in range(10):
            sound = test_dataloader[sound_i]["sound"]
            sr = test_dataloader[sound_i]["sr"]
            print(
                f"std {(torch.std(sound).item())} mean {(torch.mean(abs(sound)).item())}"
            )
            # soundfile.write(file = f"./std{int(torch.std(sound).item()*10000)}mean{int(torch.mean(sound).item()*10000)}.wav", data= sound.ravel(), samplerate=sr)


class TestTrainer(unittest.TestCase):
    def test_run_epoch(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(device)

        sound_snips_len_ms = 500
        self.test_dataloader = dataloader.DefaultDataset(
            sound_dataset_root="./testing_data/audio_raw/",
            rir_dataset_root="./testing_data/rirs/test_rirs/dataset/shoebox/alfredo-request/test/",
            sound_snip_len=sound_snips_len_ms,
            override_existing=True,
            limit_used_soundclips=32
        )

        torch_dataloader = torch.utils.data.DataLoader(
            dataset=self.test_dataloader, batch_size=4
        )
        test_model = models.filter_estimator.FilterEstimatorModel(
            input_channels=2, output_shape=(3, 1024)
        )

        test_trainer = trainer.Trainer(
            dataloader=torch_dataloader,
            loss_function=loss_functions.sound_loss,
            model=test_model,
            device=device,
            filter_length=1024,
            inner_loop_iterations=16,
            enable_debug_plotting=True,
        )

        test_trainer.run_epoch()


class TestLoss(unittest.TestCase):
    def test_sound_loss(self):

        gt_sound = torch.rand(16, 1, 100)
        ms_bz_sound = torch.rand(16, 3, 200)
        ms_dz_sound = torch.rand(16, 12, 200) * 0.1
        f_sound = torch.rand(16, 3, 150)

        data_for_loss_dict = {
            "gt_sound": gt_sound,
            "f_sound": f_sound,
            "bz_input": ms_bz_sound,
            "dz_input": ms_dz_sound,
        }

        loss_functions.sound_loss(data_for_loss_dict)


class TestModels(unittest.TestCase):
    def test_filter_estimator(self):
        test_model = models.filter_estimator.FilterEstimatorModel(
            input_channels=2, output_shape=(3, 1024)
        )

        torchinfo.summary(test_model)



if __name__ == "__main__":

    unittest.main()

    # test_dataloader = dataloader.DefaultDataset(sound_dataset_root='/home/ai/datasets/audio/LibriSpeech/train-clean-100/',
    #                                             rir_dataset_root='/home/ai/datasets/audio/test_rirs/dataset/shoebox/alfredo-request/test/', sound_snip_len=2000)
    # print(len(test_dataloader))
    # data_dict = test_dataloader[2300]
    #
    # print(data_dict)
    #
    # torch_dataloader = torch.utils.data.DataLoader(dataset=test_dataloader, batch_size=16)
    #
    # for data_dict in tqdm(torch_dataloader):
    #     sound = data_dict["sound"]
    #     bz_rirs = data_dict["bz_rirs"]
    #     dz_rirs = data_dict["dz_rirs"]
    #
    #     print(f'sound shape {sound.size()}')
    #     print(f'bz rirs {bz_rirs.size()}')
    #     print(f'dz rirs {dz_rirs.size()}')
    #
    #
    #
    #
