import argparse
import os
import yaml
import importlib
import soundfile
import sounddevice as sd
from training import dataloader
from training import trainer  # we use some helper functions from this
import torch
import numpy as np
from evaluation.model_interaction import ModelInteraction


def get_class_or_func(path):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def load_signal_distortion_test_sound():
    sound_file_dir = "./testing_data/signal_distortion_soundfiles/"
    full_testing_sound = []
    sound_files = sorted(os.listdir(sound_file_dir))
    print("loading signal distortion test sound")
    for sound_file in sound_files:
        testing_sound, sr = soundfile.read(os.path.join(sound_file_dir, sound_file))
        full_testing_sound.append(testing_sound)

    concatenated_sound = np.concatenate(full_testing_sound)
    return concatenated_sound


class UserCommandDispatcher:
    def __init__(self, sound_dict, model_interactor, savepath):
        """Function to handle user command"""
        self.should_continue = True
        self.sound = np.zeros(10)
        self.sound_dict = sound_dict
        self.current_filter = "model"
        self.model = model_interactor.model
        self.savepath = savepath
        self.posible_checkpoints = os.listdir(os.path.join(savepath, "checkpoints"))
        self.command_to_function_dict = {
            "p": self.play_sound,
            "save": self.save_sound,
            "sds": self.sound_device_selector,
            "fs": self.select_filter,
            "ss": self.sound_selector,
            "svs": self.save_sound,
            "nr": self.next_room,
            "cs": self.checkpoint_selector,
            "help": self.helper,
        }

    def helper(self):
        """Function to help"""
        for key in self.command_to_function_dict:
            print(key)
            print(self.command_to_function_dict[key].__doc__)

    def next_room(self):
        """Function to go to the next room"""
        self.should_continue = False

    def select_filter(self):
        """Function to select the filter"""
        filter_names = []

        print("select a filter")
        for i, key in enumerate(self.sound_dict.keys()):
            filter_names.append(key)
            print(f"{i}: {key}")

        selected_index = int(input("select: "))
        self.current_filter = filter_names[selected_index]

    def sound_selector(self):
        """Function to have the user select the sound to play"""
        i = 0
        n_bz = 0
        n_dz = 0
        self.bz_sounds = self.sound_dict[self.current_filter][0]
        self.dz_sounds = self.sound_dict[self.current_filter][1]

        for sound in self.bz_sounds[0, :, :]:
            print(f"bz_{i}")
            i += 1
            n_bz += 1

        for sound in self.dz_sounds[0, :, :]:
            print(f"dz_{i}")
            i += 1
            n_dz += 1

        selected_sound_index = int(input("sound: "))

        if selected_sound_index <= n_bz and selected_sound_index >= 0:
            selected_sound = self.bz_sounds[0, selected_sound_index, :]

        elif selected_sound_index > n_bz and selected_sound_index <= n_dz + n_bz:

            selected_sound = self.dz_sounds[0, selected_sound_index - (n_bz), :]

        else:
            print("sound dont exist")

        self.sound = selected_sound

    def sound_device_selector(self):
        """Function to select the sound device"""
        print("available devices")
        print(sd.query_devices())
        selected_device = int(input("select: "))
        sd.default.device = selected_device

    def play_sound(self):
        """Function to play sound"""
        print(self.sound)
        sd.play(self.sound)

    def checkpoint_selector(self):
        """Function to select checkpoint"""
        print("available checkpoints")
        for i, checkpoint in enumerate(self.posible_checkpoints):
            print(f"{checkpoint}: {i}")

        user_command = input("checkpoint: ")

        checkpoint_to_use = self.posible_checkpoints[int(user_command)]
        print(f"selected checkpoint: {checkpoint_to_use}")

        checkpoint_path = os.path.join(self.savepath, "checkpoints", checkpoint_to_use)

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(state_dict_path, weights_only=True))
        else:
            self.model.load_state_dict(
                torch.load(
                    state_dict_path, weights_only=True, map_location=torch.device("cpu")
                )
            )

    def save_sound(self):
        """Function to save sound"""

    def user_interactor(self):

        while self.should_continue:
            user_command = input("command: ")

            try:
                function = self.command_to_function_dict[user_command]
            except Exception as e:
                print(f"{e}")
                function = self.helper

            try:
                function()
            except Exception as e:
                print(f"{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path")
    args = parser.parse_args()

    # Load config
    with open(os.path.join(args.exp_path, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    training_config = config["training_run"]
    testing_config = config["testing_run"]

    # === Get components from config ===
    model_class = get_class_or_func(training_config["model"])

    sound_dataset_path = training_config["sound_dataset_path"]
    sound_dataset_path = "/home/a/datasets/audio/LibriSpeech"
    rir_dataset_path = testing_config["rir_dataset_path"]
    rir_dataset_path = "/home/a/datasets/audio/testing_rirs/dataset/shoebox/run2/test"
    batch_size = training_config["batch_size"]
    batch_size = 1
    filter_length = training_config["filter_length"]
    inner_loop_iterations = training_config["inner_loop_iterations"]
    save_path = training_config["savepath"]
    sound_snip_len = training_config["sound_snip_length"]

    ### load testing sound ###
    testing_sound, sr = soundfile.read(
        "./testing_data/longer_speech_samples/only_for_testing/wav/concatenated_test_audio_44100.wav"
    )
    testing_sound = testing_sound[:, 1]

    # testing_sound = load_signal_distortion_test_sound()

    sd.default.device = 0
    # print(sd.query_devices())
    # sd.play(testing_sound, sr, blocking=True)
    print(testing_sound)

    ### load rirs ###
    test_rirs_path = rir_dataset_path.replace("train", "test")

    dataset = dataloader.DefaultDataset(
        sound_dataset_root=sound_dataset_path,
        rir_dataset_root=rir_dataset_path,
        sound_snip_len=sound_snip_len,
        sound_snip_save_path="/tmp/sound_snips/test",
        limit_used_soundclips=100,
        override_existing=True,  # Add this if needed
        load_pre_computed_filters=True,
    )

    torch_dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False
    )

    # === Instantiate Model ===
    model = model_class(
        input_channels=2,  # Update as needed
        output_shape=(3, filter_length),  # Adjust based on number of sources/mics
    )

    model_state_dict_fn = sorted(os.listdir(os.path.join(save_path, "checkpoints")))[
        -1
    ]  # make this user selectebel later
    print(model_state_dict_fn)
    state_dict_path = os.path.join(save_path, "checkpoints", model_state_dict_fn)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    else:
        model.load_state_dict(
            torch.load(
                state_dict_path, weights_only=True, map_location=torch.device("cpu")
            )
        )

    model_interacter = ModelInteraction(
        model=model,
        dataloader=torch_dataloader,
        filter_length=filter_length,
        inner_loop_iterations=inner_loop_iterations,
    )

    testing_sound = testing_sound[np.newaxis, np.newaxis, :]
    testing_sound_tensor = torch.tensor(testing_sound).to(torch.float)

    for i, data_dict in enumerate(torch_dataloader):
        evaluation_data = model_interacter.run_inner_feedback_testing(
            data_dict, inner_loop_iterations
        )

        filters = evaluation_data["filters_time"]

        pre_comp_filters = dict(evaluation_data["data_dict"]["precomp_filters"])

        pre_comp_filters["model"] = filters

        sound_dict = {}

        for filter_name, f in pre_comp_filters.items():
            filtered_sound = model_interacter.apply_filter(testing_sound_tensor, f)

            bz_rirs = data_dict["bz_rirs"]
            dz_rirs = data_dict["dz_rirs"]

            bz_sound = model_interacter.auralizer(filtered_sound, bz_rirs)
            dz_sound = model_interacter.auralizer(filtered_sound, dz_rirs)

            sound_max_amp = torch.max(torch.abs(torch.cat((bz_sound, dz_sound), dim=1)))

            noise_floor = np.random.rand(bz_sound.shape[-1]) * 0.001

            bz_sound = bz_sound / sound_max_amp + noise_floor
            dz_sound = dz_sound / sound_max_amp + noise_floor

            sound_dict[filter_name] = [bz_sound, dz_sound]

        print(f"room {i}")
        user_command_dispatcher = UserCommandDispatcher(
            sound_dict, model_interacter, savepath=save_path
        )
        user_command_dispatcher.user_interactor()
        print(bz_sound.size())
