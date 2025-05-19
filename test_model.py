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
from evaluation.acoustic_contrast import acc_evaluation



def get_class_or_func(path):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path")
    args = parser.parse_args()

    # Load config
    with open(os.path.join(args.exp_path, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    training_config = config["training_run"]

    # === Get components from config ===
    model_class = get_class_or_func(training_config["model"])

    sound_dataset_path = training_config["sound_dataset_path"]
    rir_dataset_path = training_config["rir_dataset_path"]
    batch_size = training_config["batch_size"]
    filter_length = training_config["filter_length"]
    inner_loop_iterations = training_config["inner_loop_iterations"]
    save_path = training_config["savepath"]
    sound_snip_len = training_config["sound_snip_length"]

    ### load testing sound ###
    testing_sound, sr = soundfile.read(
        "./testing_data/relaxing-guitar-loop-v5-245859.wav"
    )
    testing_sound = testing_sound[:, 1]
    # print(sd.query_devices())
    # sd.play(testing_sound, sr, blocking=True)
    print(testing_sound)

    ### load rirs ###
    test_rirs_path = rir_dataset_path.replace("train", "test")

    dataset = dataloader.DefaultDataset(
        sound_dataset_root=sound_dataset_path,
        rir_dataset_root=rir_dataset_path,
        sound_snip_len=sound_snip_len,
        limit_used_soundclips=42,
        override_existing=True,  # Add this if needed
    )

    torch_dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False
    )

    # === Instantiate Model ===
    model = model_class(
        input_channels=2,  # Update as needed
        output_shape=(3, filter_length),  # Adjust based on number of sources/mics
    )

    model_state_dict_fn = os.listdir(os.path.join(save_path, "checkpoints"))[
        0
    ]  # make this user selectebel later
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
        evaluation_data = model_interacter.run_inner_feedback_testing(data_dict, 16)

        filters = evaluation_data["filters_time"]

        bl_filters = torch.zeros_like(filters)

        print(bl_filters.size())

        bl_filters[:,:,0] = 1

        filtered_sound = model_interacter.apply_filter(testing_sound_tensor, filters)

        bz_rirs = data_dict["bz_rirs"]
        dz_rirs = data_dict["dz_rirs"]

        bz_sound = model_interacter.auralizer(filtered_sound, bz_rirs)
        dz_sound = model_interacter.auralizer(filtered_sound, dz_rirs)

        sound_max_amp = torch.max(torch.abs(torch.cat((bz_sound, dz_sound), dim=1)))

        bz_sound = bz_sound / sound_max_amp
        dz_sound = dz_sound / sound_max_amp

        acc = acc_evaluation(filters, bz_rirs, dz_rirs)

        acc_bl = acc_evaluation(bl_filters, bz_rirs, dz_rirs)

        print(f"model acc {acc}, base line acc {acc_bl}")
