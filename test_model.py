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
from evaluation.distortion import normalized_signal_distortion
from evaluation.array_effort import array_effort
from tqdm import tqdm


def get_class_or_func(path):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def load_signal_distortion_test_sound():
    sound_file_dir = "./testing_data/signal_distortion_soundfiles/"
    full_testing_sound = []
    sound_files = sorted(os.listdir(sound_file_dir))
    print("loading signal distortion test sound")
    for sound_file in tqdm(sound_files):
        testing_sound, sr = soundfile.read(os.path.join(sound_file_dir, sound_file))
        full_testing_sound.append(testing_sound)

    concatenated_sound = np.concatenate(full_testing_sound)
    return concatenated_sound, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path")
    args = parser.parse_args()

    # Load config
    with open(os.path.join(args.exp_path, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    training_config = config["training_run"]
    testing_config = config["testing_run"]
    sd_test_sound = np.zeros(10)
    testing_metrics = testing_config["metrics"]
    baseline_filters = testing_config["baseline_filters"]
    print(f"testing metrics {testing_metrics}")

    metrics_results_dict = {}

    for metric in testing_metrics:
        if metric == "sd":
            sd_test_sound = load_signal_distortion_test_sound()

        metrics_results_dict[metric] = []

    # === Get components from config ===
    model_class = get_class_or_func(training_config["model"])

    sound_dataset_path = testing_config["sound_dataset_path"]
    rir_dataset_path = testing_config["rir_dataset_path"]
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
        rir_dataset_root=test_rirs_path,
        sound_snip_len=sound_snip_len,
        limit_used_soundclips=42,
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

    # create filter result dict
    filter_result_dict = {}
    baseline_filters.append("model")
    for filter_name in baseline_filters:
        filter_result_dict[filter_name] = {}
        for metric in testing_metrics:
            filter_result_dict[filter_name][metric] = []

    for i, data_dict in tqdm(enumerate(torch_dataloader)):
        evaluation_data = model_interacter.run_inner_feedback_testing(data_dict, 16)

        filters = evaluation_data["filters_time"]
        precomp_filters = evaluation_data["data_dict"]["precomp_filters"]

        dirac_filters = torch.zeros_like(filters)
        dirac_filters[:, :, 0] = 1

        filters_to_test = {}

        filters_to_test["model"] = filters

        for filter_name in baseline_filters:
            if filter_name == "dirac":
                filters_to_test[filter_name] = dirac_filters
            elif filter_name == "vast":
                filters_to_test[filter_name] = precomp_filters["q_vast"]
            elif filter_name == "acc":
                filters_to_test[filter_name] = precomp_filters["q_acc"]
            elif filter_name == "pm":
                filters_to_test[filter_name] = precomp_filters["q_pm"]

        for filter_name, f in zip(filters_to_test.keys(), filters_to_test.values()):

            filtered_sound = model_interacter.apply_filter(testing_sound_tensor, f)

            bz_rirs = data_dict["bz_rirs"]
            dz_rirs = data_dict["dz_rirs"]

            bz_sound = model_interacter.auralizer(filtered_sound, bz_rirs)
            dz_sound = model_interacter.auralizer(filtered_sound, dz_rirs)

            sound_max_amp = torch.max(torch.abs(torch.cat((bz_sound, dz_sound), dim=1)))

            bz_sound = bz_sound / sound_max_amp
            dz_sound = dz_sound / sound_max_amp
            filter_results = {}
            # print(filter_name)
            for metric in testing_metrics:
                if metric == "sd":
                    result = normalized_signal_distortion(
                        testing_sound, f, bz_rirs, bz_sound
                    )
                    # print(f"sd {result}")
                elif metric == "acc":
                    result = acc_evaluation(f, bz_rirs, dz_rirs)
                    # print(f"acc {result}")
                elif metric == "ae":
                    result = torch.mean(array_effort(f, bz_rirs))

                filter_result_dict[filter_name][metric].append(result)

            # for metric, result in zip(metrics_results_dict.keys(), metrics_results_dict.values()):
            #     print(f"{metric}: {10 * np.log10(result[-1])}")
            #
