import argparse
import os
from scipy._lib.array_api_compat import device
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
from evaluation.intelligibility import evaluate_mos, evaluate_stoi, evaluate_pesq
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal



def get_class_or_func(path):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

def load_test_sound():
    sound_file_dir = "./testing_data/longer_speech_samples/only_for_testing/wav/"
    full_testing_sound = []
    sound_files = sorted(os.listdir(sound_file_dir))
    print("loading signal distortion test sound")
    for sound_file in sound_files:
        testing_sound, sr = soundfile.read(os.path.join(sound_file_dir, sound_file))
        full_testing_sound.append(testing_sound)

    concatenated_sound = np.concatenate(full_testing_sound)
    stacked_sound = np.stack([concatenated_sound, concatenated_sound, concatenated_sound]) # stack to speaker dimensions
    stacked_sound = stacked_sound[np.newaxis, ...] #add speaker dimension

    return stacked_sound



def load_signal_distortion_test_sound():
    sound_file_dir = "./testing_data/signal_distortion_soundfiles/"
    full_testing_sound = []
    sound_files = sorted(os.listdir(sound_file_dir))
    print("loading signal distortion test sound")
    for sound_file in tqdm(sound_files):
        testing_sound, sr = soundfile.read(os.path.join(sound_file_dir, sound_file))
        full_testing_sound.append(testing_sound)

    concatenated_sound = np.concatenate(full_testing_sound)
    return concatenated_sound


def plot_filters(ax, filters, name, sample_rate=48000):
    """
    Plots the magnitude of the filter frequency responses.

    Args:
        ax: Matplotlib Axes object
        filters: Tensor of shape (B, S, K) — filters per speaker
        name: Title for the subplot
        sample_rate: Sample rate of the system (for frequency axis scaling)
    """
    filter_len = filters.shape[-1]
    freqs = torch.fft.rfftfreq(filter_len, d=1/sample_rate)

    for fi in range(min(3, filters.shape[1])):  # Plot up to 3 speakers
        mag = torch.abs(torch.fft.rfft(filters[0, fi, :]))
        ax.plot(freqs, mag)

    ax.set_title(name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

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
    testing_rirs = testing_config["rir_dataset_path"]
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



    ### load rirs ###
    # test_rirs_path = rir_dataset_path.replace("train", "test")

    dataset = dataloader.DefaultDataset(
        sound_dataset_root=sound_dataset_path,
        rir_dataset_root=testing_rirs,
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

    model_state_dict_fn = sorted(os.listdir(os.path.join(save_path, "checkpoints")))[
        -1
    ]  # make this user selectebel later
    state_dict_path = os.path.join(save_path, "checkpoints", model_state_dict_fn)

    print(model_state_dict_fn)
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

    testing_sound = load_test_sound()[:,:,:,0]

    # create filter result dict
    filter_result_dict = {}
    baseline_filters.append("model")
    for filter_name in baseline_filters:
        filter_result_dict[filter_name] = {}
        for metric in testing_metrics:
            filter_result_dict[filter_name][metric] = []
    n_tests = 0


    # load avg filters

    avg_filters = dict(np.load("/home/ai/Downloads/combined_filters.npz"))

    fig, ax = plt.subplots(6, 1)
    for i, data_dict in tqdm(enumerate(torch_dataloader)):

        n_tests += 1
        evaluation_data = model_interacter.run_inner_feedback_testing(
            data_dict, inner_loop_iterations
        )

        filters = evaluation_data["filters_time"]
        precomp_filters = evaluation_data["data_dict"]["precomp_filters"]

        dirac_filters = torch.zeros_like(filters)
        dirac_filters[:, :, int(dirac_filters.shape[-1]/2)] = 1

        filters_to_test = {}

        filters_to_test["model"] = filters

        # plot_filters(ax[0], filters, "model")


        for n, filter_name in enumerate(baseline_filters):
            if filter_name == "dirac":
                filters_to_test[filter_name] = dirac_filters
            elif filter_name == "vast":
                filters_to_test[filter_name] = precomp_filters["q_vast"]
            elif filter_name == "acc":
                filters_to_test[filter_name] = precomp_filters["q_acc"]
            elif filter_name == "pm":
                filters_to_test[filter_name] = precomp_filters["q_pm"]
            elif filter_name == "avg_vast":
                filters_to_test[filter_name] = avg_filters["q_vast"]
            elif filter_name == "avg_acc":
                filters_to_test[filter_name] = avg_filters["q_acc"]
            elif filter_name == "avg_pm":
                filters_to_test[filter_name] = avg_filters["q_pm"]


            # plot_filters(ax[n+1], filters_to_test[filter_name], filter_name)

        for filter_name, f in zip(filters_to_test.keys(), filters_to_test.values()):

            bz_rirs = data_dict["bz_rirs"]
            dz_rirs = data_dict["dz_rirs"]

            if len(f.shape) != 3:
                f = f[np.newaxis,...]

            print(f"testing_sound shape {testing_sound.shape}")


            filtered_test_sound = scipy.signal.oaconvolve(testing_sound, f, axes=2)

            filtered_test_sound = filtered_test_sound[:, np.newaxis, ...] # adding microphone dim

            delayed_dry_sound = scipy.signal.oaconvolve(testing_sound, filters_to_test["dirac"])

            auralized_test_sound_bz = np.sum(scipy.signal.oaconvolve(filtered_test_sound, bz_rirs.detach().cpu(), axes=3), axis=2) # [B, M, K]
            auralized_test_sound_dz = np.sum(scipy.signal.oaconvolve(filtered_test_sound, dz_rirs.detach().cpu(), axes=3), axis=2) # [B, M, K]


            print(f"auralized_test_sound_bz shape {auralized_test_sound_bz.shape}")

            number_of_samples = round(testing_sound.shape[-1] * float(16000) / 44100)


            ear_sound:np.ndarray = scipy.signal.resample(auralized_test_sound_bz[0,0,:testing_sound.shape[-1]], number_of_samples, axis=-1 ) #[K]

            dz_sound:np.ndarray = scipy.signal.resample(auralized_test_sound_dz[-1,-1,:testing_sound.shape[-1]], number_of_samples, axis=-1 ) #[K]
            dry_sound:np.ndarray = scipy.signal.resample(delayed_dry_sound[0,0,:testing_sound.shape[-1]], number_of_samples, axis=-1) #[K]

            # sound_max_amp = np.max(np.abs(np.concatenate((ear_sound, dz_sound))))
            #
            # noise_floor = np.random.rand(ear_sound.shape[-1]) * 0.001
            #
            # ear_sound = ear_sound / sound_max_amp + noise_floor
            # dz_sound = dz_sound / sound_max_amp + noise_floor
            # filter_results = {}


            print(f"--{filter_name}--")
            for metric in testing_metrics:
                if metric == "sd":
                    result = normalized_signal_distortion(
                        dry_sound, f, bz_rirs
                    )
                    print(f"sd {result}")
                elif metric == "ac":
                    result = acc_evaluation(f, bz_rirs, dz_rirs)
                    print(f"ac {result}")
                elif metric == "ae":
                    result = torch.mean(10 * torch.log10((array_effort(torch.fft.rfft(f), bz_rirs))))
                    print(f"ae {result}")
                elif metric == "bz_mos":
                    result = evaluate_mos(ear_sound[np.newaxis,...], dry_sound[np.newaxis,...], device="cuda:1")
                    print(f"bz_mos {result}")
                elif metric == "dz_mos":
                    result = evaluate_mos(dz_sound[np.newaxis,...], dry_sound[np.newaxis,...], device="cuda:1")
                    print(f"dz_mos {result}")
                elif metric == "bz_stoi":
                    result = evaluate_stoi(ear_sound, dry_sound)
                    print(f"bz_stoi {result}")
                elif metric == "dz_stoi":
                    result = evaluate_stoi(dz_sound, dry_sound)
                    print(f"dz_stoi {result}")
                elif metric == "bz_pesq":
                    result = evaluate_pesq(ear_sound, dry_sound)
                    print(f"bz_pesq {result}")
                elif metric == "dz_pesq":
                    result = evaluate_pesq(dz_sound, dry_sound)
                    print(f"dz_pesq {result}")
                else:
                    print(f"{metric} not found")
                    break

                filter_result_dict[filter_name][metric].append(result)
        if i > 100:
            break

    for filter_name in filters_to_test.keys():

        for metric in testing_metrics:

            print(f"{filter_name}, {metric}")
            print(filter_result_dict[filter_name][metric])

            filter_result_dict[filter_name][metric] = np.mean(

                filter_result_dict[filter_name][metric]
            )

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(filter_result_dict, orient="index")

    # Save to CSV
    df.to_csv(os.path.join(save_path, "filter_results.csv"))

    print(df)
