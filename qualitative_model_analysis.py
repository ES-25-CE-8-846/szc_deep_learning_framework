import argparse
import os
import yaml
import importlib
import soundfile
import sounddevice as sd
import models
from training import dataloader
from training import trainer #we use some helper functions from this
import torch
from torch._C import device
import torch.nn.functional as F
import numpy as np

def get_class_or_func(path):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


class ModelInteraction:
    def __init__(
        self,
        model,
        dataloader,
        device=None,
        filter_length=2048,
        inner_loop_iterations=16,
        enable_debug_plotting=False,
    ) -> None:
        self.model = model.eval()
        self.dataloader = dataloader
        self.filter_length = filter_length
        self.device = device
        self.inner_loop_iterations = inner_loop_iterations
        self.enable_debug_plotting = enable_debug_plotting

        if device is not None:
            self.model = self.model.to(device)

    def auralizer(self, sound, impulse_responses):
        """
        Function to auralize sound from multiple speakers.

        Args:
            sound (Tensor): shape (b. n, k) where n = number of speakers, k = sound length
            impulse_responses (Tensor): shape (b, n, m, s) where
                                        b = batch size
                                        n = number of speakers,
                                        m = number of microphones,
                                        s = length of impulse response

        Returns:
            Tensor: shape (b, m, k + s - 1), the auralized output at each microphone
        """
        B, N, K = sound.shape
        _, M, _, S = impulse_responses.shape
        output_len = K + S - 1

        impulse_responses = impulse_responses.type_as(sound)

        mic_output = torch.zeros((B, M, output_len), device=sound.device)

        for b in range(B):
            for n in range(N):
                speaker = sound[b, n].view(1, 1, -1)  # [1, 1, K]
                for m in range(M):
                    rir = impulse_responses[b, m, n].flip(0).view(1, 1, -1)  # [1, 1, S]
                    convolved = F.conv1d(speaker, rir, padding=0).squeeze()  # [L]

                    # Pad or trim to match expected output length
                    if convolved.shape[0] < output_len:
                        pad_len = output_len - convolved.shape[0]
                        convolved = F.pad(convolved, (0, pad_len))
                    elif convolved.shape[0] > output_len:
                        convolved = convolved[:output_len]

                    mic_output[b, m] += convolved

        return mic_output

    def apply_filter(self, sound, filters):
        """
        Function to apply filters to the sound.

        Args:
            sound (Tensor or ndarray): shape (b, k) the input sound signal
            filters (Tensor or ndarray): shape (b, n, m) where
                                         b = batch size
                                         n = number of filters,
                                         m = filter length

        Returns:
            Tensor: shape (n, k + m - 1), each row is the filtered sound
        """
        if not isinstance(sound, torch.Tensor):
            sound = torch.tensor(sound, dtype=torch.float32)
        if not isinstance(filters, torch.Tensor):
            filters = torch.tensor(filters, dtype=torch.float32)

        B, _, K = sound.shape
        _, N, M = filters.shape
        output_len = K + M - 1

        output = torch.zeros(B, N, output_len, device=filters.device)
        sound = sound.to(filters.device)

        for b in range(B):
            input_signal = sound[b].view(1, 1, -1)  # [1, 1, K]
            for n in range(N):
                filt = filters[b, n].flip(0).view(1, 1, -1)  # [1, 1, M]
                conv = F.conv1d(input_signal, filt, padding=0).squeeze()  # [output_len]

                # Pad if needed (for safety)
                if conv.shape[0] < output_len:
                    conv = F.pad(conv, (0, output_len - conv.shape[0]))
                output[b, n] = conv

        return output

    def format_to_model_input(self, output_sound, mic_inputs):
        """
        Function to ensure correct network input
        Args:
            output_sound(tensor): the output sound of each speaker
            mic_inputs(tensor): the input of each of the microphones available during inference

        Returns:
            stacked_tensor(tensor): the stacked tensor with shape (n, c, h, w)
                                    n: batch size
                                    c: channels (2)
                                    h: height
                                    w: width
        """

        # print(f"sp shape {output_sound.size()}")
        # print(f"mc shape {mic_inputs.size()}")

        # cropping
        out_len = output_sound.size()[2]
        mic_inputs = mic_inputs[:, :, 0:out_len]

        if device is not None:
            mic_inputs = mic_inputs.to(self.device)
            output_sound = output_sound.to(self.device)

        stacked_tensor = torch.stack([output_sound, mic_inputs], dim=1)
        if self.device is not None:
            stacked_tensor = stacked_tensor.to(self.device)

        assert stacked_tensor.size()[0] == output_sound.size()[0]
        assert stacked_tensor.size()[1] == 2

        return stacked_tensor.transpose(2, 3)

    def run_inner_feedback_testing(self, data_dict, n_iterations):
        """
        Function to run the inner feedback training loop

        Args:
            data_dict(dict[sounds, rirs]): dictionary containing the sampled sounds and the room impulse responses
            n_iterations (int): the amount of iterations to run the inner training loop

        """
        sound = data_dict["sound"]
        bz_rirs = data_dict["bz_rirs"]
        dz_rirs = data_dict["dz_rirs"]

        n_speakers = bz_rirs.size()[2]
        batch_size = bz_rirs.size()[0]
        old_filters = torch.ones((batch_size, n_speakers, self.filter_length))

        with torch.no_grad():
            for iteration in range(n_iterations):
                filtered_sound = self.apply_filter(sound, old_filters)
                bz_microphone_input = self.auralizer(sound, bz_rirs)

                nn_input = self.format_to_model_input(filtered_sound, bz_microphone_input)

                model_output = self.model.forward(nn_input)

                if self.model.output_filter_domain == "time":
                    filters_frq = torch.fft.rfft(model_output)
                    filters_time = model_output
                elif self.model.output_filter_domain == "frequency":
                    filters_frq = model_output
                    filters_time = torch.fft.irfft(model_output)

                # listening point is obtained by convolution between the ILZ FIR print(f"output filter shape {filters.size()}")

                filtered_sound = self.apply_filter(sound, filters_time)
                bz_microphone_input = self.auralizer(filtered_sound, bz_rirs)
                dz_microphone_input = self.auralizer(filtered_sound, dz_rirs)

                data_for_eval_dict = {
                    "gt_sound": sound,
                    "f_sound": filtered_sound,
                    "bz_input": bz_microphone_input,
                    "dz_input": dz_microphone_input,
                    "filters_time": filters_time,
                    "filters_frq":filters_frq,
                    "data_dict": data_dict,
                }

                old_filters = filters_time.detach()

            return data_for_eval_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path")
    args = parser.parse_args()

    # Load config
    with open(os.path.join(args.exp_path,"config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    training_config = config['training_run']

    # === Get components from config ===
    model_class = get_class_or_func(training_config['model'])

    sound_dataset_path = training_config['sound_dataset_path']
    rir_dataset_path = training_config['rir_dataset_path']
    batch_size = training_config['batch_size']
    filter_length = training_config['filter_length']
    inner_loop_iterations = training_config['inner_loop_iterations']
    save_path = training_config['savepath']
    sound_snip_len = training_config['sound_snip_length']

    ### load testing sound ###
    testing_sound, sr = soundfile.read("./testing_data/relaxing-guitar-loop-v5-245859.wav")
    testing_sound = testing_sound[:,1]
    sd.default.device = 0
    # print(sd.query_devices())
    # sd.play(testing_sound, sr, blocking=True)
    print(testing_sound)


    ### load rirs ###
    test_rirs_path = rir_dataset_path.replace('train','test')

    dataset = dataloader.DefaultDataset(
        sound_dataset_root=sound_dataset_path,
        rir_dataset_root=rir_dataset_path,
        sound_snip_len=sound_snip_len,
        limit_used_soundclips=42,
        override_existing=True  # Add this if needed
    )

    torch_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True
    )

    # === Instantiate Model ===
    model = model_class(
        input_channels=2,  # Update as needed
        output_shape=(3, filter_length)  # Adjust based on number of sources/mics
    )


    model_state_dict_fn = os.listdir(os.path.join(save_path, "checkpoints"))[0] # make this user selectebel later
    state_dict_path = os.path.join(save_path, "checkpoints", model_state_dict_fn)

    model.load_state_dict(torch.load(state_dict_path, weights_only=True))


    model_interacter = ModelInteraction(model = model,
                                        dataloader = torch_dataloader,
                                        filter_length = filter_length,
                                        inner_loop_iterations=inner_loop_iterations)

    testing_sound = testing_sound[np.newaxis, np.newaxis, :]
    testing_sound_tensor = torch.tensor(testing_sound).to(torch.float)


    for i, data_dict in enumerate(torch_dataloader):
        evaluation_data = model_interacter.run_inner_feedback_testing(data_dict, 16)

        filters = evaluation_data['filters_time']

        filtered_sound = model_interacter.apply_filter(testing_sound_tensor, filters)

        bz_rirs = data_dict['bz_rirs']
        dz_rirs = data_dict['dz_rirs']

        bz_sound = model_interacter.auralizer(filtered_sound, bz_rirs)
        dz_sound = model_interacter.auralizer(filtered_sound, dz_rirs)

        print(bz_sound.size())






