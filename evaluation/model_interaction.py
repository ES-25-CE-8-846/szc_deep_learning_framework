import torch
import torch
from torch._C import device
import torch.nn.functional as F

import scipy.signal
import numpy as np

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

        # for b in range(B):
        #     for n in range(N):
        #         speaker = sound[b, n].view(1, 1, -1)  # [1, 1, K]
        #         for m in range(M):
        #             rir = impulse_responses[b, m, n].flip(0).view(1, 1, -1)  # [1, 1, S]
        #             convolved = F.conv1d(speaker, rir, padding=0).squeeze()  # [L]
        #
        #             # Pad or trim to match expected output length
        #             if convolved.shape[0] < output_len:
        #                 pad_len = output_len - convolved.shape[0]
        #                 convolved = F.pad(convolved, (0, pad_len))
        #             elif convolved.shape[0] > output_len:
        #                 convolved = convolved[:output_len]
        #
        #             mic_output[b, m] += convolved



        sound = sound.detach().cpu()
        sound = sound[:,None,:,:]
        impulse_responses = impulse_responses.detach().cpu()

        mic_output = scipy.signal.fftconvolve(sound, impulse_responses, axes=3)

        mic_output = np.sum(mic_output, axis=2)

        return torch.from_numpy(mic_output)


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

        # for b in range(B):
        #     input_signal = sound[b].view(1, 1, -1)  # [1, 1, K]
        #     for n in range(N):
        #         filt = filters[b, n].flip(0).view(1, 1, -1)  # [1, 1, M]
        #         conv = F.conv1d(input_signal, filt, padding=0).squeeze()  # [output_len]
        #
        #         # Pad if needed (for safety)
        #         if conv.shape[0] < output_len:
        #             conv = F.pad(conv, (0, output_len - conv.shape[0]))
        #         output[b, n] = conv
        #

        sound = sound.detach().cpu()
        sound = sound[:,:,:]
        filters = filters.detach().cpu()

        output = scipy.signal.fftconvolve(sound, filters, axes=2)

        return torch.from_numpy(output)


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
