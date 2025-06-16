from time import time
from warnings import filters
from numpy.core.fromnumeric import mean
import torch
from torch._C import device
from torch.utils.checkpoint import checkpoint
import torchaudio
import wandb
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchinfo
import os
import glob

import scipy.signal

from evaluation.acoustic_contrast import acc_evaluation
import evaluation.intelligibility

import scipy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        loss_function,
        rank,
        world_size,
        loss_weights=None,
        learning_rate=0.001,
        validation_dataloader=None,
        optimizer=torch.optim.AdamW,
        filter_length=2048,
        inner_loop_iterations=16,
        checkpointing_mode="none",
        save_path="",
        log_to_wandb=False,
        enable_debug_plotting=False,
        validation_metrics=['ac'],
        validation_test_sound = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.loss_function = loss_function

        self.loss_weights = loss_weights
        self.filter_length = filter_length
        self.device = rank
        self.rank = rank
        self.inner_loop_iterations = inner_loop_iterations
        self.checkpointing_mode = checkpointing_mode
        self.save_path = save_path
        self.enable_debug_plotting = enable_debug_plotting and rank == 0
        self.log_to_wandb = log_to_wandb and rank == 0

        self.validation_metrics = validation_metrics
        self.validation_test_sound = validation_test_sound

        torch.cuda.set_device(rank)
        self.model = self.model.to(rank)
        self.ddp_model = DDP(self.model, device_ids=[device])

        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

        self.current_ep = 1

        if self.checkpointing_mode != "none":
            self.checkpoint_path = os.path.join(self.save_path, "checkpoints")
            try:
                os.makedirs(self.checkpoint_path)
            except Exception as e:
                os.makedirs(self.checkpoint_path, exist_ok=True)
                print(e)

    def auralizer(self, sound, impulse_responses):
        """
        Function to auralize sound from multiple speakers.

        Args:
            sound (Tensor): shape (b. n, k) where n = number of speakers, k = sound length
            impulse_responses (Tensor): shape (b, n, m, s) where
                                        b = batch size
                                        m = number of microphones,
                                        n = number of speakers,
                                        s = length of impulse response

        Returns:
            Tensor: shape (b, m, k + s - 1), the auralized output at each microphone
        """
        B, N, K = sound.shape
        _, M, _, S = impulse_responses.shape
        output_len = K + S - 1

        impulse_responses = impulse_responses.type_as(sound)

        mic_output = torch.zeros((B, M, output_len), device=sound.device)

        sound = sound.detach().cpu()
        sound = sound[:, None, :, :]
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

        sound = sound.detach().cpu()
        sound = sound[:, :, :]
        filters = filters.detach().cpu()

        output = scipy.signal.fftconvolve(sound, filters, axes=2)

        return torch.from_numpy(output)

    def format_to_model_input(self, output_sound, mic_inputs):
        """
        Function to ensure correct network input shape and range
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
        mic_inputs = mic_inputs[:, 1:4, 0:out_len]

        if device is not None:
            mic_inputs = mic_inputs.to(self.device)
            output_sound = output_sound.to(self.device)


        stacked_tensor = torch.stack([output_sound, mic_inputs], dim=1)
        if self.device is not None:
            stacked_tensor = stacked_tensor.to(self.device)

        assert stacked_tensor.size()[0] == output_sound.size()[0]
        assert stacked_tensor.size()[1] == 2

        #normalize input to be between 1 and -1
        max_amp = torch.max(torch.abs(stacked_tensor))
        stacked_tensor = stacked_tensor / max_amp

        return stacked_tensor.transpose(2, 3)



    def run_inner_feedback_training(self, data_dict, n_iterations):
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
        old_filters = torch.zeros((batch_size, n_speakers, self.filter_length))
        old_filters[..., 0] = 1
        for iteration in range(n_iterations):
            filtered_sound = self.apply_filter(sound, old_filters)
            bz_microphone_input = self.auralizer(sound, bz_rirs)

            nn_input = self.format_to_model_input(filtered_sound, bz_microphone_input)

            # print(f"nn input shape {nn_input.shape}")

            model_output = self.ddp_model.forward(nn_input)

            if self.model.output_filter_domain == "time":
                filters_frq = torch.fft.rfft(model_output)
                filters_time = model_output
            elif self.model.output_filter_domain == "frequency":
                filters_frq = model_output
                filters_time = torch.fft.irfft(model_output)

            filtered_sound = self.apply_filter(sound, filters_time)
            bz_microphone_input = self.auralizer(filtered_sound, bz_rirs)
            dz_microphone_input = self.auralizer(filtered_sound, dz_rirs)

            data_for_loss_dict = {
                "gt_sound": sound,
                "f_sound": filtered_sound,
                "bz_input": bz_microphone_input,
                "dz_input": dz_microphone_input,
                "filters_time": filters_time,
                "filters_frq": filters_frq,
                "data_dict": data_dict,
            }

            loss_dict = self.loss_function(
                data_for_loss_dict, weights=self.loss_weights, device=self.device
            )
            loss = loss_dict["loss"]

            if self.enable_debug_plotting:
                self.debug_plotting(data_dict, data_for_loss_dict, loss_dict)

            loss.backward()
            abs_mean_gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Get parent module using name
                    module_name = name.rsplit(".", 1)[
                        0
                    ]  # Handles 'layer.weight', 'block.bn1.bias', etc.
                    module = dict(self.model.named_modules()).get(module_name, None)

                    # Skip if it's a BatchNorm layer
                    if isinstance(
                        module,
                        (
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d,
                        ),
                    ):
                        pass
                    else:
                        abs_mean_gradients[name] = torch.mean(torch.abs(param.grad)).item()

            gradient_dict = self.compute_gradient_normed_dict(abs_mean_gradients)

            self.optimizer.step()
            self.optimizer.zero_grad()

            old_filters = filters_time.detach()

            # compute acoustic contrast for logging
            acc = acc_evaluation(
                filters_time.detach().cpu(), bz_rirs.cpu(), dz_rirs.cpu()
            )
            loss_dict["acc"] = acc
            if self.rank == 0:
                print(
                    f"loss: {loss.item()}, ac {acc}, inner loop iteration: {iteration}"
                )
                self.print_gradient_dict(gradient_dict)
            if self.log_to_wandb:
                loss_dict["gradients"] = gradient_dict
                self.wandb_logger(loss_dict, "training")

        return loss_dict

        # take optimizer step

    def run_inner_feedback_validation(self, data_dict, n_iterations, evaluations_dict:dict):
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
        old_filters = torch.zeros((batch_size, n_speakers, self.filter_length))
        old_filters[..., 0] = 1
        with torch.no_grad():
            for iteration in range(n_iterations):
                filtered_sound = self.apply_filter(sound, old_filters)
                bz_microphone_input = self.auralizer(sound, bz_rirs)

                nn_input = self.format_to_model_input(
                    filtered_sound, bz_microphone_input
                )

                model_output = self.ddp_model.forward(nn_input)

                if self.model.output_filter_domain == "time":
                    filters_frq = torch.fft.rfft(model_output)
                    filters_time = model_output
                elif self.model.output_filter_domain == "frequency":
                    filters_frq = model_output
                    filters_time = torch.fft.irfft(model_output)

                filtered_sound = self.apply_filter(sound, filters_time)
                bz_microphone_input = self.auralizer(filtered_sound, bz_rirs)
                dz_microphone_input = self.auralizer(filtered_sound, dz_rirs)

                data_for_loss_dict = {
                    "gt_sound": sound,
                    "f_sound": filtered_sound,
                    "bz_input": bz_microphone_input,
                    "dz_input": dz_microphone_input,
                    "filters_time": filters_time,
                    "filters_frq": filters_frq,
                    "data_dict": data_dict,
                }

                loss_dict = self.loss_function(
                    data_for_loss_dict, weights=self.loss_weights, device=self.device
                )
                loss = loss_dict["loss"]

                if self.enable_debug_plotting:
                    self.debug_plotting(data_dict, data_for_loss_dict, loss_dict)

                if self.rank == 0:
                    print(f"loss: {loss.item()}, inner loop iteration: {iteration}")

                old_filters = filters_time.detach()

                # compute acoustic contrast for logging




                if len(evaluations_dict["ac"]) < 30  :
                    print(f"validation test sound shape {self.validation_test_sound.shape}")

                    filtered_test_sound = scipy.signal.oaconvolve(self.validation_test_sound, old_filters.detach().cpu(), axes=2)

                    filtered_test_sound = filtered_test_sound[:, np.newaxis, ...] # adding microphone dim

                    auralized_test_sound_bz = np.sum(scipy.signal.oaconvolve(filtered_test_sound, bz_rirs.detach().cpu(), axes=3), axis=2) # [B, M, K]

                    print(f"auralized_test_sound_bz shape {auralized_test_sound_bz.shape}")

                    number_of_samples = round(self.validation_test_sound.shape[-1] * float(16000) / 44100)


                    ear_sound = scipy.signal.resample(auralized_test_sound_bz[0,0,:self.validation_test_sound.shape[-1]], number_of_samples, axis=-1 ) #[K]
                    dry_sound = scipy.signal.resample(self.validation_test_sound[0,0,:], number_of_samples, axis=-1) #[K]

                    for metric in self.validation_metrics:
                        if metric == 'ac':
                            result = acc_evaluation(
                                filters_time.detach().cpu(), bz_rirs.cpu(), dz_rirs.cpu()
                            )
                            loss_dict["acc"] = result
                        elif metric == 'stoi':
                            result = evaluation.intelligibility.evaluate_stoi(signal=ear_sound, dry_signal=dry_sound, sample_rate=16000)

                        elif metric == 'pesq':
                            result = evaluation.intelligibility.evaluate_pesq(signal=ear_sound, dry_signal=dry_sound, sample_rate=16000)

                        elif metric == 'mos':
                            result = evaluation.intelligibility.evaluate_mos(signal=ear_sound[np.newaxis, ...], dry_signal=dry_sound[np.newaxis, ...], device=self.ddp_model.device)
                        evaluations_dict[metric].append(result)

                    print(evaluations_dict)

            loss_dict["evaluations"] = evaluations_dict
        return loss_dict

    def save_checkpoint(self, current_ep, last_loss):
        """Function to save the model weights based on the checkpointing_mode and current model weights"""

        if self.checkpointing_mode != "none":
            if self.checkpointing_mode == "all":
                fn = f"chkp_ep{str(current_ep).rjust(3,'0')}_loss{str(last_loss).replace('.','_')}.pth"
                print(f"saving checkpoint {fn}")
                torch.save(
                    self.model.state_dict(), os.path.join(self.checkpoint_path, fn)
                )

            elif self.checkpointing_mode == "last":

                fn = f"chkp_loss{str(last_loss).replace('.','_')}.pth"
                # ensure all files are removed so we only store the last model
                old_files = glob.glob(f"{self.checkpoint_path}/*")
                for file in old_files:
                    os.remove(file)

                print(f"saving checkpoint {fn}")
                torch.save(
                    self.model.state_dict(), os.path.join(self.checkpoint_path, fn)
                )

    def wandb_logger(self, loss_dict, namespace):
        if namespace == "validation":
            wandb.log(
                data={
                    f"{namespace}_loss": loss_dict["loss"].item(),
                    f"{namespace}_ac": loss_dict["evaluations"]['ac'],
                    f"{namespace}_evals" : loss_dict["evaluations"],
                }
            )
        elif namespace == "training":
            wandb.log(
                data={
                    f"{namespace}_loss": loss_dict["loss"].item(),
                    f"{namespace}_ac": loss_dict["acc"],
                    f"{namespace}_grd": loss_dict["gradients"],
                }
            )

    def compute_normalized_gradient_dict(self, gradient_dict: dict):
        max_gradient = max(gradient_dict.values())

        normalized_gradient_dict = {}

        for key, value in zip(gradient_dict.keys(), gradient_dict.values()):
            normalized_gradient_dict[key] = value / max_gradient

        return normalized_gradient_dict

    def compute_gradient_normed_dict(self, gradient_dict: dict):
        collector = np.zeros(4)
        counter = np.zeros(4)
        for key, value in zip(gradient_dict.keys(), gradient_dict.values()):
            if "encoder" in key and "weight" in key:
                collector[0] += value
                counter[0] += 1
            elif "encoder" in key and "bias" in key:
                collector[1] += value
                counter[1] += 1
            elif "mlp" in key and "weight" in key:
                collector[2] += value
                counter[2] += 1
            elif "mlp" in key and "bias" in key:
                collector[3] += value
                counter[3] += 1

        # print(f"collector sum: {np.sum(collector)}")
        # print(f"collector: {collector}")

        norm_valus = collector/counter

        norm_dict = {"encoder_weights":norm_valus[0],
                        "encoder_biases":norm_valus[1],
                        "mlp_weights":norm_valus[2],
                        "mlp_biases":norm_valus[3],
                        }

        return norm_dict




    def print_gradient_dict(self, gradient_dict):
        for key, value in zip(gradient_dict.keys(), gradient_dict.values()):
            print(f"{key}: {value}")

    def debug_plotting(self, data_dict, data_for_loss_dict, loss_dict):
        """
        Function to plot the ffts and the filters of the model to verify that it is training
        Args:
            data_dict
            data_for_loss_dict
            loss_dict
        """
        dry_sound = data_for_loss_dict["gt_sound"]

        # print(f"dry sound shape {dry_sound.size()}")

        dry_sound_len = dry_sound.size()[2]
        hann_window = torch.windows.hann(dry_sound_len)

        # cropping is needed to have the same dimensions in the fft output
        bz_input_sound = data_for_loss_dict["bz_input"][:, :, 0:dry_sound_len]
        dz_input_sound = data_for_loss_dict["dz_input"][:, :, 0:dry_sound_len]

        # compute the desired frequency responses
        des_bz_h = torch.fft.rfft(dry_sound * hann_window)
        des_dz_h = torch.fft.rfft(torch.zeros_like(dry_sound))

        # compute the measured frequency responses
        bz_h = torch.fft.rfft(bz_input_sound * hann_window)
        dz_h = torch.fft.rfft(dz_input_sound * hann_window)

        fft_bz = bz_h
        fft_dz = dz_h
        fft_bz_des = des_bz_h

        filters = data_for_loss_dict["filters_time"].cpu().detach().numpy()

        filter_axis = np.arange(filters.shape[2])

        frequency_axis = np.arange(fft_bz_des.shape[2])
        print(f"fft desired bz shape {fft_bz_des.shape}")
        batch_size = fft_bz_des.shape[0]
        fig, axs = plt.subplots(3, 4)
        for i in range(3):
            axs[i, 0].plot(frequency_axis, fft_bz[i, 1, :])
            axs[i, 1].plot(frequency_axis, fft_dz[i, 1, :])
            axs[i, 2].plot(frequency_axis, fft_bz_des[i, 0, :])
            for f in range(3):
                axs[i, 3].plot(filter_axis, filters[i, f, :])
        torchinfo.summary(self.model)
        plt.show()

    def run_epoch(self, epoch):
        """Function to initialize running an epoch"""
        total_batches = len(self.dataloader)

        self.dataloader.sampler.set_epoch(epoch)

        # training step
        self.ddp_model.train()
        for i, data_dict in enumerate(self.dataloader):
            time_begin = time()
            # Ensure all data is passed to the correct device
            if self.device is not None:
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key].to(self.device)

            loss_dict = self.run_inner_feedback_training(
                data_dict, self.inner_loop_iterations
            )

            if self.rank == 0:
                percent_done = (i + 1) / total_batches * 100
                print(f"epoch {self.current_ep}: {percent_done:.2f}% complete")
                print(f"time taken {time()-time_begin}")

        # validation step
        if self.rank == 0:
            print("initiating validation")

        total_val_batches = len(self.validation_dataloader)

        evaluations_dict = {}

        for metric in self.validation_metrics:
            evaluations_dict[metric] = []

        if self.validation_dataloader is not None:
            self.ddp_model.eval()
            summed_loss_dict = {"loss": 0.0, "acc": 0.0}
            losses = []
            n_losses = 0
            for i, data_dict in enumerate(self.validation_dataloader):
                # Ensure all data is passed to the correct device
                if self.device is not None:
                    for key in data_dict.keys():
                        data_dict[key] = data_dict[key].to(self.device)

                with torch.no_grad():
                    loss_dict = self.run_inner_feedback_validation(
                        data_dict, self.inner_loop_iterations, evaluations_dict
                    )
                    losses.append(loss_dict['loss'].detach().item())

                if self.rank == 0:
                    percent_done = (i + 1) / total_val_batches * 100
                    print(f"epoch {self.current_ep}: {percent_done:.2f}% complete")

                # this is more of a temporary logging, the evaluation metrics sohuld be logged as well
            loss_dict['loss'] = np.mean(losses)

            for key in loss_dict["evaluations"].keys():
                loss_dict["evaluations"][key] = np.mean(loss_dict["evaluations"][key])


            if self.rank == 0:
                print(f"mean loss: {loss_dict}")

            if self.log_to_wandb:
                self.wandb_logger(loss_dict, "validation")

        if self.rank == 0:
            self.save_checkpoint(
                current_ep=self.current_ep, last_loss=loss_dict["loss"].item()
            )
            self.current_ep += 1
