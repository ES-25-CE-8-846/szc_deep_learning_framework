from warnings import filters
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
        optimizer=torch.optim.AdamW,
        filter_length=2048,
        inner_loop_iterations=16,
        checkpointing_mode = "none",
        save_path = "",
        log_to_wandb = False,
        enable_debug_plotting=False,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_function = loss_function
        self.filter_length = filter_length
        self.device = rank
        self.rank = rank
        self.inner_loop_iterations = inner_loop_iterations
        self.checkpointing_mode = checkpointing_mode
        self.save_path = save_path
        self.enable_debug_plotting = enable_debug_plotting and rank == 0
        self.log_to_wandb = log_to_wandb and rank == 0

        torch.cuda.set_device(rank)
        self.model = self.model.to(rank)
        self.ddp_model = DDP(self.model, device_ids=[device])

        self.optimizer = optimizer(self.model.parameters(), lr=0.001)

        self.current_ep = 1

        if self.checkpointing_mode != 'none':
            self.checkpoint_path = os.path.join(self.save_path, 'checkpoints')
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
        old_filters = torch.ones((batch_size, n_speakers, self.filter_length))

        for iteration in range(n_iterations):
            filtered_sound = self.apply_filter(sound, old_filters)
            bz_microphone_input = self.auralizer(sound, bz_rirs)

            nn_input = self.format_to_model_input(filtered_sound, bz_microphone_input)

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
                "filters_frq":filters_frq,
                "data_dict": data_dict,
            }

            loss_dict = self.loss_function(data_for_loss_dict, device=self.device)
            loss = loss_dict["loss"]

            if self.enable_debug_plotting:
                self.debug_plotting(data_dict, data_for_loss_dict, loss_dict)

            if self.rank == 0:
                print(f"loss: {loss.item()}, inner loop iteration: {iteration}")

            loss.backward()

            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} gradient shape: {param.grad.shape}")
            #         print(param.grad)
            #     else:
            #         print(f"{name} has no gradient")

            # print(f"loss: {loss}, inner loop iteration: {iteration}")
            self.optimizer.step()
            self.optimizer.zero_grad()

            old_filters = filters_time.detach()
        return loss_dict

        # take optimizer step

    def save_checkpoint(self, current_ep, last_loss):
        """Function to save the model weights based on the checkpointing_mode and current model weights"""

        if self.checkpointing_mode != 'none':
            if self.checkpointing_mode == "all":
                fn = f"chkp_ep{str(current_ep).rjust(3,'0')}_loss{str(last_loss).replace('.','_')}.pth"
                print(f"saving checkpoint {fn}")
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, fn))

            elif self.checkpointing_mode == "last":

                fn = f"chkp_loss{str(last_loss).replace('.','_')}.pth"
                # ensure all files are removed so we only store the last model
                old_files = glob.glob(f"{self.checkpoint_path}/*")
                for file in old_files:
                    os.remove(file)

                print(f"saving checkpoint {fn}")
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, fn))

    def wandb_logger(self, loss_dict):
        wandb.log(data={"training_loss":loss_dict['loss'].item()})

    def debug_plotting(self, data_dict, data_for_loss_dict, loss_dict):
        """
        Function to plot the ffts and the filters of the model to verify that it is training
        Args:
            data_dict
            data_for_loss_dict
            loss_dict
        """
        fft_bz = loss_dict["fft_bz"].cpu().detach().numpy()
        fft_dz = loss_dict["fft_dz"].cpu().detach().numpy()
        fft_bz_des = loss_dict["fft_bz_des"].cpu().detach().numpy()

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

        for i, data_dict in enumerate(self.dataloader):
            # Ensure all data is passed to the correct device
            if self.device is not None:
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key].to(self.device)

            loss_dict = self.run_inner_feedback_training(data_dict, self.inner_loop_iterations)

            if self.rank == 0:
                percent_done = (i + 1) / total_batches * 100
                print(f"epoch {self.current_ep}: {percent_done:.2f}% complete")

            if self.log_to_wandb:
                self.wandb_logger(loss_dict)

        if self.rank == 0:
            self.save_checkpoint(current_ep=self.current_ep, last_loss=loss_dict['loss'].item())
            self.current_ep += 1
