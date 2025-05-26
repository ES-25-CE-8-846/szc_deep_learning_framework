import argparse
import yaml
import importlib
import torch
import torch.utils.data
from training import dataloader, trainer  # Assuming your custom Trainer class is here
import os
import shutil
import wandb
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import soundfile
import numpy as np

def load_test_sound():
    sound_file_dir = "./testing_data/longer_speech_samples/resampled/"
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


def get_class_or_func(path):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train(rank, world_size, config, validation_test_sound):
    ddp_setup(rank, world_size)
    training_config = config["training_run"]
    validation_config = config["validation_run"]
    loss_function = get_class_or_func(training_config["loss_function"])

    dataset = config["training_dataset"]
    validation_dataset = config["validation_dataset"]

    learning_rate = training_config["learning_rate"]
    model = training_config["model"]
    filter_length = training_config["filter_length"]
    inner_loop_iterations = training_config["inner_loop_iterations"]
    save_path = training_config["savepath"]
    checkpointing_mode = training_config["checkpointing_mode"]
    epochs = training_config["epochs"]
    log_to_wandb = training_config["log_to_wandb"]
    batch_size = training_config["batch_size"]
    wandb_key_path = training_config["wandb_key_path"]

    validation_metrics = validation_config["metrics"]



    loss_weights = training_config['loss_weights']

    if log_to_wandb and rank == 0:
        with open(wandb_key_path, "r") as wandb_key_file:
            wandb.login(key=wandb_key_file.read().strip(), relogin=True)

        run = wandb.init(
            name=save_path,
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="avs-846",
            # Set the wandb project where this run will be logged.
            project="scz-test-loss-weights",
            # Track hyperparameters and run metadata.
            config=config,
        )


    torch_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        sampler=DistributedSampler(dataset),
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=world_size,
        num_workers=1,
        sampler=DistributedSampler(validation_dataset),
    )

    training_loop = trainer.Trainer(
        dataloader=torch_dataloader,
        validation_dataloader=validation_dataloader,
        validation_metrics=validation_metrics,
        validation_test_sound=validation_test_sound,
        loss_function=loss_function,
        loss_weights=loss_weights,
        model=model,
        learning_rate = learning_rate,
        world_size=world_size,
        rank=rank,
        filter_length=filter_length,
        inner_loop_iterations=inner_loop_iterations,
        save_path=save_path,
        checkpointing_mode=checkpointing_mode,
        log_to_wandb=log_to_wandb,
    )

    # === wandb init ===

    # === Run Training ===
    for epoch in range(epochs):
        print(f"=== Running Epoch {epoch + 1}/{epochs} ===")
        training_loop.run_epoch(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    validation_test_sound = load_test_sound()
    print(f"loaded validation test sound with shape {validation_test_sound.shape}")
    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    training_config = config["training_run"]
    validation_config =config["validation_run"]

    # === Get components from config ===
    model_class = get_class_or_func(training_config["model"])
    loss_function = get_class_or_func(training_config["loss_function"])
    loss_weights = training_config['loss_weights']

    sound_dataset_path = training_config["sound_dataset_path"]
    rir_dataset_path = training_config["rir_dataset_path"]
    batch_size = training_config["batch_size"]
    filter_length = training_config["filter_length"]
    inner_loop_iterations = training_config["inner_loop_iterations"]
    n_gpus = training_config["n_gpus"]
    save_path = training_config["savepath"]
    checkpointing_mode = training_config["checkpointing_mode"]
    learning_rate = training_config["learning_rate"]
    sound_snip_len = training_config["sound_snip_length"]
    limit_uses_sound_snips = training_config["limit_used_sound_snips"]
    epochs = training_config["epochs"]
    log_to_wandb = training_config["log_to_wandb"]
    wandb_key_path = training_config["wandb_key_path"]

    validation_rir_dataset_path = validation_config["rir_dataset_path"]
    validation_sound_dataset_path = validation_config["sound_dataset_path"]


    assert batch_size % n_gpus == 0
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "22356"  # use a free port

    os.makedirs(save_path, exist_ok=True)
    shutil.copy(args.config_path, os.path.join(save_path, "config.yaml"))

    # === Build Dataset & DataLoader ===
    dataset = dataloader.DefaultDataset(
        sound_dataset_root=sound_dataset_path,
        rir_dataset_root=rir_dataset_path,
        sound_snip_len=sound_snip_len,
        sound_snip_save_path="/tmp/sound_snips/train/",
        limit_used_soundclips=limit_uses_sound_snips,
        override_existing=True,  # Add this if needed
    )

    config["training_dataset"] = dataset


    validation_dataset = dataloader.DefaultDataset(
        sound_dataset_root=validation_sound_dataset_path,
        rir_dataset_root=validation_rir_dataset_path,
        sound_snip_len=sound_snip_len,
        sound_snip_save_path="/tmp/sound_snips/val/",
        limit_used_soundclips=limit_uses_sound_snips,
        override_existing=True,
    )

    config["validation_dataset"] = validation_dataset

    # === Instantiate Model ===
    model = model_class(
        input_channels=2,  # Update as needed
        output_shape=(3, filter_length),  # Adjust based on number of sources/mics
    )

    training_config["model"] = model

    world_size = n_gpus
    mp.spawn(train, args=(world_size, config, validation_test_sound), nprocs=world_size, join=True)
