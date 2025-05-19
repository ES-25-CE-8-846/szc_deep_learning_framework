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


def train(rank, world_size, config):
    ddp_setup(rank, world_size)
    training_config = config["training_run"]
    loss_function = get_class_or_func(training_config["loss_function"])
    dataset = config["training_dataset"]
    validation_dataset = config["validation_dataset"]
    model = training_config["model"]
    filter_length = training_config["filter_length"]
    inner_loop_iterations = training_config["inner_loop_iterations"]
    save_path = training_config["savepath"]
    checkpointing_mode = training_config["checkpointing_mode"]
    epochs = training_config["epochs"]
    log_to_wandb = training_config["log_to_wandb"]
    batch_size = training_config["batch_size"]
    wandb_key_path = training_config["wandb_key_path"]



    if log_to_wandb and rank == 0:
        with open(wandb_key_path, "r") as wandb_key_file:
            wandb.login(key=wandb_key_file.read().strip(), relogin=True)

        run = wandb.init(
            name=save_path,
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="avs-846",
            # Set the wandb project where this run will be logged.
            project="test-scz",
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
        batch_size=batch_size,
        num_workers=4,
        sampler=DistributedSampler(dataset),
    )

    training_loop = trainer.Trainer(
        dataloader=torch_dataloader,
        validation_dataloader=validation_dataloader,
        loss_function=loss_function,
        model=model,
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

    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    training_config = config["training_run"]
    validation_config =config["validation_run"]

    # === Get components from config ===
    model_class = get_class_or_func(training_config["model"])
    loss_function = get_class_or_func(training_config["loss_function"])

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
    os.environ["MASTER_PORT"] = "22355"  # use a free port

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
        rir_dataset_root=rir_dataset_path,
        sound_snip_len=sound_snip_len,
        sound_snip_save_path="/tmp/sound_snips/val/",
        limit_used_soundclips=limit_uses_sound_snips,
        override_existing=False
    )

    config["validation_dataset"] = validation_dataset

    # === Instantiate Model ===
    model = model_class(
        input_channels=2,  # Update as needed
        output_shape=(3, filter_length),  # Adjust based on number of sources/mics
    )

    training_config["model"] = model

    world_size = n_gpus
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
