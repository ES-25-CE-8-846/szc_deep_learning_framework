from training import dataloader


if __name__ == "__main__":
    test_dataloader = dataloader.DefaultDataset(sound_dataset_root='/home/ai/datasets/audio/LibriSpeech/train-clean-100/',
                                                rir_dataset_root='/home/ai/datasets/kaggel/')
    print(len(test_dataloader))
    test_dataloader[2300]
