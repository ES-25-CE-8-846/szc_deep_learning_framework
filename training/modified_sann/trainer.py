import torch
from torch.utils.data import DataLoader
from training.modified_sann.dataloader import AudioFilterDataset
from models.modified_sann import AudioFilterEstimator
from training.modified_sann.loss_function import sound_loss

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                # Forward pass
                outputs = self.model(batch['input'])
                loss = self.loss_fn(outputs, batch['target'])

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation step
            self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in self.val_loader:
                outputs = self.model(batch['input'])
                loss = self.loss_fn(outputs, batch['target'])
                total_loss += loss.item()

        print(f"Epoch {epoch}: Validation Loss: {total_loss / len(self.val_loader)}")

if __name__ == "__main__":
    # Declare environment variables
    NUM_BZ_MICS = 3
    NUM_DZ_MICS = 11
    NUM_SPEAKERS = 3
    OUTPUT_DIM = NUM_SPEAKERS * 2 * 239 # no_of speakers * 2 (real + imag) * 239 bins

    # Make the dataset & Dataloader
    dataset = AudioFilterDataset(data_dir="/home/morten/GitHub/dataset/shoebox/alfredo-request")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = AudioFilterEstimator(num_mics=NUM_BZ_MICS+NUM_DZ_MICS, output_dim=OUTPUT_DIM)
    otimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = sound_loss()