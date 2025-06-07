import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from models.vae import VAE
from utils.dataset import SequenceDataset

input_dim = 60
latent_dim = 30
hidden_dim = 32
batch_size = 64
epochs = 100
lr = 0.001

sequences = np.load("data/sequences.npy")
dataset = SequenceDataset(sequences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

history = []
for epoch in range(epochs):
    model.train()
    beta = min(1.0, epoch/30)
    total_loss, total_mse, total_kl = 0,0,0

    for batch in dataloader:
        x, _ = batch
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, mse, kl = model.loss_function(recon, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_kl += kl.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, MSE: {total_mse:.4f}, KL: {total_kl:.4f}")
    history.append((epoch + 1, total_loss, total_mse, total_kl))


df = pd.DataFrame(history, columns=["epoch", "loss", "mse", "kl"])
df.to_csv("training_log.csv", index=False)