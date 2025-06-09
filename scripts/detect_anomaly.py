import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vae import VAE
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import SequenceDataset
import pandas as pd

input_dim = 60
latent_dim = 30
hidden_dim = 32
batch_size = 64
window_size = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load("models/vae_trained.pth", map_location=device))
model.eval()

sequences = np.load("data/sequences.npy")
dataset = SequenceDataset(sequences)
dataloader = DataLoader(dataset, batch_size=batch_size) 

reconstruction_errors = []
with torch.no_grad():
    for x, _ in dataloader:
        x = x.to(device)
        recon, _, _ = model(x)
        mse = ((x - recon) ** 2).mean(dim=1)  
        reconstruction_errors.extend(mse.cpu().numpy())
reconstruction_errors = np.array(reconstruction_errors)

mean_error = np.mean(reconstruction_errors)
std_error = np.std(reconstruction_errors)
threshold = mean_error + 3*std_error
anomaly_indices = np.where(reconstruction_errors > threshold)[0]
anomaly_timestamps = [int(i+window_size) for i in anomaly_indices]  
print(f"Anomaly threshold: {threshold:.4f}")
print(f"Detected {len(anomaly_indices)} anomalies")

df = pd.DataFrame({
    "timestamp_index": anomaly_timestamps,
    "reconstruction_error": reconstruction_errors[anomaly_indices]
})
df.to_csv("data/anomalies.csv", index=False)
print("Saved anomalies to data/anomalies.csv")
np.save("data/anomaly_indices.npy", anomaly_indices)