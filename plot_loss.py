import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("training_log.csv")

plt.figure(figsize=(10, 6))
plt.plot(log["epoch"], log["loss"], label="Total Loss")
plt.plot(log["epoch"], log["mse"], label="MSE (Reconstruction)")
plt.plot(log["epoch"], log["kl"], label="KL Divergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
