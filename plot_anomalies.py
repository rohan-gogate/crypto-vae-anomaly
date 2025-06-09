import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Rohan\Code\crypto-vae-anomaly\data\btc_1m.csv")
anomalies = pd.read_csv("data/anomalies.csv")["timestamp_index"].values
plt.figure(figsize=(15, 6))
plt.plot(df["close"], label="Close Price", linewidth=1)

plt.scatter(anomalies, df.loc[anomalies, "close"], color="red", label="Anomaly", s=10)

plt.title("Price with Anomalies")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
