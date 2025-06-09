import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/btc_1m.csv",index_col="timestamp", parse_dates=True)
log_returns = np.log(df["close"]).diff()
mean = log_returns.mean()
std = log_returns.std()
log_returns_normalized = (log_returns - mean)/std
df["log_returns_normalized"] = log_returns_normalized
df = df.dropna()

window_size = 60 #can be adjusted later
returns = np.array(df["log_returns_normalized"].values)
sequences = []
for i in range(len(returns) - window_size):
    sequences.append(returns[i:i + window_size])
sequences = np.array(sequences)
np.save("data/sequences.npy", sequences) #now we have sequences of log returns as a 2d numpy array
close_prices = df["close"].values
np.save("data/close_prices.npy", close_prices) #will need later for backtesting
