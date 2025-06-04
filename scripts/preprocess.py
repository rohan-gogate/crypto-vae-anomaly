import pandas as pd
import numpy as np

df = pd.read_csv("data/btc_1m.csv",index_col="timestamp", parse_dates=True)
log_returns = np.log(df["close"]).diff()
df["log_returns"] = log_returns
df = df.dropna()
print(df['close'].head(10))
print(df['close'].describe())
import matplotlib.pyplot as plt
df['log_returns'].plot()
plt.title("Log Returns")
plt.show()
