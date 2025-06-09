import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prices = np.load("data/close_prices.npy")          
anomalies = np.load("data/anomaly_indices.npy")

window = 5
prices_series = pd.Series(prices)
sma = prices_series.rolling(window=window).mean()

capital = 10000
position = 0
entry_price = 0
equity_curve = [capital]
positions = []

holding_period = 10
fee = 0.001

for i in range(len(prices)):
    if i in anomalies and position == 0 and not np.isnan(sma[i]):
        if prices[i] < sma[i]:
            position = 1
            entry_price = prices[i]
            positions.append((i, "buy", prices[i]))

        elif prices[i] > sma[i]:
            position = -1
            entry_price = prices[i]
            positions.append((i, "short", prices[i]))

    if position != 0 and i >= positions[-1][0] + holding_period:
        exit_price = prices[i]
        if position == 1:
            pnl = (exit_price - entry_price) * (1 - fee)
        else:
            pnl = (entry_price - exit_price) * (1 - fee)
        capital += pnl
        position = 0
        positions.append((i, "exit", prices[i]))

    equity_curve.append(capital)

plt.figure(figsize=(12, 6))
plt.plot(prices, label="Price")
plt.plot(sma, label=f"{window}-period SMA", linestyle="--")
for idx, action, price in positions:
    color = {"buy": "green", "short": "red", "exit": "blue"}[action]
    plt.scatter(idx, price, color=color, label=action if idx == positions[0][0] else "")
plt.title("Backtest with SMA-Filtered VAE Anomaly Strategy")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(equity_curve, label="Equity Curve")
plt.title("Equity Curve (SMA Filtered Anomalies)")
plt.xlabel("Time")
plt.ylabel("Capital")
plt.grid(True)
plt.legend()
plt.show()
print(f"PNL: {pnl:.2f}")
