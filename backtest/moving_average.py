import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prices = np.load("data/close_prices.npy")
anomalies = np.load("data/anomaly_indices.npy")

window = 5
holding_period = 10
fee = 0.001
initial_capital = 10000

prices_series = pd.Series(prices)
sma = prices_series.rolling(window=window).mean()

capital = initial_capital
position = 0
entry_price = 0
equity_curve = [capital]
positions = []

pnl_per_trade = []
num_profitable_trades = 0
total_trades = 0

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

        pnl_per_trade.append(pnl)
        if pnl > 0:
            num_profitable_trades += 1
        total_trades += 1

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

final_equity = equity_curve[-1]
total_return = (final_equity / initial_capital - 1) * 100
returns = np.diff(equity_curve) / equity_curve[:-1]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
drawdown = equity_curve / np.maximum.accumulate(equity_curve) - 1
max_drawdown = np.min(drawdown)
win_rate = num_profitable_trades / total_trades if total_trades > 0 else 0
avg_return = np.mean(pnl_per_trade) if pnl_per_trade else 0

print(f"Total Return: {total_return:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Average Return per Trade: {avg_return:.2f}")
print(f"Total Trades: {total_trades}, Profitable Trades: {num_profitable_trades}")
