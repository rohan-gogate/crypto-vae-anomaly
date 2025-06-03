import pandas as pd
import ccxt 
import time

def fetch_binance_ohlcv(symbol="BTC/USD", timeframe="1m", max_points=10000): #will try eth and sol later
    exchange = ccxt.binanceus()
    all_data = []
    since = exchange.parse8601('2023-01-01T00:00:00Z')  #starting jan 1 2023
    limit = 1000
    
    while len(all_data)*limit <max_points:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if not ohlcv:
            break
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        all_data.append(df)
        since = int(df['timestamp'].iloc[-1].timestamp() * 1000)
        time.sleep(1) #binance rate limit
        
    result = pd.concat(all_data).drop_duplicates(subset='timestamp').set_index("timestamp")
    return result

if __name__ == "__main__":
        df = fetch_binance_ohlcv()
        df.to_csv("data/btc_1m.csv")
        print("Data saved to data/btc_1m.csv")