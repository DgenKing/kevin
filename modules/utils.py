import ccxt
import logging
import time

logging.basicConfig(filename="logs/kevin.log", level=logging.INFO)

def fetch_data(symbol="SOL/USDT", days=500, run_num=0):
    try:
        exchange = ccxt.binance()
        # Shift back by run_num * days; end_time is days ago from now
        current_time = int(time.time() * 1000)
        end_time = current_time - (run_num * days * 24 * 60 * 60 * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        # Fetch data starting earlier to ensure coverage
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1d", since=start_time - (days * 24 * 60 * 60 * 1000), limit=days * 2)
        # Slice to exact 500-day window from start_time to end_time
        filtered_ohlcv = [row for row in ohlcv if start_time <= row[0] <= end_time]
        if len(filtered_ohlcv) < days:
            print(f"Warning: Only {len(filtered_ohlcv)} days available for Run {run_num}")
        filtered_ohlcv = filtered_ohlcv[-days:] if len(filtered_ohlcv) > days else filtered_ohlcv
        print(f"Run {run_num} - Start: {start_time}, End: {end_time}, First timestamp: {filtered_ohlcv[0][0]}, Last: {filtered_ohlcv[-1][0]}")
        normalized = []
        for row in filtered_ohlcv:
            norm_row = []
            for i in range(1, 6):  # OHLCV values
                col_vals = [r[i] for r in filtered_ohlcv]
                min_val, max_val = min(col_vals), max(col_vals)
                norm_val = (row[i] - min_val) / (max_val - min_val or 1)
                norm_row.append(norm_val)
            normalized.append(norm_row)
        print(f"Data sample (Run {run_num}): {normalized[:2]}")
        print(f"Data shape: {len(normalized)} rows, {len(normalized[0])} columns")
        return normalized
    except Exception as e:
        print(f"Error fetching data: {e}")
        return [[0.1, 0.2, 0.3, 0.4, 0.5]] * days

def log(message):
    logging.info(message)
