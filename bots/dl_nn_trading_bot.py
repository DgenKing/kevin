# DL NN Trading Bot (Backtest Version)
# Generated automatically

def run_backtest(data):
    threshold = 0.3322555720806122
    risk_percent = 0.5
    trades = []
    
    for i in range(len(data) - 1):
        current_price = data[i][3]  # Close price
        next_price = data[i + 1][3]
        if current_price > threshold:
            # Simulate a buy
            profit = (next_price - current_price) / current_price * risk_percent
            trades.append({'profit': profit})
    
    total_profit = sum(trade['profit'] for trade in trades) / max(1, len(trades))
    return {'trades': trades, 'profit': total_profit}

if __name__ == "__main__":
    print("This is a backtest bot. Run it via DLNNBacktester.")