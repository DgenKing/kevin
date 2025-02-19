import os

class DLNNCodeGenerator:
    def __init__(self, config):
        self.config = config
        self.bot_path = "bots/dl_nn_trading_bot.py"

    def generate_bot_code(self, strategy):
        code = f"""
# DL NN Trading Bot (Backtest Version)
# Generated automatically

def run_backtest(data):
    threshold = {strategy['threshold']}
    risk_percent = {strategy['risk_percent']}
    trades = []
    
    for i in range(len(data) - 1):
        current_price = data[i][3]  # Close price
        next_price = data[i + 1][3]
        if current_price > threshold:
            # Simulate a buy
            profit = (next_price - current_price) / current_price * risk_percent
            trades.append({{'profit': profit}})
    
    total_profit = sum(trade['profit'] for trade in trades) / max(1, len(trades))
    return {{'trades': trades, 'profit': total_profit}}

if __name__ == "__main__":
    print("This is a backtest bot. Run it via DLNNBacktester.")
"""
        os.makedirs("bots", exist_ok=True)
        with open(self.bot_path, "w") as f:
            f.write(code.strip())
        return self.bot_path
