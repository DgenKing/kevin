from . import utils

class DLNNBacktester:
    def __init__(self, config, run_num=0):
        self.config = config
        self.run_num = run_num

    def run_backtest(self, bot_path):
        data = utils.fetch_data(run_num=self.run_num)
        trades = []
        for i in range(len(data) - 1):
            current_price = data[i][3]  # Close price
            next_price = data[i + 1][3]
            if current_price != 0:  # Avoid division by zero
                profit = (next_price - current_price) / current_price
            else:
                profit = 0  # Default to no profit if price is zero
            trades.append({"profit": profit})
        total_profit = sum(trade["profit"] for trade in trades) / max(1, len(trades))
        return {"trades": trades, "profit": total_profit}
