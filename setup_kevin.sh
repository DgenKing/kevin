#!/bin/bash

# Base directory
BASE_DIR="kevin"

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$BASE_DIR/modules" "$BASE_DIR/bots" "$BASE_DIR/data" "$BASE_DIR/models" "$BASE_DIR/logs"

# Create main.py
echo "Creating main.py..."
cat << 'EOF' > "$BASE_DIR/main.py"
import yaml
from modules.dl_nn_strategy_generator import DLNNStrategyGenerator
from modules.dl_nn_code_generator import DLNNCodeGenerator
from modules.dl_nn_backtester import DLNNBacktester
from modules.dl_nn_optimizer import DLNNOptimizer
from modules.dl_nn_deployer import DLNNDeployer
from modules.dl_nn_trainer import DLNNTrainer

def run_kevin():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    trainer = DLNNTrainer(config)
    trainer.train_models()

    sg = DLNNStrategyGenerator(config)
    strategy = sg.generate_strategy()

    cg = DLNNCodeGenerator(config)
    bot_path = cg.generate_bot_code(strategy)

    bt = DLNNBacktester(config)
    results = bt.run_backtest(bot_path)

    opt = DLNNOptimizer(config)
    optimized_strategy = opt.optimize_strategy(strategy, results)

    if optimized_strategy["profitable"]:
        cg.generate_bot_code(optimized_strategy)
        dp = DLNNDeployer(config)
        dp.deploy_bot(bot_path)

if __name__ == "__main__":
    run_kevin()
EOF

# Create modules/dl_nn_strategy_generator.py
echo "Creating dl_nn_strategy_generator.py..."
cat << 'EOF' > "$BASE_DIR/modules/dl_nn_strategy_generator.py"
import torch
import torch.nn as nn
from . import utils

class DLNNStrategyGenerator:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()

    def load_model(self):
        model = nn.LSTM(input_size=5, hidden_size=64, num_layers=2, batch_first=True)
        model.load_state_dict(torch.load("models/dl_nn_predictor.pt"))
        model.eval()
        return model

    def generate_strategy(self):
        data = utils.fetch_data()[-50:]
        tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred, _ = self.model(tensor_data)
            threshold = pred[0, -1].item()
        return {"type": "dl_nn_price", "threshold": threshold, "risk_percent": 0.5}
EOF

# Create modules/dl_nn_code_generator.py
echo "Creating dl_nn_code_generator.py..."
cat << 'EOF' > "$BASE_DIR/modules/dl_nn_code_generator.py"
class DLNNCodeGenerator:
    def __init__(self, config):
        self.config = config

    def generate_bot_code(self, strategy):
        bot_code = f"""
import ccxt
import torch
import torch.nn as nn

exchange = ccxt.binance({{'apiKey': '{self.config['api_key']}', 'secret': '{self.config['api_secret']}'}})
model = nn.LSTM(input_size=5, hidden_size=64, num_layers=2, batch_first=True)
model.load_state_dict(torch.load('models/dl_nn_predictor.pt'))
model.eval()

def predict(data):
    tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred, _ = model(tensor_data)
    return pred[0, -1].item()

def trade():
    data = exchange.fetch_ohlcv('SOL/USDT', '1d')[-50:]
    pred = predict(data)
    if pred > {strategy['threshold']}:
        exchange.create_market_buy_order('SOL/USDT', {strategy['risk_percent']} * exchange.fetch_balance()['free']['USDT'])
"""
        bot_path = "bots/dl_nn_trading_bot.py"
        with open(bot_path, "w") as f:
            f.write(bot_code)
        return bot_path
EOF

# Create modules/dl_nn_backtester.py
echo "Creating dl_nn_backtester.py..."
cat << 'EOF' > "$BASE_DIR/modules/dl_nn_backtester.py"
import importlib.util
from . import utils

class DLNNBacktester:
    def __init__(self, config):
        self.config = config

    def run_backtest(self, bot_path):
        spec = importlib.util.spec_from_file_location("dl_nn_trading_bot", bot_path)
        bot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot)

        data = utils.fetch_data()
        trades = []
        for i in range(50, len(data)):
            pred = bot.predict(data[i-50:i])
            if pred > 0:
                profit = data[i][4] - data[i-1][4]
                trades.append({"profit": profit})
        profit = sum(t["profit"] for t in trades)
        return {"trades": trades, "profit": profit}
EOF

# Create modules/dl_nn_optimizer.py
echo "Creating dl_nn_optimizer.py..."
cat << 'EOF' > "$BASE_DIR/modules/dl_nn_optimizer.py"
from stable_baselines3 import PPO
from . import utils

class DLNNOptimizer:
    def __init__(self, config):
        self.config = config
        self.model = PPO("MlpPolicy", env="TradingEnv")

    def optimize_strategy(self, strategy, results):
        if results["profit"] <= 0:
            self.model.learn(total_timesteps=self.config["dl_timesteps"])
            action, _ = self.model.predict({"profit": results["profit"], "threshold": strategy["threshold"]})
            strategy["threshold"] = action[0]
            strategy["profitable"] = False
        else:
            strategy["profitable"] = True
        return strategy
EOF

# Create modules/dl_nn_deployer.py
echo "Creating dl_nn_deployer.py..."
cat << 'EOF' > "$BASE_DIR/modules/dl_nn_deployer.py"
import subprocess

class DLNNDeployer:
    def __init__(self, config):
        self.config = config

    def deploy_bot(self, bot_path):
        subprocess.run(["python", bot_path])
        utils.log("DL/NN Bot deployed successfully!")
EOF

# Create modules/dl_nn_trainer.py
echo "Creating dl_nn_trainer.py..."
cat << 'EOF' > "$BASE_DIR/modules/dl_nn_trainer.py"
import torch
import torch.nn as nn
from . import utils

class DLNNTrainer:
    def __init__(self, config):
        self.config = config
        self.predictor = nn.LSTM(input_size=5, hidden_size=64, num_layers=2, batch_first=True)

    def train_models(self):
        data = utils.fetch_data()
        tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        opt = torch.optim.Adam(self.predictor.parameters())
        criterion = nn.MSELoss()

        for epoch in range(self.config["nn_epochs"]):
            pred, _ = self.predictor(tensor_data[:, :-1])
            loss = criterion(pred, tensor_data[:, 1:, 4:5])
            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.save(self.predictor.state_dict(), "models/dl_nn_predictor.pt")
EOF

# Create modules/utils.py
echo "Creating utils.py..."
cat << 'EOF' > "$BASE_DIR/modules/utils.py"
import ccxt
import logging

logging.basicConfig(filename="logs/kevin.log", level=logging.INFO)

def fetch_data(symbol="SOL/USDT"):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1d")
    return [[(v - min(ohlcv, key=lambda x: x[i])[i]) / (max(ohlcv, key=lambda x: x[i])[i] - min(ohlcv, key=lambda x: x[i])[i])
             for i, v in enumerate(row)] for row in ohlcv]

def log(message):
    logging.info(message)
EOF

# Create config.yaml
echo "Creating config.yaml..."
cat << 'EOF' > "$BASE_DIR/config.yaml"
api_key: "your_binance_api_key"
api_secret: "your_binance_api_secret"
exchange: "binance"
data_dir: "data/"
log_dir: "logs/"
model_dir: "models/"
nn_epochs: 50
dl_timesteps: 10000
EOF

# Create requirements.txt
echo "Creating requirements.txt..."
cat << 'EOF' > "$BASE_DIR/requirements.txt"
ccxt
torch
stable-baselines3
pyyaml
EOF

# Create __init__.py for modules
echo "Creating __init__.py..."
touch "$BASE_DIR/modules/__init__.py"

# Set permissions
echo "Setting executable permissions..."
chmod +x "$BASE_DIR/main.py"
chmod +x "$BASE_DIR/setup_kevin.sh"  # Make this script executable if saved separately

# Done
echo "Kevin structure created successfully in $BASE_DIR!"
echo "Next steps:"
echo "1. cd $BASE_DIR"
echo "2. python3 -m venv venv"
echo "3. source venv/bin/activate"
echo "4. pip install -r requirements.txt"
echo "5. Update config.yaml with your API keys"
echo "6. Run: python3 main.py"
