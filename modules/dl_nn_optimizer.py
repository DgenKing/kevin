from stable_baselines3 import PPO
import numpy as np
from . import utils
from .trading_env import TradingEnv
import os

class DLNNOptimizer:
    def __init__(self, config, run_num=0):
        self.config = config
        self.run_num = run_num
        self.model_path = "models/ppo_model.zip"
        env = TradingEnv()
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path, env=env)  # Let it use GPU if available
            print(f"Loaded existing PPO model from {self.model_path}")
        else:
            self.model = PPO("MlpPolicy", env=env)  # Let it use GPU if available
            print("Created new PPO model")
        self.data = utils.fetch_data(run_num=self.run_num)

    def optimize_strategy(self, strategy, results):
        print(f"Optimizing with results: {results}")
        if results["profit"] <= 0:
            self.model.learn(total_timesteps=self.config["dl_timesteps"])
            last_obs = np.array(self.data[-1], dtype=np.float32)
            action, _ = self.model.predict(last_obs)
            strategy["threshold"] = strategy["threshold"] + (0.01 if action == 1 else -0.01)
            strategy["profitable"] = False
            print(f"New threshold: {strategy['threshold']}")
            os.makedirs("models", exist_ok=True)
            self.model.save(self.model_path)
            print(f"Saved PPO model to {self.model_path}")
        else:
            strategy["profitable"] = True
            print(f"Strategy profitable, threshold: {strategy['threshold']}")
        return strategy
