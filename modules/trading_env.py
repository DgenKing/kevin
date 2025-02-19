import gymnasium as gym
import numpy as np
from . import utils

class TradingEnv(gym.Env):
    def __init__(self):
        super(TradingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # Buy (1) or hold (0)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.data = utils.fetch_data()
        self.current_step = 0
        self.balance = 1000.0
        self.position = 0.0
        self.max_steps = len(self.data) - 1  # Prevent stepping beyond data

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = 1000.0
        self.position = 0.0
        return np.array(self.data[self.current_step], dtype=np.float32), {}

    def step(self, action):
        current_price = self.data[self.current_step][4]  # Close price
        reward = 0
        if action == 1 and self.balance > 0 and current_price > 0:  # Buy only if price > 0
            self.position = self.balance / (current_price + 1e-8)  # Add epsilon to avoid zero division
            self.balance = 0
        elif action == 0 and self.position > 0:  # Sell
            self.balance = self.position * current_price
            reward = self.balance - 1000.0  # Profit/loss
            self.position = 0

        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = np.array(self.data[self.current_step] if not done else np.zeros(5), dtype=np.float32)
        return obs, reward, done, False, {}

# Register the environment
gym.register(id="TradingEnv-v0", entry_point="modules.trading_env:TradingEnv")
