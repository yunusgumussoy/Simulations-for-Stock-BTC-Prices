# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:02:47 2024

@author: Yunus
"""
# pip install gym stable-baselines3 yfinance

import numpy as np
import pandas as pd
import gym
from gym import spaces
import yfinance as yf
from stable_baselines3 import PPO

# Step 1: Create the Trading Environment
class BitcoinTradingEnv(gym.Env):
    def __init__(self, prices):
        super(BitcoinTradingEnv, self).__init__()
        self.prices = prices
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # Actions: Buy, Hold, Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        self.initial_balance = 1000
        self.current_balance = self.initial_balance
        self.btc_amount = 0

    def reset(self):
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.btc_amount = 0
        return np.array([self.prices[self.current_step]])

    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        
        if action == 0:  # Buy
            self.btc_amount += self.current_balance / current_price
            self.current_balance = 0
        elif action == 2:  # Sell
            self.current_balance += self.btc_amount * current_price
            self.btc_amount = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        if self.btc_amount > 0:  # If holding BTC
            reward = (self.btc_amount * current_price + self.current_balance) - self.initial_balance
        else:
            reward = self.current_balance - self.initial_balance
        
        return np.array([self.prices[self.current_step]]), reward, done, {}

# Step 2: Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end="2024-01-01")
prices = btc_data['Adj Close'].values

# Step 3: Initialize the Environment
env = BitcoinTradingEnv(prices)

# Step 4: Train the RL Agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

# Step 5: Test the agent
obs = env.reset()
for _ in range(len(prices) - 1):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break

print("Final Balance:", env.current_balance + env.btc_amount * prices[env.current_step])
