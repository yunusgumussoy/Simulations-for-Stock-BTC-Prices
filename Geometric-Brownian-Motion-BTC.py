# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:22:36 2024

@author: Yunus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

# Step 1: Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())

prices = btc_data['Adj Close'].values

# Step 2: Calculate daily returns
log_returns = np.log(prices[1:] / prices[:-1])
mu = np.mean(log_returns)
sigma = np.std(log_returns)

# Step 3: Set up the simulation parameters
num_simulations = 10000
num_days = 365  # 1 year
last_price = prices[-1]

# Step 4: Generate random price paths using GBM
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = last_price

for t in range(1, num_days):
    Z = np.random.standard_normal(num_simulations)
    simulated_prices[:, t] = simulated_prices[:, t - 1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

# Step 5: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_days), simulated_prices.T, color='blue', alpha=0.01)
plt.title('Monte Carlo Simulation of Bitcoin Prices using GBM')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

# Step 6: Analyze and print some statistics
final_prices = simulated_prices[:, -1]
mean_price = np.mean(final_prices)
median_price = np.median(final_prices)
percentile_95 = np.percentile(final_prices, 95)
percentile_5 = np.percentile(final_prices, 5)

print(f"Mean Price after 1 year: {mean_price}")
print(f"Median Price after 1 year: {median_price}")
print(f"95th Percentile Price after 1 year: {percentile_95}")
print(f"5th Percentile Price after 1 year: {percentile_5}")
