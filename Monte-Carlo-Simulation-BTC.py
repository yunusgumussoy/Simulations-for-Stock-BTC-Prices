# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 14:55:15 2024

@author: Yunus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

# Step 1: Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())

# print(btc_data.head()) #data check

prices = btc_data['Adj Close'].values

# Step 2: Calculate daily returns
returns = np.diff(prices) / prices[:-1]

# Step 3: Analyze the distribution of returns (mean and standard deviation)
mu = np.mean(returns)
sigma = np.std(returns)

# Step 4: Set up the simulation parameters
num_simulations = 10000
num_days = 365  # 1 year

# Step 5: Generate random returns
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = prices[-1]

for t in range(1, num_days):
    random_returns = np.random.normal(mu, sigma, num_simulations)
    simulated_prices[:, t] = simulated_prices[:, t - 1] * (1 + random_returns)

# Step 6: Analyze and plot the results
plt.figure(figsize=(10, 6))
for i in range(num_simulations):
    plt.plot(simulated_prices[i, :], color='blue', alpha=0.01)
plt.title('Monte Carlo Simulation of Bitcoin Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

# Step 7: Statistical analysis
final_prices = simulated_prices[:, -1]
mean_price = np.mean(final_prices)
median_price = np.median(final_prices)
percentile_95 = np.percentile(final_prices, 95)
percentile_5 = np.percentile(final_prices, 5)

print(f"Mean Price after 1 year: {mean_price}")
print(f"Median Price after 1 year: {median_price}")
print(f"95th Percentile Price after 1 year: {percentile_95}")
print(f"5th Percentile Price after 1 year: {percentile_5}")

