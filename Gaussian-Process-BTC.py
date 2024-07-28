# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:17:23 2024

@author: Yunus
"""

# pip install numpy pandas matplotlib yfinance GPy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import GPy
import datetime

# Step 1: Download Bitcoin data
try:
    btc_data = yf.download("BTC-USD", start="2023-01-01", end=datetime.date.today()) # for computation ease
    if btc_data.empty:
        raise ValueError("No data downloaded. Check the ticker or date range.")
    bitcoin_prices = btc_data['Adj Close'].values
    dates = np.arange(len(bitcoin_prices)).reshape(-1, 1)  # Convert dates to numeric format
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# Step 2: Create a Gaussian Process model
kernel = GPy.kern.RBF(input_dim=1)  # Radial Basis Function kernel
model = GPy.models.GPRegression(dates, bitcoin_prices.reshape(-1, 1), kernel)

# Step 3: Optimize the model
model.optimize()

# Step 4: Make predictions
future_dates = np.arange(len(bitcoin_prices), len(bitcoin_prices) + 30).reshape(-1, 1)
mean, variance = model.predict(future_dates)

# Step 5: Plot results
plt.figure(figsize=(10, 6))
plt.plot(dates, bitcoin_prices, label='Historical Prices')
plt.plot(future_dates, mean, label='Predicted Prices', color='red')
plt.fill_between(future_dates.flatten(),
                 mean.flatten() - 1.96 * np.sqrt(variance.flatten()),
                 mean.flatten() + 1.96 * np.sqrt(variance.flatten()),
                 color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('Bitcoin Price Forecast with Gaussian Processes')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

