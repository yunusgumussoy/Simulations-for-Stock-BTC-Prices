# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:27:27 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import yfinance as yf
import datetime

# Step 1: Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())
prices = btc_data['Adj Close'].values

# Step 2: Split the data into training and testing sets
train_data, test_data = prices[:int(len(prices)*0.8)], prices[int(len(prices)*0.8):]
train_data = pd.Series(train_data, index=btc_data.index[:int(len(prices)*0.8)])
test_data = pd.Series(test_data, index=btc_data.index[int(len(prices)*0.8):])

# Step 3: Use auto_arima to find the best parameters
model = auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)
print(model.summary())

# Step 4: Fit the ARIMA model
arima_model = ARIMA(train_data, order=model.order)
arima_result = arima_model.fit()

# Step 5: Forecast future prices
forecast_steps = len(test_data)
forecast = arima_result.forecast(steps=forecast_steps)

# Step 6: Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Test Data')
plt.plot(test_data.index, forecast, label='Forecast')
plt.title('Bitcoin Price Forecast using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

