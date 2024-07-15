# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:54:20 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf
import datetime

# Step 1: Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())
prices = btc_data['Adj Close']

# Step 2: Calculate daily returns
returns = 100 * prices.pct_change().dropna()

# Step 3: Fit the GARCH model
# Using a GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1)
garch_result = model.fit(disp='off')
print(garch_result.summary())

# Step 4: Forecast future volatility and returns
forecast_horizon = 30  # Forecasting for the next 30 days
garch_forecast = garch_result.forecast(start=returns.index[-1], horizon=forecast_horizon)

# Extracting the forecasted conditional variances and means
forecast_variances = garch_forecast.variance.values[-1, :]
forecast_returns = garch_forecast.mean.values[-1, :]

# Step 5: Construct price forecasts based on the returns and initial price
last_price = prices.iloc[-1]
price_forecast = [last_price]

for i in range(forecast_horizon):
    next_return = forecast_returns[i] / 100  # converting back to percentage
    next_price = price_forecast[-1] * (1 + next_return)
    price_forecast.append(next_price)

# Removing the initial price to keep only forecasted prices
price_forecast = price_forecast[1:]

# Step 6: Plot the results
plt.figure(figsize=(12, 6))
plt.plot(prices.index, prices, label='Historical Prices')
forecast_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
plt.plot(forecast_dates, price_forecast, label='Forecasted Prices', color='red')
plt.title('Bitcoin Price Forecast using GARCH')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Printing the forecasted prices
print("Forecasted Prices: ")
for date, price in zip(forecast_dates, price_forecast):
    print(f"{date.date()}: {price:.2f}")
