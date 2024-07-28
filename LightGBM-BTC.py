# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:29:23 2024

@author: Yunus
"""


# pip install lightgbm

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

# Load data
data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())

# Create lag features
data['lag_1'] = data['Adj Close'].shift(1)
data['lag_2'] = data['Adj Close'].shift(2)
data['lag_3'] = data['Adj Close'].shift(3)

# Create moving averages
data['ma_7'] = data['Adj Close'].rolling(window=7).mean()
data['ma_14'] = data['Adj Close'].rolling(window=14).mean()

# Drop missing values
data.dropna(inplace=True)

# Define features and target
features = ['lag_1', 'lag_2', 'lag_3', 'ma_7', 'ma_14']
target = 'Adj Close'
X = data[features]
y = data[target]

# Train-Test Split
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Define the callback for early stopping
callbacks = [lgb.early_stopping(stopping_rounds=10)]

# Train the model
model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data], callbacks=callbacks)

# Predict
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')

# Step 5: Forecast Future Prices
future_steps = 30  # Number of steps to forecast
last_data = data.iloc[-1][features].values.reshape(1, -1)

future_prices = []
for _ in range(future_steps):
    future_price = model.predict(last_data)
    future_prices.append(future_price[0])
    
    # Update last_data with the new forecasted price
    new_lag_1 = future_price[0]
    new_lag_2 = last_data[0][0]  # Previous lag_1
    new_lag_3 = last_data[0][1]  # Previous lag_2
    new_ma_7 = np.mean([new_lag_1] + list(last_data[0][2:6]))
    new_ma_14 = np.mean([new_lag_1] + list(last_data[0][3:6]) + list(last_data[0][4:6]))
    last_data = np.array([[new_lag_1, new_lag_2, new_lag_3, new_ma_7, new_ma_14]])

print(f'Forecasted Prices: {future_prices}')

# Visualize Original Prices, Predicted Prices, and Forecasted Prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Adj Close'], label='Original Prices')
plt.plot(data.index[train_size:], y_pred, label='Predicted Prices', color='orange')

# Extend the dates for forecasted prices
forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps)
plt.plot(forecast_dates, future_prices, label='Forecasted Prices', color='red')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Prices: Original, Predicted, and Forecasted')
plt.legend()
plt.show()
