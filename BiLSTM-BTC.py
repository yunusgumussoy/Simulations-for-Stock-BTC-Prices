# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:23:31 2024

@author: Yunus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import datetime

# Step 1: Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())
prices = btc_data['Adj Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences of data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Using last 60 days to predict the next day
X, y = create_sequences(scaled_prices, seq_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Step 2: Build the BiLSTM model
model = Sequential()
model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(seq_length, 1)))
model.add(Bidirectional(LSTM(units=50)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 3: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Forecast future prices
forecast_horizon = 30  # Forecasting for the next 30 days
forecast_prices = []

last_sequence = X_test[-1]
for _ in range(forecast_horizon):
    next_price = model.predict(last_sequence.reshape(1, seq_length, 1))
    forecast_prices.append(next_price[0, 0])
    last_sequence = np.append(last_sequence[1:], next_price, axis=0)

forecast_prices = scaler.inverse_transform(np.array(forecast_prices).reshape(-1, 1))

# Step 5: Plot the results
plt.figure(figsize=(12, 6))
plt.plot(btc_data.index, prices, label='Historical Prices')
forecast_dates = pd.date_range(start=btc_data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
plt.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='red')
plt.title('Bitcoin Price Forecast using BiLSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Printing the forecasted prices
print("Forecasted Prices: ")
for date, price in zip(forecast_dates, forecast_prices):
    print(f"{date.date()}: {price[0]:.2f}")
