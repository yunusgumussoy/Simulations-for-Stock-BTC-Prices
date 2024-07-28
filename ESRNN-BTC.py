# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:30:29 2024

@author: Yunus
"""

# ESRNN is a hybrid time series forecasting model that combines the strengths of exponential smoothing 
# and recurrent neural networks (RNNs)

# pip install pandas numpy yfinance tensorflow scikit-learn

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime

# Load and prepare the data
btc_data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())
bitcoin_prices = btc_data['Adj Close'].values.reshape(-1, 1)

# print (btc_data)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(bitcoin_prices)

# Define parameters
sequence_length = 30

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, sequence_length)

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define ESRNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, color='blue', label='Actual Bitcoin Prices')
plt.plot(predictions, color='red', label='Predicted Bitcoin Prices')
plt.title('Actual vs Predicted Bitcoin Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()