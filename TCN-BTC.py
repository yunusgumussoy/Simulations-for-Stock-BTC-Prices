# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:50:25 2024

@author: Yunus
"""

# Temporal Convolutional Networks (TCNs) are a type of neural network architecture 
# designed specifically for sequential data tasks, such as time series forecasting

# pip install pandas numpy yfinance tensorflow scikit-learn

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Input, Dropout, Flatten
from tensorflow.keras.models import Model
import datetime
import matplotlib.pyplot as plt

# Load and prepare the data
try:
    btc_data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())
    if btc_data.empty:
        raise ValueError("No data downloaded. Check the ticker or date range.")
    bitcoin_prices = btc_data['Adj Close'].values.reshape(-1, 1)
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(bitcoin_prices)

# Define parameters
sequence_length = 30

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, sequence_length)

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for TCN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define TCN model
def create_tcn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(32, kernel_size=3, padding='causal', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)  # Flatten the 3D tensor to 2D
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    return model

# Create and compile the model
model = create_tcn_model((sequence_length, 1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions
predictions = scaler.inverse_transform(predictions)

# Inverse transform the actual values for plotting
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, color='blue', label='Actual Bitcoin Prices')
plt.plot(predictions, color='red', label='Predicted Bitcoin Prices')
plt.title('Actual vs Predicted Bitcoin Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Evaluate the model
def evaluate_model(predictions, y_test, scaler):
    # Inverse transform the test data
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(np.mean((predictions - y_test_scaled) ** 2))
    return rmse

rmse = evaluate_model(predictions, y_test, scaler)
print(f"Root Mean Squared Error: {rmse}")

# Forecasting the next 30 days
def forecast_next_days(model, data, scaler, seq_length, days):
    last_sequence = data[-seq_length:]
    forecast = []
    for _ in range(days):
        last_sequence_reshaped = last_sequence.reshape((1, seq_length, 1))
        next_prediction = model.predict(last_sequence_reshaped)
        forecast.append(next_prediction[0, 0])
        last_sequence = np.append(last_sequence, next_prediction)[1:]
    
    forecast = np.array(forecast).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast)
    return forecast

# Forecast the next 30 days
next_30_days = forecast_next_days(model, scaled_data, scaler, sequence_length, 30)

# Print forecasted prices
print("Forecasted Bitcoin Prices for the next 30 days:")
print(next_30_days)

# Plot forecasted prices
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(bitcoin_prices)), bitcoin_prices, label='Historical Bitcoin Prices')
plt.plot(np.arange(len(bitcoin_prices), len(bitcoin_prices) + 30), next_30_days, color='red', label='Forecasted Bitcoin Prices')
plt.title('Historical and Forecasted Bitcoin Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
