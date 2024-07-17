# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:55:38 2024

@author: Yunus
"""

# Gated Recurrent Unit (GRU) model for predicting and forecasting Bitcoin prices

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data
end_date = datetime.date.today()
data = yf.download("BTC-USD", start="2015-01-01", end=end_date)

# Use 'Adj Close' for modeling
data = data[['Adj Close']]

# Create features and target variable
data['Returns'] = data['Adj Close'].pct_change()
data['Moving_Average_5'] = data['Adj Close'].rolling(window=5).mean()
data['Moving_Average_10'] = data['Adj Close'].rolling(window=10).mean()
data['Volatility'] = data['Adj Close'].rolling(window=5).std()

# Drop NaNs
data = data.dropna()

# Define features and target variable
X = data[['Returns', 'Moving_Average_5', 'Moving_Average_10', 'Volatility']].values
y = data['Adj Close'].values

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape the input for GRU [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Initialize and build the GRU model
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-Squared: {r2}')

# Forecasting for the next 30 days
forecast_horizon = 30
forecasted_data = []

last_known_data = data.iloc[-1][['Returns', 'Moving_Average_5', 'Moving_Average_10', 'Volatility']].values.reshape(1, -1)
last_known_data_scaled = scaler_X.transform(last_known_data)

for i in range(forecast_horizon):
    # Reshape the input for GRU
    last_known_data_scaled = last_known_data_scaled.reshape(1, 1, -1)
    
    # Predict the next closing price
    next_close_price_scaled = model.predict(last_known_data_scaled)
    next_close_price = scaler_y.inverse_transform(next_close_price_scaled)[0][0]
    
    # Append the predicted closing price to the forecasted data
    forecasted_data.append(next_close_price)
    
    # Update last known data
    new_row = {
        'Returns': (next_close_price - last_known_data[0][0]) / last_known_data[0][0],
        'Moving_Average_5': (last_known_data[0][1] * 4 + next_close_price) / 5,
        'Moving_Average_10': (last_known_data[0][2] * 9 + next_close_price) / 10,
        'Volatility': np.std(data['Adj Close'].values[-4:].tolist() + [next_close_price])
    }
    
    last_known_data = np.array(list(new_row.values())).reshape(1, -1)
    last_known_data_scaled = scaler_X.transform(last_known_data)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Prices', alpha=0.7)
plt.plot(pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon), forecasted_data, label='Forecasted Prices', color='red')
plt.title('Bitcoin Price Prediction and Forecast using GRU')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
