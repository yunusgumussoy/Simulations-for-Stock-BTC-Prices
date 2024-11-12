# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:52:55 2024

@author: Yunus
"""

# pip install yfinance pandas scikit-learn xgboost matplotlib

import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end=datetime.date.today())

# Prepare the data
btc_data['Date'] = btc_data.index
btc_data = btc_data[['Date', 'Close']].reset_index(drop=True)

# Feature Engineering
btc_data['day'] = btc_data['Date'].dt.day
btc_data['month'] = btc_data['Date'].dt.month
btc_data['year'] = btc_data['Date'].dt.year

# Lag features
for lag in range(1, 8):
    btc_data[f'lag_{lag}'] = btc_data['Close'].shift(lag)

# Drop missing values and reset index
btc_data = btc_data.dropna().reset_index(drop=True)

# Prepare X and y
features = ['day', 'month', 'year'] + [f'lag_{lag}' for lag in range(1, 8)]
X = btc_data[features]
y = btc_data['Close']

# Sequential Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Standardize the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=6)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")
print(f"Test R^2 Score: {r2}")

# Prepare data for the next 30 days forecast
last_date = btc_data['Date'].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]

# Initialize future DataFrame
future_df = pd.DataFrame(future_dates, columns=['Date'])
future_df['day'] = future_df['Date'].dt.day
future_df['month'] = future_df['Date'].dt.month
future_df['year'] = future_df['Date'].dt.year

# Initialize lag features with the last known prices
for lag in range(1, 8):
    future_df[f'lag_{lag}'] = btc_data['Close'].iloc[-lag]

# Make predictions for the next 30 days
future_predictions = []
last_lags = btc_data['Close'].iloc[-7:].tolist()[::-1]

for i in range(len(future_df)):
    # Create the feature array for the current prediction
    future_features = np.array([future_df.iloc[i]['day'], 
                                 future_df.iloc[i]['month'], 
                                 future_df.iloc[i]['year']] + last_lags).reshape(1, -1)

    # Scale the features
    future_features_scaled = scaler.transform(future_features)

    # Predict the next day price
    next_price = model.predict(future_features_scaled)[0]
    future_predictions.append(next_price)

    # Update lag features for the next iteration
    last_lags = [next_price] + last_lags[:-1]

# Print future predictions
print("Future Predictions:")
for i, price in enumerate(future_predictions):
    print(f"Day {i + 1}: {price}")

# Plot actual vs predicted
plt.figure(figsize=(14, 7))
plt.plot(btc_data['Date'], btc_data['Close'], label='Actual Prices')
plt.plot(btc_data['Date'].iloc[train_size:], y_pred, label='Predicted Prices', color='orange')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title('Bitcoin Price Prediction using XGBoost')
plt.show()

# Plot future predictions
plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title('Future Bitcoin Price Prediction using XGBoost')
plt.show()
