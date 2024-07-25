# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:32:18 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data
end_date = datetime.date.today()
data = yf.download("BTC-USD", start="2015-01-01", end=end_date)

# Define new features
returns = data['Adj Close'].pct_change().rename('Return')
moving_average_5 = data['Adj Close'].rolling(window=5).mean().rename('Moving_Average_5')
moving_average_10 = data['Adj Close'].rolling(window=10).mean().rename('Moving_Average_10')
volatility = data['Adj Close'].rolling(window=5).std().rename('Volatility')

# Add new features to the DataFrame
data = data.join([returns, moving_average_5, moving_average_10, volatility])

# Drop NaNs
data = data.dropna()

# Check for infinite values and replace them
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any remaining NaN values
data = data.dropna()

# Print data info to check for any large values
print(data.describe())

# Define features and target variable
X = data[['Open', 'High', 'Low', 'Volume', 'Return', 'Moving_Average_5', 'Moving_Average_10', 'Volatility']]
y = data['Adj Close']

# Polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Check for extreme values
if np.any(np.isinf(X_poly)) or np.any(np.isnan(X_poly)):
    print("Warning: Input contains infinity or NaN values after transformation.")
else:
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Initialize and train the Polynomial Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R-Squared: {r2}')

    # Forecasting for the next 30 days
    forecast_horizon = 30
    last_known_data = data.iloc[-1]

    forecasted_data = []

    for i in range(forecast_horizon):
        # Prepare the input data
        last_data_point = pd.DataFrame(last_known_data).T
        X_forecast = last_data_point[['Open', 'High', 'Low', 'Volume', 'Return', 'Moving_Average_5', 'Moving_Average_10', 'Volatility']]
        X_forecast_poly = poly.transform(X_forecast)
        
        # Check for extreme values in forecast input
        if np.any(np.isinf(X_forecast_poly)) or np.any(np.isnan(X_forecast_poly)):
            print("Warning: Forecast input contains infinity or NaN values.")
            break
        
        # Predict the next closing price
        next_close_price = model.predict(X_forecast_poly)[0]
        
        # Append the predicted closing price to the forecasted data
        forecasted_data.append(next_close_price)
        
        # Update the last known data with the new predicted value
        new_row = {
            'Open': last_known_data['Adj Close'],
            'High': last_known_data['Adj Close'],
            'Low': last_known_data['Adj Close'],
            'Close': next_close_price,
            'Volume': last_known_data['Volume'],  # Keeping volume constant, adjust as needed
            'Return': (next_close_price - last_known_data['Adj Close']) / last_known_data['Adj Close'],
            'Moving_Average_5': (last_known_data['Moving_Average_5'] * 4 + next_close_price) / 5,
            'Moving_Average_10': (last_known_data['Moving_Average_10'] * 9 + next_close_price) / 10,
            'Volatility': np.std(data['Adj Close'].values[-4:].tolist() + [next_close_price]),
            'Adj Close': next_close_price
        }
        last_known_data = pd.Series(new_row)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices', alpha=0.7)
    plt.plot(range(len(data), len(data) + forecast_horizon), forecasted_data, label='Forecasted Prices', color='red')
    plt.title('Bitcoin Price Prediction and Forecast using Polynomial Regression')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
