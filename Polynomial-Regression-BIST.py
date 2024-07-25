# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:02:36 2024

@author: Yunus
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Download data
data = yf.download("XU100.IS", start="2020-07-27", interval="1wk")["Adj Close"]
data = pd.DataFrame(data)
data.rename(columns={"Adj Close": "Price"}, inplace=True)

# Define x and y
x = np.arange(len(data["Price"]))
y = data["Price"]

# Fit a 2nd degree polynomial
coefficients = np.polyfit(x, y, 2)
polynomial_function = np.poly1d(coefficients)
trend = polynomial_function(x)

# Calculate R-squared and residual standard deviation
r2 = r2_score(y, trend)
residuals = y - trend
std_dev = np.std(residuals)

# Plot data and polynomial regression
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, "bo", label="Index")
plt.plot(data.index, trend, "r-", label="Trend")
plt.fill_between(data.index, trend - std_dev, trend + std_dev, color="navy", alpha=0.3, label="±1 Standard Deviation")
plt.fill_between(data.index, trend - 2 * std_dev, trend + 2 * std_dev, color="darkred", alpha=0.3, label="±2 Standard Deviations")
plt.fill_between(data.index, trend - 3 * std_dev, trend + 3 * std_dev, color="gray", alpha=0.3, label="±3 Standard Deviations")

# Title and annotations
plt.title(f"BIST 100 Polynomial Regression (R-Squared: {r2:.2f})")
plt.text(0.005, 0.8, "*Observations are weekly data", ha="left", va="top", transform=plt.gca().transAxes, fontsize=12, color="blue")
plt.legend(loc="upper left")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()
