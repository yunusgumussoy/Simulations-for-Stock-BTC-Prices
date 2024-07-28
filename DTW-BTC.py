# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:35:52 2024

@author: Yunus
"""


# Dynamic Time Warping (DTW) is a technique used to measure the similarity 
# between two temporal sequences that may vary in speed or length

# pip install dtaidistance yfinance matplotlib

import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
import yfinance as yf
import datetime

# Step 1: Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2024-01-01", end=datetime.date.today()) # for computation ease
bitcoin_prices = btc_data['Adj Close'].values

# Simulate a second time series for comparison (e.g., slightly shifted)
shifted_prices = np.roll(bitcoin_prices, 5)  # Shifted version of the original prices

# Step 2: Calculate DTW distance
distance = dtw.distance(bitcoin_prices, shifted_prices)
print(f'DTW distance between the two time series: {distance}')

# Step 3: Visualize the original and shifted time series
plt.figure(figsize=(10, 6))
plt.plot(bitcoin_prices, label='Original Prices')
plt.plot(shifted_prices, label='Shifted Prices', linestyle='--')
plt.title('Dynamic Time Warping Example')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
