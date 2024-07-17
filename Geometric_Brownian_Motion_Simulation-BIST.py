# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 01:57:42 2024

@author: Yunus

Source Code: https://github.com/urazakgul/X-posts-python/
"""
"""
Geometric Brownian Motion is a fundamental model in financial mathematics for simulating stock prices. 
It provides a simple yet powerful way to understand and forecast the behavior of financial instruments over time.
"""
# pip install fastdtw

import numpy as np
import pandas as pd
from fastdtw import fastdtw # for calculating Dynamic Time Warping (DTW) distance
from scipy.spatial.distance import euclidean
from isyatirimhisse import StockData
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

np.random.seed(34) # Ensures that the random numbers generated are the same every time you run the script

# mu (drift), n (number of time steps), T (total time), M (number of simulations), S0 (initial stock price), sigma (volatility).
def geometric_brownian_motion_simulation(mu, n, T, M, S0, sigma):
    time_step = T / n
    stock_prices = np.exp((mu - sigma ** 2 / 2) * time_step + sigma * np.random.normal(0, np.sqrt(time_step), size=(M, n)).T)
    stock_prices = np.vstack([np.ones(M), stock_prices])
    stock_prices = S0 * stock_prices.cumprod(axis=0)
    time_array = np.full(shape=(M, n + 1), fill_value=np.linspace(0, T, n + 1)).T

    df_simulation = pd.DataFrame(stock_prices, columns=[f'Sim_{i+1}' for i in range(M)])
    df_simulation['Time'] = time_array[:, 0]

    return df_simulation

def calculate_mu_sigma(df):
    mu = df['Daily_Return'].mean() * trading_days_per_year # annualized mean return (mu)
    sigma = df['Daily_Return'].std() * np.sqrt(trading_days_per_year) # standard deviation (sigma) of daily returns
    return mu, sigma

trading_days_per_year = 252

stock_data = StockData()

symbol = 'PGSUS'
start_date = '31-12-2023'
exchange = '0'

df = stock_data.get_data(
    symbols=symbol,
    start_date=start_date,
    exchange=exchange
)[['DATE', 'CLOSING_TL']]

df['Daily_Return'] = np.log(df['CLOSING_TL'] / df['CLOSING_TL'].shift(1))
df = df.dropna().reset_index(drop=True)

mu, sigma = calculate_mu_sigma(df)
forecast_n = 23
n = len(df) - 1 + forecast_n
T = 1
M = 100
S0 = df['CLOSING_TL'].iloc[0]

final_df = geometric_brownian_motion_simulation(mu, n, T, M, S0, sigma)

final_df['Real'] = df['CLOSING_TL']

# Calculates DTW distance between real stock prices and each simulated series.
def calculate_dtw_average(df, top_n):
    cleaned_final_df = df.dropna()
    cleaned_dtw_df = pd.DataFrame(columns=['Variable-1', 'Variable-2', 'DTW'])
    sim_columns = cleaned_final_df.columns[cleaned_final_df.columns.str.startswith('Sim')]

    for col in sim_columns:
        x = cleaned_final_df['Real'].values.reshape(-1, 1)
        y = cleaned_final_df[col].values.reshape(-1, 1)
        distance, _ = fastdtw(x, y, dist=euclidean)
        cleaned_dtw_df = pd.concat([cleaned_dtw_df, pd.DataFrame({'Variable-1': [col], 'Variable-2': ['Real'], 'DTW': [distance]})], ignore_index=True)

    cleaned_dtw_df = cleaned_dtw_df.sort_values(by='DTW')

    top_variables = cleaned_dtw_df['Variable-1'].head(top_n).tolist()
    df['DTW_Average'] = 0.0

    for index, row in final_df.iterrows():
        values = [row[var] for var in top_variables]
        df.at[index, 'DTW_Average'] = sum(values) / len(values)

    return df

result_df = calculate_dtw_average(df=final_df, top_n=5)

final_df['Sim_Avg'] = final_df.drop(columns=['Time', 'Real']).mean(axis=1)
final_df['Sim_Median'] = final_df.drop(columns=['Time', 'Real']).median(axis=1)

plt.figure(figsize=(12, 8))
# plt.plot(result_df['Time'], result_df.drop(columns='Time'), color='gray', alpha=0.5, linewidth=0.5)
plt.plot(result_df['Time'], result_df['Real'], color='red', linewidth=2, label='Real Stock Price')
plt.plot(result_df['Time'], result_df['Sim_Avg'], color='green', linewidth=1, label='Simulated Average')
plt.plot(result_df['Time'], result_df['Sim_Median'], color='orange', linewidth=1, label='Simulated Median')
plt.plot(result_df['Time'], result_df['DTW_Average'], color='blue', linewidth=3, label='DTW Average')
plt.ylabel("Stock Price $(S_t)$")
plt.title(
    rf"Geometric Brownian Motion for {symbol}: $S_t=S_0 \mathrm{{e}}^{{(\mu-\frac{{\sigma^2}}{{2}}) t+\sigma W_t}}$"
    + "\n"
    + rf"$S_0 = {S0:.2f}, \mu = {mu:.4f}, \sigma = {sigma:.4f}$"
    + "\n"
    + rf"Average Price: ${result_df['Sim_Avg'].iloc[-1]:.2f}$, "
    + rf"Median Price: ${result_df['Sim_Median'].iloc[-1]:.2f}$, "
    + rf"DTW Average Price: ${result_df['DTW_Average'].iloc[-1]:.2f}$"
    + "\n"
    + "For Educational Purposes Only"
)
plt.legend()
plt.show()