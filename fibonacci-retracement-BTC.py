# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 02:12:43 2024

@author: Yunus

Source Code: https://github.com/urazakgul/X-posts-python/
"""
"""
Fibonacci retracement is a popular technical analysis tool used to identify potential support and resistance levels.
"""

import yfinance as yf
import mplfinance as mpf

# https://www.investing.com/tools/fibonacci-calculator

def fibonacci_retracement(ticker, start_date, interval='1d', fibo_type='static', window=50, trend='up'):
    df = yf.download(
        tickers=ticker,
        start=start_date,
        interval=interval
    )

    price_key = 'Max Price' if trend == 'up' else 'Min Price'
    df['Min Price'] = df['Close'].rolling(window=window).min() if fibo_type == 'dynamic' else df['Close'].min()
    df['Max Price'] = df['Close'].rolling(window=window).max() if fibo_type == 'dynamic' else df['Close'].max()
    df['Difference'] = df['Max Price'] - df['Min Price']

    fibonacci_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1, 1.382]

    for i, ratio in enumerate(fibonacci_ratios, start=1):
        df[f'F{i}'] = df[price_key] - df['Difference'] * ratio if trend == 'up' else df[price_key] + df['Difference'] * ratio

    mpf.plot(df, type='candle', style='yahoo', addplot=[
        mpf.make_addplot(df[f'F{i}'] * len(df), linestyle='--', width=1.5) for i in range(1, len(fibonacci_ratios) + 1)
    ], title=f'Fibonacci Retracement for {ticker}', figsize=(16, 8))

fibonacci_retracement(
    ticker='BTC-USD',
    start_date='2023-01-01',
    fibo_type='dynamic',
    trend='up'
)
