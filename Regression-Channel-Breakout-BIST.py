# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:48:15 2024

@author: Yunus
"""


import numpy as np
import pandas as pd
import requests
from scipy import stats
import ssl
from urllib import request

def Hisse_Temel_Veriler():
    url1 = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx#page-1"
    context = ssl._create_unverified_context()
    response = request.urlopen(url1, context=context)
    url1 = response.read()
    df = pd.read_html(url1, decimal=',', thousands='.')
    df = df[6]
    Hisseler = df['Kod'].values.tolist()
    return Hisseler

def Stock_Prices(Hisse):
    Bar = 1000
    url = f"https://www.isyatirim.com.tr/_Layouts/15/IsYatirim.Website/Common/ChartData.aspx/IntradayDelay?period=120&code={Hisse}.E.BIST&last={Bar}"
    r1 = requests.get(url).json()
    data = pd.DataFrame.from_dict(r1)
    data[['Volume', 'Close']] = pd.DataFrame(data['data'].tolist(), index=data.index)
    data.drop(columns=['data'], inplace=True)
    return data

def Trend_Channel(df):
    best_period = None
    best_r_value = 0
    periods = range(100, 201, 10)
    for period in periods:
        close_data = df['Close'].tail(period)
        x = np.arange(len(close_data))
        slope, intercept, r_value, _, _ = stats.linregress(x, close_data)
        if abs(r_value) > abs(best_r_value):
            best_r_value = abs(r_value)
            best_period = period

    return best_period, best_r_value

def List_Trend_Breaks(Hisse, data, best_period, rval=0.85):
    close_data = data['Close'].tail(best_period)
    x_best_period = np.arange(len(close_data))
    slope_best_period, intercept_best_period, r_value_best_period, _, _ = stats.linregress(x_best_period, close_data)
    trendline = slope_best_period * x_best_period + intercept_best_period
    upper_channel = trendline + (trendline.std() * 1.1)
    lower_channel = trendline - (trendline.std() * 1.1)

    upper_diff = upper_channel - close_data
    lower_diff = close_data - lower_channel
    last_upper_diff = upper_diff.iloc[-1]
    last_lower_diff = lower_diff.iloc[-1]

    if abs(r_value_best_period) > rval:
        if last_upper_diff < 0:
            return f'{Hisse}: Yukarı yönlü kırılım (R={abs(r_value_best_period):.2f}, Fark={last_upper_diff:.2f})', True, 'up'
        elif last_lower_diff < 0:
            return f'{Hisse}: Aşağı yönlü kırılım (R={abs(r_value_best_period):.2f}, Fark={last_lower_diff:.2f})', True, 'down'
    return None, False, None

Hisseler = Hisse_Temel_Veriler()
up_breaks = []
down_breaks = []

for hisse in Hisseler:
    try:
        data = Stock_Prices(hisse)
        best_period, best_r_value = Trend_Channel(data)
        result, status, direction = List_Trend_Breaks(hisse, data, best_period)
        if result:
            if direction == 'up':
                up_breaks.append(result)
            elif direction == 'down':
                down_breaks.append(result)
        print(f'{hisse} kontrol ediliyor: {status}')
    except Exception as e:
        print(f'Hisse {hisse} için hata: {e}')
        continue

# Sonuçları yazdır
if up_breaks:
    print("\nYukarı yönlü kırılımlar:")
    for break_info in up_breaks:
        print(break_info)
else:
    print("\nYukarı yönlü kırılım tespit edilmedi.")

if down_breaks:
    print("\nAşağı yönlü kırılımlar:")
    for break_info in down_breaks:
        print(break_info)
else:
    print("\nAşağı yönlü kırılım tespit edilmedi.")