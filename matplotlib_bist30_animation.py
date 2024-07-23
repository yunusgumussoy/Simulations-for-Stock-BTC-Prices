# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:50:39 2024

@author: Yunus

Source Code: https://github.com/urazakgul/X-posts-python/
"""
# pip install isyatirimhisse
# https://pypi.org/project/isyatirimhisse/
# pip install ipympl

"""
Interactive plots run well in Jupyter. For Spyder, go to Tools>Preferences>IPython console>Graphics>Backend>Automatic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib.animation import FuncAnimation, FFMpegWriter
from isyatirimhisse import StockData

# %matplotlib widget 
# %matplotlib qt5

stock_data = StockData()

# https://www.kap.org.tr/tr/Endeksler
symbols = [
    'AKBNK', 'ALARK', 'ARCLK', 'ASELS', 'ASTOR',
    'BIMAS', 'EKGYO', 'ENKAI', 'EREGL', 'FROTO',
    'GUBRF', 'SAHOL', 'HEKTS', 'KRDMD', 'KCHOL',
    'KONTR', 'KOZAL', 'ODAS', 'OYAKC', 'PGSUS',
    'PETKM', 'SASA', 'TOASO', 'TCELL', 'TUPRS',
    'THYAO', 'GARAN', 'ISCTR', 'SISE', 'YKBNK'
]

df = stock_data.get_data(
    symbols=symbols,
    start_date='31-12-2018',
    exchange='0', # TRY
    save_to_excel=True
)

df = df[['DATE','CLOSING_TL','XU100_TL','CODE']]

df2_pivot = df.pivot_table(
    index=['DATE','XU100_TL'],
    columns='CODE',
    values='CLOSING_TL'
).reset_index().rename(columns={'XU100_TL':'XU100'})

df2_pivot['DATE'] = pd.to_datetime(df2_pivot['DATE'])
df2_pivot = df2_pivot.set_index('DATE')

df_pct_change = df2_pivot.pct_change().resample('M').agg(lambda x: (x + 1).prod() - 1)
df_returns = df2_pivot.pct_change()
df_std = df_returns.resample('M').std()

data_list = []

unique_dates = df_pct_change.index.get_level_values('DATE').unique()

symbols.append('XU100')
for date in unique_dates:
    for symbol in symbols:
        return_val = df_pct_change.loc[date, symbol]
        sd_val = df_std.loc[date, symbol]

        data_list.append({
            'DATE': date,
            'STOCK': symbol,
            'RETURN': return_val,
            'SD': sd_val
        })

result_df = pd.DataFrame(data_list)
result_df['DATE'] = pd.to_datetime(result_df['DATE'])
result_df = result_df.dropna()

fig, ax = plt.subplots(figsize=(12, 8))

scatter_xu100 = ax.scatter([], [], c='red', alpha=0.5)
scatter_stocks = ax.scatter([], [], c='blue', alpha=0.5)

ax.set_xlim(result_df['SD'].min(), result_df['SD'].max())
ax.set_ylim(result_df['RETURN'].min(), result_df['RETURN'].max())

ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Return')
ax.set_title('BIST30 Stocks Return vs Standard Deviation, ')

def update(frame):
    data_xu100 = result_df[(result_df['DATE'] == frame) & (result_df['STOCK'] == 'XU100')]
    data_stocks = result_df[(result_df['DATE'] == frame) & (result_df['STOCK'] != 'XU100')]

    scatter_xu100.set_offsets(data_xu100[['SD', 'RETURN']])
    scatter_stocks.set_offsets(data_stocks[['SD', 'RETURN']])

    x_min = min(data_xu100['SD'].min(), data_stocks['SD'].min()) - 0.03
    x_max = max(data_xu100['SD'].max(), data_stocks['SD'].max()) + 0.03
    y_min = min(data_xu100['RETURN'].min(), data_stocks['RETURN'].min()) - 0.03
    y_max = max(data_xu100['RETURN'].max(), data_stocks['RETURN'].max()) + 0.03

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for annotation in ax.texts:
        annotation.remove()

    for _, point in data_xu100.iterrows():
        ax.annotate(point['STOCK'], (point['SD'], point['RETURN']), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='red')

    for _, point in data_stocks.iterrows():
        ax.annotate(point['STOCK'], (point['SD'], point['RETURN']), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='blue')

    ax.set_title(f'BIST30 Stocks Return vs Standard Deviation, {frame.strftime("%B-%Y")}')

def init():
    return scatter_xu100, scatter_stocks

anim = FuncAnimation(fig, update, frames=unique_dates, init_func=init, repeat=True, interval=2000)

plt.show()

# Resolved the NaN issue during saving
# However, the dynamism of axis limits is compromised

fig, ax = plt.subplots(figsize=(12, 8))

scatter_xu100 = ax.scatter([], [], c='red', alpha=0.5)
scatter_stocks = ax.scatter([], [], c='blue', alpha=0.5)

ax.set_xlim(result_df['SD'].min(), result_df['SD'].max())
ax.set_ylim(result_df['RETURN'].min(), result_df['RETURN'].max())

ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Return')
ax.set_title('BIST30 Stocks Return vs Standard Deviation, ')

def update(frame):
    data_xu100 = result_df[(result_df['DATE'] == frame) & (result_df['STOCK'] == 'XU100')]
    data_stocks = result_df[(result_df['DATE'] == frame) & (result_df['STOCK'] != 'XU100')]

    scatter_xu100.set_offsets(data_xu100[['SD', 'RETURN']])
    scatter_stocks.set_offsets(data_stocks[['SD', 'RETURN']])

    x_min = np.nanmin([data_xu100['SD'].min(), data_stocks['SD'].min(), result_df['SD'].min()]) - 0.03
    x_max = np.nanmax([data_xu100['SD'].max(), data_stocks['SD'].max(), result_df['SD'].max()]) + 0.03
    y_min = np.nanmin([data_xu100['RETURN'].min(), data_stocks['RETURN'].min(), result_df['RETURN'].min()]) - 0.03
    y_max = np.nanmax([data_xu100['RETURN'].max(), data_stocks['RETURN'].max(), result_df['RETURN'].max()]) + 0.03

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for annotation in ax.texts:
        annotation.remove()

    for _, point in data_xu100.iterrows():
        ax.annotate(point['STOCK'], (point['SD'], point['RETURN']), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='red')

    for _, point in data_stocks.iterrows():
        ax.annotate(point['STOCK'], (point['SD'], point['RETURN']), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='blue')

    ax.set_title(f'BIST30 Stocks Return vs Standard Deviation, {frame.strftime("%B-%Y")}')

def init():
    return scatter_xu100, scatter_stocks

anim = FuncAnimation(fig, update, frames=unique_dates, init_func=init, repeat=True, interval=2000)

# https://www.gyan.dev/ffmpeg/builds/
# Download: release builds / ffmpeg-release-essentials.zip

plt.rcParams["animation.ffmpeg_path"] = "C:\\Users\\yunus\\Downloads\\ffmpeg-6.1-essentials_build\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe"
writer = FFMpegWriter(fps=59, metadata=dict(artist = "Yunus"), bitrate=1800)
anim.save("return_sd_saved.mp4")

plt.show()