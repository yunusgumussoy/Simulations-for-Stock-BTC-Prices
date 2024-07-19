# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 22:41:33 2024

@author: Yunus
"""
# pip install selenium

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import numpy as np
import time
from isyatirimhisse import StockData
import matplotlib.pyplot as plt

def get_ipo_stock_symbols(year_page_mapping, save_to_excel=True):
    stock_symbols_list = []
    driver = webdriver.Chrome()

    for year, page_count in year_page_mapping.items():
        for page in range(1, int(page_count) + 1):
            url = f"https://halkarz.com/k/halka-arz/{year}/page/{page}/"
            driver.get(url)
            try:
                h2_elements = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.il-content h2.il-bist-kod'))
                )
                for h2_element in h2_elements:
                    stock_symbols_list.append({'YEAR': year, 'SYMBOL': h2_element.text})
            except Exception as e:
                print(f"Error retrieving data for {year} page {page}: {e}")
            time.sleep(2)  # Consider using more dynamic waits instead of a static wait

    driver.quit()
    df = pd.DataFrame(stock_symbols_list)

    if save_to_excel:
        df.to_excel('ipo_list.xlsx', index=False)

    return df

year_page_mapping = {
    '2020': '1',
    '2021': '3',
    '2022': '3',
    '2023': '3',
    '2024': '1'
}

result_df = get_ipo_stock_symbols(year_page_mapping)

filtered_symbols = result_df[result_df['YEAR'] == '2024']['SYMBOL'].tolist()

stock_data = StockData()
historical_data = stock_data.get_data(
    symbols=filtered_symbols,
    start_date='01-01-2024',
    exchange='0',
    frequency='1d'
)

if not historical_data.empty:
    historical_data = historical_data[['DATE', 'CLOSING_TL', 'CODE']]
    historical_data['CLOSING_TL'] = np.log(historical_data['CLOSING_TL'])
    historical_data['DATE'] = pd.to_datetime(historical_data['DATE'])

    plt.figure(figsize=(10, 6))

    codes = historical_data['CODE'].unique()
    for code in codes:
        code_df = historical_data[historical_data['CODE'] == code]
        plt.plot(code_df['DATE'], code_df['CLOSING_TL'], label=code)

    plt.xlabel('Date')
    plt.ylabel('Log Closing TL')
    plt.title('2024 IPOs')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()  # Adjusts subplot parameters to give specified padding
    plt.show()
else:
    print("No historical data found for the specified symbols.")
