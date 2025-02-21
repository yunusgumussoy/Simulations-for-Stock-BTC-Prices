# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:05:22 2024

@author: Yunus
"""
# pip install yahooquery

import pandas as pd
from yahooquery import Ticker
import requests
from io import StringIO

# Bist100 Hisse kodlarını alıp, yahoo finance için uygun hale getirir
def hisseler():
    url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx?endeks=01#page-1"
    html_text = requests.get(url).text
    html_io = StringIO(html_text)
    tablo = pd.read_html(html_io)[2]["Kod"]
    for i in range(len(tablo)):
        tablo[i] += ".IS"
    hissekod = tablo.to_list()
    return hissekod

# Hisselerle ilgili gerekli bilgileri alır ve dataframe olarak yazdırır
def hedef_fiyat():
    hisse = Ticker(hisseler())
    hisse_dict = hisse.financial_data
    df = pd.DataFrame.from_dict(hisse_dict, orient="index").iloc[:, 1:6].reset_index()
    df.columns = ["Hisse Adı", "Güncel Fiyat", "En Yüksek Tahmin", "En Düşük Tahmin",
                  "Ortalama Tahmin", "Medyan Tahmin"]
    df["Hisse Adı"] = df["Hisse Adı"].str.replace(".IS", "", regex=False)
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

# Fetch the data
hedef_fiyat_df = hedef_fiyat()

# Print the DataFrame
print(hedef_fiyat_df)

# Save the DataFrame to a CSV file
hedef_fiyat_df.to_csv('YF_hedef_fiyat.csv', index=False)
