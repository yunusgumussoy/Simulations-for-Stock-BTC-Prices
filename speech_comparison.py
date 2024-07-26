# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 01:13:48 2024

@author: Yunus

Source Code: https://github.com/urazakgul/X-posts-python/
"""


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import requests

def compare_cb_texts(URL1, URL2, exc_first=4, exc_last=1):
    def fetch_text_from_url(url, exc_first, exc_last):
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'lxml')
        content_div = soup.find('div', class_='tcmb-content') # Finds a specific <div> with the class tcmb-content.
        if content_div:
            tum_div_p = content_div.find_all('p') # Extracts text from all <p> tags within this <div>.
            alt_dizi = [div_p.get_text() for div_p in tum_div_p]
            return ' '.join(alt_dizi[exc_first:-exc_last])
        return ''

    texts = [fetch_text_from_url(URL1, exc_first, exc_last), fetch_text_from_url(URL2, exc_first, exc_last)]

# Split the texts into sentences based on punctuation (., !, ?)
    sentences_text1 = re.split(r'(?<=[.!?])\s+', texts[0])
    sentences_text2 = re.split(r'(?<=[.!?])\s+', texts[1])

    columns = ['TEXT-1', 'TEXT-2', 'SCORE', 'DIFF_WORDS']
    cosine_similarity_df = pd.DataFrame(columns=columns)
    vectorizer = CountVectorizer()

    for sentence_text1 in sentences_text1:
        for sentence_text2 in sentences_text2:
            if sentence_text1 and sentence_text2:
                vectors = vectorizer.fit_transform([sentence_text1, sentence_text2])
                cosine_sim = cosine_similarity(vectors)[0, 1]

                words_text1 = set(sentence_text1.split())
                words_text2 = set(sentence_text2.split())
                diff_words = ', '.join(words_text1.symmetric_difference(words_text2))
                diff_words = diff_words.translate(str.maketrans("", "", string.punctuation)).replace(" ", ", ")

                df_to_append = pd.DataFrame({
                    'TEXT-1': [sentence_text1.strip()],
                    'TEXT-2': [sentence_text2.strip()],
                    'SCORE': [cosine_sim],
                    'DIFF_WORDS': [diff_words]
                })
                
                if not df_to_append.empty:
                    cosine_similarity_df = pd.concat([cosine_similarity_df, df_to_append], ignore_index=True)

    groups = cosine_similarity_df.groupby('TEXT-1', sort=False)
    selected_rows = groups.apply(lambda group: group.loc[group['SCORE'].idxmax()])
    selected_rows.reset_index(drop=True, inplace=True)

    return selected_rows

URL1 = 'https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/2023/DUY2023-45'
URL2 = 'https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/2023/DUY2023-51'

result_df = compare_cb_texts(URL1, URL2)
result_df.to_excel('diff_words.xlsx', index=False)
result_df