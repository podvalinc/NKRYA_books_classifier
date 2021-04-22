import os
import requests
import math
import pandas as pd

from predict.baseline_model import get_books_list

books_paths = ['C:\\Users\\misha\\Downloads\\НКРЯ\\txt 1-4',
               'C:\\Users\\misha\\Downloads\\НКРЯ\\txt 5-9',
               'C:\\Users\\misha\\Downloads\\НКРЯ\\txt 10-11']


from tqdm import tqdm
df_books = get_books_list()
for index, row in tqdm(df_books.iterrows()):
    path = os.path.join(books_paths[row['class']], row['filename'])
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        response = requests.post("http://api.plainrussian.ru/api/1.0/ru/measure/", data={"text": text})
        try:
            readability_str = response.json()['indexes']['grade_SMOG']
            readability_index = -1
            if ('1 - 3' in readability_str):
                readability_index = 0
            if ('4 - 6' in readability_str):
                readability_index = 1
            if ('7 - 9' in readability_str):
                readability_index = 2
            if ('10 - 11' in readability_str):
                readability_index = 3
            row['readability'] = readability_index
        except:
            print('error on ' + row['filename'])
        pass

df_books.to_csv('readability_res.csv')
readability_sum = 0
for index, row in df_books.iterrows():
    if (row['class'] == 0):
        if (row['readability'] == 0):
            readability_sum += 1
        if (row['readability'] == 1):
            readability_sum += 1 / 3
    if (row['class'] == 1):
        if (row['readability'] == 1):
            readability_sum += 2 / 3
        if (row['readability'] == 2):
            readability_sum += 1
    if (row['class'] == 2 and row['readability'] == 3):
        readability_sum += 1
print ('acc')
print(readability_sum/len(df_books))
