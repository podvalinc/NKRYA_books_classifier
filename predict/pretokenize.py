import numpy as np
import pandas as pd
import os
import math
import re
from deeppavlov.models.tokenizers.ru_sent_tokenizer import ru_sent_tokenize
from baseline_model import get_books_list

# books_paths = ['C:\\Users\\misha\\Downloads\\НКРЯ\\rusage\\txt_6',
#                'C:\\Users\\misha\\Downloads\\НКРЯ\\rusage\\txt_12',
#                'C:\\Users\\misha\\Downloads\\НКРЯ\\rusage\\txt_16',
#                'C:\\Users\\misha\\Downloads\\НКРЯ\\rusage\\txt_18']
df_books = get_books_list()

sentence_slice = 70

separator = '\\sep'

def tokenize_books(df_books):
    for index, row in df_books.iterrows():
        print('processing ' + row['filename'])
        idx = row['class']
        file_name = row['filename']
        parts = []
        with open(os.path.join(books_paths[idx], file_name), 'r', encoding='utf-8') as file:
            text = file.read()
            text = text.replace('\n', '')
            text = re.sub(' +', ' ', text)
            sentences = ru_sent_tokenize(text)
            for num in np.arange(0, len(sentences), sentence_slice):
                text_slice = ' '.join(sentences[num:num + sentence_slice])
                parts.append(text_slice)
        folder_name = 'tokenized_' + str(sentence_slice) + '_' + str(idx)
        # os.mkdir(folder_name)
        with open(os.path.join(folder_name, row['filename']), 'w',
                  encoding='utf-8') as f:
            separated_text = separator.join(parts)
            f.write(separated_text)

# tokenize_books(df_books)