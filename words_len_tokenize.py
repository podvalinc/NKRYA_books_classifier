import numpy as np
import pandas as pd
import os
import math
import re
from deeppavlov.models.tokenizers.ru_sent_tokenizer import ru_sent_tokenize
from tqdm.notebook import tqdm

from predict.baseline_model import get_books_list
from preprocess.tokenize_1 import Tokenizer

books_paths = ['txt 1-4',
               'txt 5-9',
               'txt 10-11']
df_books = get_books_list()

word_slice = 600

separator = '\\sep '


def tokenize_books(df_books):
    for index, row in tqdm(df_books.iterrows()):

        # print('processing ' + row['filename'])
        idx = row['class']
        # if index < 123:
        #   continue
        file_name = row['filename']
        parts = []
        folder_name = 'tokenized_' + str(word_slice) + '_' + str(idx)

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        with open(os.path.join(folder_name, row['filename']), 'w', encoding='utf-8') as f:
            pass

        with open(os.path.join(books_paths[idx], file_name), 'r', encoding='utf-8') as input_file, \
                open(os.path.join(folder_name, row['filename']), 'w+', encoding='utf-8') as out_file:
            text = input_file.read()
            text = text.replace('\n', '')
            text = re.sub(' +', ' ', text)
            sentences = ru_sent_tokenize(text)
            tokenizer = Tokenizer()
            sum_len = 0
            # parts = []
            for sent in (sentences):
                cur_len = len(tokenizer.tokenize(sent, remove_punctuation=False))
                if (sum_len + cur_len > word_slice):
                    sum_len = 0
                    out_file.write(separator)
                sum_len+=cur_len
                out_file.write(sent + ' ')


tokenize_books(df_books)
