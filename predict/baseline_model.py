# from preprocess.druzhkin import DruzhkinAnalyzer
# from preprocess.druzhkin import SyntaxFeatures
import pandas as pd
import numpy as np
import os
import logging
import re
# from deeppavlov.models.tokenizers.ru_sent_tokenizer import ru_sent_tokenize
from scipy.stats.mstats import gmean, hmean
import math

books_paths = ['C:\\Users\\misha\\Downloads\\НКРЯ\\txt 1-4',
               'C:\\Users\\misha\\Downloads\\НКРЯ\\txt 5-9',
               'C:\\Users\\misha\\Downloads\\НКРЯ\\txt 10-11']

sentence_slice = 70


def get_books_list():
    df_books = pd.DataFrame(columns=['filename', 'class'])
    for idx in range(len(books_paths)):
        for file_name in os.listdir(books_paths[idx]):
            append_data = dict(zip(df_books.columns, [file_name, idx]))
            df_books = df_books.append(append_data, ignore_index=True)
    return df_books


def get_books_by_percent(df, perc):
    new_df = pd.DataFrame(columns=['filename', 'class'])
    new_df = new_df.append(df[df['class'] == 0].sample(math.ceil(len(df[df['class'] == 0]) * perc)))
    new_df = new_df.append(df[df['class'] == 1].sample(math.ceil(len(df[df['class'] == 1]) * perc)))
    new_df = new_df.append(df[df['class'] == 2].sample(math.ceil(len(df[df['class'] == 2]) * perc)))
    return new_df


folder_name = 'tokenized_70_'
separator = '\\sep'


def prepare_books_dataframe(df_books):
    df = pd.DataFrame(columns=['bookname', 'metadata', 'text', 'class'])

    for index, row in df_books.iterrows():
        idx = row['class']
        file_name = row['filename']

        with open(os.path.join(folder_name + str(idx), file_name), 'r', encoding='utf-8') as text_file:
            text = text_file.read()
            parts = text.split(separator)
            metadata = np.loadtxt('analyzed_' + str(idx) + '/' + file_name)
            metadata = np.reshape(metadata, (-1, 147))
            for i in range(len(parts)):
                append_data = dict(zip(df.columns, [file_name, metadata[i], parts[i], idx]))
                df = df.append(append_data, ignore_index=True)

    df.fillna(0)
    return df


# def prepare_books_dataframe_old(df_books):
#     logger = logging.getLogger('my-logger')
#     logger.propagate = False
#
#     df = pd.DataFrame(columns=['bookname', 'text', 'class'])
#     for index, row in df_books.iterrows():
#         idx = row['class']
#         file_name = row['filename']
#         # print("processing idx= " + str(idx) + ' ' + file_name)
#         with open(os.path.join(books_paths[idx], file_name), 'r', encoding='utf-8') as file:
#             text = file.read()
#             text = text.replace('\n', '')
#             text = re.sub(' +', ' ', text)
#             sentences = ru_sent_tokenize(text)
#             for num in np.arange(0, len(sentences), sentence_slice):
#                 text_slice = ' '.join(sentences[num:num + sentence_slice])
#                 append_data = dict(zip(df.columns, [file_name, text_slice, idx]))
#                 try:
#                     # print(append_data)
#                     df = df.append(append_data, ignore_index=True)
#                 except:
#                     print('ERROR on ' + file_name)
#
#     logger.propagate = True
#     df.fillna(0)
#     return df


if __name__ == '__main__':
    df_books = get_books_list()

    df_books.to_csv('books_list.csv')
    print("TOTAL ALL BOOKS")
    print(df_books['class'].value_counts())

    valid_books = get_books_by_percent(df_books, 0.15)
    df_books = df_books.drop(valid_books.index)
    print("Val")
    print(valid_books['class'].value_counts(sort=False))

    test_books = get_books_by_percent(df_books, 0.15)
    df_books = df_books.drop(test_books.index)
    print("test")
    print(test_books['class'].value_counts(sort=False))

    print("train")
    print(df_books['class'].value_counts(sort=False))

    train_books = df_books
    train = prepare_books_dataframe(train_books)
    train.to_csv('train_texts_70.csv')

    test = prepare_books_dataframe(test_books)
    test.to_csv('test_texts_70.csv')

    valid = prepare_books_dataframe(valid_books)
    train.to_csv('valid_texts_70.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#
# def train(df):
#     cols = list(df.columns)
#     cols.remove('Unnamed: 0')
#     # cols.remove('Unnamed: 0.1')
#     cols.remove('label')
#     x = np.array(df[cols])
#     # print(x)
#
#     for x_val in x:
#         x_val[np.isnan(x_val)] = 0
#     y = np.array(df['label'])
#     # print(x)
#     # print(y)
#     # reg = LogisticRegression(solver='saga').fit(x, y)
#     reg = RandomForestClassifier().fit(x, y)
#     # reg = QuadraticDiscriminantAnalysis().fit(x, y)
#     return reg

#
# def make_prediction(predictor, filename):
#     with open('C:\\Users\\misha\\Downloads\\НКРЯ\\' + filename, 'r', encoding='utf-8') as input_file:
#         text = input_file.read()
#         text = text.replace('\n', '')
#         text = re.sub(' +', ' ', text)
#         sentences = ru_sent_tokenize(text)
#         x_vals = []
#         for num in np.arange(0, len(sentences), sentence_slice):
#             text_to_analyze = ' '.join(sentences[num:num + sentence_slice])
#             try:
#                 data = get_text_analysis('test', 0, text_to_analyze)
#                 data = data.drop('label')
#                 x = np.array(data)
#                 x[np.isnan(x)] = 0
#                 x_vals.append(x)
#             except:
#                 print('ERROR on ' + filename)
#         res = predictor.predict(x_vals)
#         print(np.mean(res), gmean(res), hmean(res))
#
#
# csv = pd.read_csv('train150.csv')
# predictor = train(csv)
# for filename in os.listdir('C:\\Users\\misha\\Downloads\\НКРЯ'):
#     if (filename.endswith('.txt')):
#         print(filename)
#         make_prediction(predictor, filename)

#
# def get_text_analysis(file_name, idx, text):
#     analyzer = DruzhkinAnalyzer()
#     indexes = ['label'] + analyzer.grammems + analyzer.word_ends + analyzer.tops + ['max_sent_len', 'avg_sent_len']
#
#     dr_grammems, dr_ends, dr_tops = analyzer.analyze(text)
#     sent_lengths = SyntaxFeatures().get_sentence_lengths(text)
#     # punctuation_count = SyntaxFeatures().get_punctuation_count(text)
#     data = {'label': idx, **dr_grammems, **dr_ends, **dr_tops, 'max_sent_len': max(sent_lengths),
#             'avg_sent_len': np.average(sent_lengths)}
#     data = pd.Series(data, name=file_name, index=indexes)
#     data.fillna(0)
#     return data
