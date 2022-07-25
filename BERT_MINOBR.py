from transformers import BertTokenizer, BertModel
import ipywidgets
import tqdm
import pandas as pd
import torch.optim as optim
import torch
from torchtext.legacy import data
import torch.nn as nn

import torch.nn as nn


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output


tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

HIDDEN_DIM = 32
OUTPUT_DIM = 3
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens


def correct_answer(age):
    if (age == '0'):
        return '1-4 классы'
    elif (age == '1'):
        return '5-9 классы'
    else:
        return '10-11 классы'


def make_good_arr(array):
    res_small1 = np.array([array[1][0], array[1][2], array[1][1]])
    res_small2 = np.array([array[3][1], array[3][0], array[3][2]])
    result = np.array([array[0], res_small1, array[2], res_small2, array[4]])
    return result

    return array


def predict_sentiment(model, tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    preds = torch.sigmoid(model(tensor))

    del tensor
    del tokens
    del indexed

    torch.cuda.empty_cache()

    preds = preds.cpu().detach().numpy()

    return preds[0]


def Kfold_predict(text, number_models):
    models_predicts = []
    for i in range(number_models):
        model.load_state_dict(torch.load('readability_minobr' + str(i) + '_model.pt'))

        model_prediction = predict_sentiment(model, tokenizer, text)
        models_predicts.append(model_prediction)
    models_predicts = make_good_arr(models_predicts)
    final_pred = np.mean(np.array(models_predicts), axis=0)
    max_preds = final_pred.argmax()
    category = correct_answer(d0[max_preds])
    return category

# a =  'Никто не мог сказать этого наверное, но многие догадывались, что молчаливый пан Попельский пленился панной Яценко именно в ту короткую четверть часа, когда она исполняла трудную пьесу.'
# Kfold_predict(a, 5)