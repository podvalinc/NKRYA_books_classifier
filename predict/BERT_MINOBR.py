from transformers import BertTokenizer, BertModel
import ipywidgets
import tqdm
import pandas as pd
import torch.optim as optim
import torch
from torchtext.legacy import data
import torch.nn as nn
import numpy as np
import torch.nn as nn

from pathlib import Path

from predict.BERT_RUSAGE import max_input_length

base_path = Path(__file__).parent
predict_path = (base_path / "../predict").resolve()


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
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        output = self.out(hidden)
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


def predict_for_model(model, tokenizer, sentence):
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
    d = ['2', '1', '0']
    models_predicts = []
    for i in range(number_models):
        model.load_state_dict(
            torch.load(predict_path / 'minobr_models' / ('readability_minobr' + str(i) + '_model.pt')))

        model_prediction = predict_for_model(model, tokenizer, text)
        models_predicts.append(model_prediction)

    final_pred = np.mean(np.array(models_predicts), axis=0)
    max_preds = final_pred.argmax()
    category = correct_answer(d[max_preds])
    return category, final_pred[max_preds] / np.sum(final_pred)

# a = 'Никто не мог сказать этого наверное, но многие догадывались, что молчаливый пан Попельский пленился панной Яценко именно в ту короткую четверть часа, когда она исполняла трудную пьесу.'
# Kfold_predict(a, 5)
