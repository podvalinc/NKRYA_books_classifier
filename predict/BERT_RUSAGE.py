from transformers import BertTokenizer, BertModel
import ipywidgets
import tqdm
import pandas as pd
import torch.optim as optim
import torch
from torchtext.legacy import data
import torch.nn as nn
from pathlib import Path
import numpy as np

base_path = Path(__file__).parent
predict_path = (base_path / "../predict").resolve()

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
max_input_length = 512

labels = ['high', 'middle', 'adult', 'low']

HIDDEN_DIM = 256
OUTPUT_DIM = 4
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25


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


model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

model.load_state_dict(torch.load(predict_path / 'readability_with_big_data_4_categ.pt'))
model.eval()
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
    if (age == 'low'):
        return '6+'
    elif (age == 'middle'):
        return '12+'
    elif (age == 'high'):
        return '16+'
    else:
        return '18+'


def predict_rusage_class(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))

    del tensor
    del tokens
    del indexed

    max_preds_idx = prediction.argmax(dim=1)
    prediction = prediction.cpu().detach().numpy()
    probability = prediction[0][max_preds_idx] / np.sum(prediction)
    del prediction
    # print(max_preds_idx)
    torch.cuda.empty_cache()
    real_pred = labels[max_preds_idx]
    pred = correct_answer(real_pred)
    return pred, probability
