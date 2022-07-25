from transformers import BertTokenizer, BertModel
import ipywidgets
import tqdm
import pandas as pd
import torch.optim as optim
import torch
from torchtext.legacy import data
import torch.nn as nn

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

d = ['high', 'middle', 'adult', 'low']

HIDDEN_DIM = 256
OUTPUT_DIM = 4
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

model.load_state_dict(torch.load('readability_with_big_data_4_categ.pt'))
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


def predict_sentiment(model, tokenizer, sentence):
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
    del prediction
    # print(max_preds_idx)
    torch.cuda.empty_cache()
    real_pred = d[max_preds_idx]
    pred = correct_answer(real_pred)
    return pred


