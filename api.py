import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from fastapi import FastAPI

# initialize app
app = FastAPI()

# initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# initailize model
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id,
                                     attention_mask=mask,
                                     return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

@app.on_event('startup')
async def startup():
    model = BertClassifier()
    model.load_state_dict(torch.load('bert_cased_cpu.pt'))

    app.package = {
        'model': model,
    }

@app.get('/{phrase}')
async def predict(phrase: str):
    with torch.no_grad():
        tokenized = tokenizer(phrase, padding='max_length',
                              max_length=25, truncation=True,
                              return_tensors="pt")
        output = app.package['model'](tokenized['input_ids'], tokenized['attention_mask'])
        predicted_score = output.argmax(dim=1).numpy()[0]
        return f'predicted book review score {predicted_score}'