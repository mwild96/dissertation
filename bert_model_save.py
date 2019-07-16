#!/usr/bin/env python3

import torch
import pytorch_pretrained_bert as bert

bert_tokenizer = bert.BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = bert.BertModel.from_pretrained('bert-base-uncased')

bert_tokenizer.save_vocabulary('bert_tokenizer')#I'M NOT SURE WHAT THE EXTENSION ON THIS FILE SHOULD BE
torch.save({'model_state_dict': bert_tokenizer.state_dict()}, 'bert_model.pth')


#retrieve with:
#load_vocab()

#and also we already know how to reload a model but I'd have to double check it