#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:21:21 2019

@author: s1834310
"""

import pandas as pd
import numpy as np
import gensim.downloader as api
import gensim.models
import torch 
#import torch.nn as nn
import torch.optim as optim
import arg_extractor
import embedder
import architectures
import mynetworks



args, device = arg_extractor.get_args()
np.random.RandomState(seed = args.seed)
torch.manual_seed(seed=args.seed) #not sure why we need both and what each of them do?


#parameters
epochs = args.epochs
batch_size = args.batch_size 
output_file = args.output_filename
train_size = args.train_size
val_size = args.val_size
train_path = args.data_path + args.train_dataset_name +'.csv'
val_path = args.data_path + args.val_dataset_name +'.csv'
dataset_class = args.dataset_class
network = args.network

embedding_dim = args.embedding_dim
embedding_type = args.embedding_type
max_tokens = args.max_tokens
text_column1 = args.text_column1
text_column2 = args.text_column2
encoding = args.encoding 
D = embedding_dim*max_tokens
hidden_size = args.hidden_size
num_layers = args.num_layers
out_features = args.out_features
dropout = args.dropout
bidirectional = True

vocab_path = args.data_path + embedding_type
#vocab.save('word2vec')

#prepare word embeddings
if embedding_type == 'word2vec' or embedding_type == 'fasttext':
    
    #import the data so we can use it to expand the vocabulary 
    train = pd.read_csv(train_path, header = 0, encoding = encoding)
    val = pd.read_csv(val_path, header = 0, encoding = encoding)
    
    if embedding_type == 'word2vec':
        #vocab = api.load('word2vec-google-news-300')
        vocab = gensim.models.KeyedVectors.load(vocab_path, mmap = 'r')
    else:
        #vocab = api.load('fasttext-wiki-news-subwords-300')
        vocab = gensim.models.KeyedVectors.load(vocab_path, mmap = 'r')
    
    #embedder.expand_vocabulary(vocab, train, text_column1, text_column2, embedding_dim)
    #add padding index
    vocab.add('[PAD]', np.zeros(embedding_dim,))
    vocab.add('[UNK]', np.random.randn(embedding_dim,))
    unk_index = vocab.vocab.get('[UNK]').index
    embedding_weights = torch.FloatTensor(vocab.vectors)
    

if embedding_type == 'bert' or embedding_type == 'elmo':
    vocab = None
    unk_index = None

print('VOCAB LOAD COMPLETE')    
    
#set up the data
train_loader, val_loader = architectures.data_loaders_builder(dataset_class, batch_size, embedding_type, train_path = train_path, 
                                                              val_path = val_path, D = None, text_column1 = text_column1, 
                                                              text_column2 = text_column2, vocab = vocab, unk_index = unk_index, max_tokens = max_tokens)

print('DATA LOADER BUILDER COMPLETE') 

#create an instance of the network

if network == 'LSTMSiameseNet':
    net = eval('mynetworks.' + network + '(embedding_dim, hidden_size, num_layers, \
               out_features, dropout, bidirectional, embedding_weights, max_tokens, vocab.vocab.get(\'[PAD]\').index)')
elif network == 'BertLSTMSiameseNet' or network == 'ElmoLSTMSiameseNet':
    net = eval('mynetworks.' + network + '(embedding_dim, hidden_size, num_layers, out_features, dropout, bidirectional)')
    
net = net.to(device)

print('NETWORK INSTANCE CREATED')


#I THINK THESE TWO WILL ALWAYS BE THE SAME
#loss 
criterion = architectures.ContrastiveLoss()
criterion = criterion.to(device)

#optimizer
optimizer = optim.Adam(net.parameters())



print('TRAINING FOR: ', args.train_dataset_name, ', ', 
      str(args.dataset_class), ', ',
      str(network), ', ',
      sep = '')


architectures.train(epochs, batch_size, train_loader, val_loader, train_size, val_size, D, optimizer, net, criterion, device, output_file, args.train_dataset_name, val = True)


print('VALIDATION FOR: ', args.train_dataset_name, ', ', 
      str(args.dataset_class), ', ',
      str(network), ', ',
      sep = '')

architectures.evaluate(val_loader, D, net, device)


#does this division of things make sense?


