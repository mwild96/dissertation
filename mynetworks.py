#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_transformers as bert
import allennlp.modules.elmo as elmo


class HighwayReluNet(nn.Module):
    '''
    This class defines a two layer fully connected ReLU HighwayNet. The architecture is taken from the paper "Deep Learning for 
    Entity Matching: A Design Space Exploration"
    '''
    def __init__(self, input_size):
        super(HighwayReluNet, self).__init__()
        self.input_size = input_size
        self.plain1 = nn.Linear(self.input_size, self.input_size)
        self.plain2 = nn.Linear(self.input_size,self.input_size)
        self.plain3 = nn.Linear(self.input_size,1)
        
        self.gate1 = nn.Linear(self.input_size,self.input_size) 
        self.gate2 = nn.Linear(self.input_size,self.input_size)


    def forward(self,x):
        '''
        general layout of a 2-layer mlp:
        h1 = a(W1@X +b1)
        h2 = a(W2@h1 + b2)
        y_hat = sigmoid(W3@h2 + b3
        (is this right...)
        
        with highway we use:
        O = H*T + (1-T)x
        
        '''
        
        h1 = F.relu(self.plain1(x)) 
        t1 = torch.sigmoid(self.gate1(x))
        o1 = torch.add(h1*t1, (1.0-t1)*x)
        
        h2 = F.relu(self.plain2(o1))
        t2 = torch.sigmoid(self.gate2(o1))
        o2 = torch.add(h2*t2, (1.0-t2)*o1)
        
        return torch.sigmoid(self.plain3(o2))
    
        
        
class LSTMSiameseNet(nn.Module):
    '''
    Adapated from the paper "Learning Text Similarity with Siamese Recurrent Networks"
    
    "The proposed network contains four layers of Bidirectional LSTM nodes. The activations at each timestep of the final 
    BLSTM layer are averaged to produce a fixed dimensional output. This output is projected through a single densely 
    connected feedforward layer"
    
    Hyperparameters were chosen according to this paper:
    
    hidden_size = 64
    num_layers = 4
    out_features = 128 (?)
    bidirectional = True
    dropout = 0.2
    
    '''
    
    
    def __init__(self, embedding_dim, hidden_size, number_layers, out_features, dropout, bidirectional, embeddings, max_tokens, padding_idx):
        super(LSTMSiameseNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_features = out_features
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.max_tokens = max_tokens
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx = self.padding_idx)
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size,
                            num_layers = self.number_layers, dropout = self.dropout, bidirectional = self.bidirectional, batch_first = True)

        self.fc = nn.Linear(in_features = self.hidden_size*2, out_features = self.out_features)
    
    def forward_once(self, x):
        x = self.embedding(x)
        x, (hidden, cell) = self.lstm(x)
        x = torch.mean(x, dim = 1)
        x = F.relu(self.fc(x))
        return x
    
    def forward(self ,x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return torch.clamp(F.cosine_similarity(x1, x2).reshape(-1,1), min=0.0, max = 1.0)
        #alternatively: 
        #return F.sigmoid(F.pairwise_distance(x1,x2).reshape(-1,1))
    
    
    
class BertLSTMSiameseNet(nn.Module):
    '''
    Adapated from the paper "Learning Text Similarity with Siamese Recurrent Networks"
    
    "The propose network contains four layers of Bidirectional LSTM nodes. The activations at each timestep of the final 
    BLSTM layer are averaged to produce a fixed dimensional output. This output is projected through a single densely 
    connected feedforward layer"
    
    Hyperparameters were chosen according to this paper:
    
    hidden_size = 64
    num_layers = 4
    out_features = 128?????? <-I'm really unclear about this part from the paper
    bidirectional = True
    dropout = 0.2?
    
    according to the BERT paper: "the best performing method is to concatenate the token representations 
    from the top four hidden layers of the pre-trained Transformer"
    '''
    
    
    def __init__(self, embedding_dim, hidden_size, number_layers, out_features, dropout, bidirectional):
        
        super(BertLSTMSiameseNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_features = out_features
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.bert = bert.BertModel.from_pretrained('bert-base-uncased',
                                                   output_hidden_states=True, output_attentions=True)
        #self.bert = bert.BertModel.from_pretrained('/home/s1834310/Dissertation/PretrainedBert',
        #                                           output_hidden_states=True, output_attentions=True)
        
        for param in self.bert.parameters():
            param.requires_grad = False
                
        self.lstm = nn.LSTM(input_size = 4*self.embedding_dim, hidden_size = self.hidden_size,
                            num_layers = self.number_layers, dropout = self.dropout, bidirectional = self.bidirectional, batch_first = True)
        self.fc = nn.Linear(in_features = self.hidden_size*2, out_features = self.out_features)
        #for sequence summary token (CLS)only ^^ use in_features = 4*768
        
    def forward_once(self, tokens, segments, input_mask):
        self.bert.eval()
        with torch.no_grad():
            x, _ = self.bert(tokens, segments, input_mask)[-2:]
        x = torch.stack(x).squeeze()[8:12,:,:,:].reshape(tokens.shape[0],-1,4*self.embedding_dim).squeeze() 
        
        #alternative: use the sequence summary token (CLS) only:
        #x = torch.stack(x).squeeze()[8:12,:,0,:].reshape(tokens.shape[0],-1, 4*self.embedding_dim)
        #DO NOT USE LSTM OR MEAN WHEN WE USE CLS ONLY
        x, (hidden, cell) = self.lstm(x)
        x = torch.mean(x, dim = 1) 
        x = F.relu(self.fc(x))
        return x
    
    def forward(self, tokens1, segments1, input_mask1, tokens2, segments2, input_mask2):
        x1 = self.forward_once(tokens1, segments1, input_mask1)
        x2 = self.forward_once(tokens2, segments2, input_mask2)
        return torch.clamp(F.cosine_similarity(x1, x2).reshape(-1,1), min = 0.0, max = 1.0)
        #alternatively: 
        #return F.sigmoid(F.pairwise_distance(x1,x2).reshape(-1,1))
    
    
    
    
    
    
class ElmoLSTMSiameseNet(nn.Module):
    '''
    Adapated from the paper "Learning Text Similarity with Siamese Recurrent Networks"
    
    "The propose network contains four layers of Bidirectional LSTM nodes. The activations at each timestep of the final 
    BLSTM layer are averaged to produce a fixed dimensional output. This output is projected through a single densely 
    connected feedforward layer"
    
    Hyperparameters were chosen according to this paper:
    
    hidden_size = 64
    num_layers = 4
    out_features = 128 (?)
    bidirectional = True
    dropout = 0.2
    '''
    
    
    def __init__(self, embedding_dim, hidden_size, number_layers, out_features, dropout, bidirectional):#, embeddings, max_tokens, padding_idx
        super(ElmoLSTMSiameseNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_features = out_features
        self.dropout = dropout
        self.bidirectional = bidirectional
        #WE'RE NOT GOING TO BE ABLE TO ACCESS THIS ON THE CLUSTER...
        #self.options_file = '/home/s1834310/Dissertation/ELMo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        #self.weight_file = '/home/s1834310/Dissertation/ELMo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        self.options_file = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        self.weight_file = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        self.elmo = elmo.Elmo(self.options_file, self.weight_file, 1, dropout = 0)
        
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size,
                            num_layers = self.number_layers, dropout = self.dropout, bidirectional = self.bidirectional, batch_first = True)
        self.fc = nn.Linear(in_features = 2*self.hidden_size, out_features = self.out_features)
    
    def forward_once(self, x):
        x = self.elmo(x)
        x = x['elmo_representations'][0]  
        x = x.detach()
        x, (hidden, cell) = self.lstm(x)
        x = torch.mean(x, dim = 1) 
        x = F.relu(self.fc(x)) 
        return x
    
    def forward(self ,x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return torch.clamp(F.cosine_similarity(x1, x2).reshape(-1,1), min=0.0, max = 1.0)
        #alternatively: 
        #return F.sigmoid(F.pairwise_distance(x1,x2).reshape(-1,1))
    
    
