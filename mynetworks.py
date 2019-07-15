# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:48:25 2019
@author: Monica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn




class LinearNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(self.in_features, self.out_features, bias = False)#is this a decent way to do this?
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class LogisticNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(LogisticNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(self.in_features, self.out_features, bias = False)#is this a decent way to do this?
        
    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x    


class HighwayReluNet(nn.Module):
    #two layer fully connected ReLU HighwayNet
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
        
        return torch.sigmoid(self.plain3(o2))#F.relu()?
    
        
        
class LSTMSiameseNet(nn.Module):
    '''
    Adapated from the paper "Learning Text Similarity with Siamese Recurrent Networks"
    
    "The propose network contains four layers of Bidirectional LSTM nodes. The activations at each timestep of the final 
    BLSTM layer are averaged to produce a fixed dimensional output. This output is projected through a single densely 
    connected feedforward layer"
    
    Hyperparameters were chosen according to this paper ***EXCEPT FOR THE DROPOUT PROBABILITY, 
    which was currently chosen arbitrarily**
    
    hidden_size = 64
    num_layers = 4
    out_features = 128?????? <-I'm really unclear about this part from the paper
    bidirectional = True
    dropout = 0.2?
    '''
    
    
    def __init__(self, embedding_dim, hidden_size, number_layers, out_features, dropout, bidirectional, embeddings, max_tokens, padding_idx):
        super(LSTMSiameseNet, self).__init__()
        #self.input_size = input_size
        #SIZE OF HIDDEN LAYERS???
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_features = out_features
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.max_tokens = max_tokens
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx = self.padding_idx)
        #^^NOTE: WE MAY WANT TO EXPAND OUR VOCABULARY BEFORE THIS, DEPENDING ON WHICH MODEL WE'RE USING
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size,
                            num_layers = self.number_layers, dropout = self.dropout, bidirectional = self.bidirectional)
        #HOW DO THEY KNOW THE OUTPUT DIMENSION I WANT?
        self.fc = nn.Linear(in_features = self.hidden_size, out_features = self.out_features)#I'M NOT CLEAR ON HOW THIS ENDS
    
    def forward_once(self, x):
        #DO THE EMBEDDING STEP
        x = self.embedding(x)#.long()
        #https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        #x = rnn.pack_padded_sequence(x, self.max_tokens, batch_first=True)
        _, x = self.lstm(x) #first item is all hidden states, second is most recent hidden state
        #I think the paper just uses the last hidden state
        #x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        #I can't tell if we're going to be okay without this or not
        x = torch.cat(x)
        x = torch.mean(x, dim = 1) #I'M NOT SURE IF I'M DOING THIS RIGHT
        x = self.fc(x) #WHAT IS THE ACTIVATION FUNCTION HERE? none?
        return x
    
    def forward(self ,x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return F.cosine_similarity(x1, x2)#is this what I'm supposed to return?
#outcome is cosince similarity IS bounded between [0,1] which is obviously useful for us
