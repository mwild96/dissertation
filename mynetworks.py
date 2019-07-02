# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:48:25 2019

@author: Monica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F




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
    
    
class LSTMNet(nn.Module):
    '''
    Adapated from the paper "Learning Text Similarity with Siamese Recurrent Networks"
    
    "The propose network contains four layers of Bidirectional LSTM nodes. The activations at each timestep of the final 
    BLSTM layer are averaged to produce a fixed dimensional output. This output is projected through a single densely 
    connected feedforward layer"
    
    Hyperparameters were chosen according to this paper ***EXCEPT FOR THE DROPOUT PROBABILITY, 
    which was currently chosen arbitrarily**
    
    hidden_size = 64
    num_layers = 4
    bidirectional = True
    '''
    
    
    def __init__(self, embedding_dim, hidden_size, number_layers, dropout, bidirectional,):
        super(LSTMNet, self).__init__()
        #self.input_size = input_size
        #SIZE OF HIDDEN LAYERS???
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        #SOMWEHERE IN HERE WE NEED TO DEFINE THE EMBEDDINGS SO WE CAN GET THEM LATER
        
        self.lstm = nn.LSTM(input_size = self.input_size, hidde_size = self.hidden_size,
                            num_layers = self.number_layers, dropout = self.dropout, bidirectional = self.bidirectional)
        #HOW DO THEY KNOW THE OUTPUT DIMENSION I WANT?
        
        
        #how do we make this bidirectional?
        
        #self.plain2 = nn.Linear(self.input_size,self.input_size)
        #self.plain3 = nn.Linear(self.input_size,1)
    
    def forward(self,x):
        #DO THE EMBEDDING STEP
        x = self.lstm(x)
        
#https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings        
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('path/to/file')
weights = torch.FloatTensor(model.vectors)


weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
# Get embeddings for index 1
input = torch.LongTensor([1])
embedding(input)
            
        
class SiameseNet(nn.Module):
    def __init__(self, ...):
        super(SiameseNet, self).__init__()
        
        
