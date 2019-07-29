# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:48:25 2019
@author: Monica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.utils.rnn as rnn
#import pytorch_pretrained_bert as bert
import pytorch_transformers as bert
#import allennlp.modules.elmo as elmo




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
    
    ***HOW DOES THE LSTM KNOW WHAT OUR PADDING INDEX IS / HOW TO DEAL WITH PADDING APPROPRIATELY?***
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
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
        #HOW DO THEY KNOW THE OUTPUT DIMENSION I WANT?
        self.fc = nn.Linear(in_features = self.hidden_size*(2*self.bidirectional), out_features = self.out_features)#I'M NOT CLEAR ON HOW THIS ENDS
    
    def forward_once(self, x):
        x = self.embedding(x)
        x, (hidden, cell) = self.lstm(x)
        x = torch.mean(x, dim = 1)
        x = F.relu(self.fc(x))
        return x
    
    def forward(self ,x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return torch.clamp(F.cosine_similarity(x1, x2).reshape(-1,1), min=0.4, max = 0.6)#F.pairwise_distance(x1,x2).reshape(-1,1)
    
    
    
class BertLSTMSiameseNet(nn.Module):
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
        #self.bert = bert.BertModel.from_pretrained('bert-base-uncased',
        #                                           output_hidden_states=True, output_attentions=True)
        self.bert = bert.BertModel.from_pretrained('/home/s1834310/Dissertation/PretrainedBert',
                                                   output_hidden_states=True, output_attentions=True)
                
        self.lstm = nn.LSTM(input_size = 4*self.embedding_dim, hidden_size = self.hidden_size,
                            num_layers = self.number_layers, dropout = self.dropout, bidirectional = self.bidirectional, batch_first = True)
        self.fc = nn.Linear(in_features = self.hidden_size*(2*self.bidirectional), out_features = self.out_features)
    
    def forward_once(self, tokens, segments, input_mask):
        self.bert.eval()
        with torch.no_grad():
            x, _ = self.bert(tokens, segments, input_mask)[-2:]
        x = torch.stack(x).squeeze()[8:12,:,:,:].reshape(tokens.shape[0],-1,
                                                         4*self.embedding_dim).squeeze() 
        x, (hidden, cell) = self.lstm(x)
        x = torch.mean(x, dim = 1) 
        x = F.relu(self.fc(x)) #WHAT IS THE ACTIVATION FUNCTION HERE? none?
        return x
    
    def forward(self, tokens1, segments1, input_mask1, tokens2, segments2, input_mask2):
        x1 = self.forward_once(tokens1, segments1, input_mask1)
        x2 = self.forward_once(tokens2, segments2, input_mask2)
        return F.cosine_similarity(x1, x2).reshape(-1,1)#F.pairwise_distance(x1,x2).reshape(-1,1)#
        
    
    
    
    
    
    
class ElmoLSTMSiameseNet(nn.Module):
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
    
    
    def __init__(self, embedding_dim, hidden_size, number_layers, out_features, dropout, bidirectional):#, embeddings, max_tokens, padding_idx
        super(BertLSTMSiameseNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.out_features = out_features
        self.dropout = dropout
        self.bidirectional = bidirectional
        #WE'RE NOT GOING TO BE ABLE TO ACCESS THIS ON THE CLUSTER...
        self.options_file ='https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        self.weight_file = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        self.elmo = elmo.Elmo(self.options_file, self.weight_file, 1, dropout = 0)
                
        
        
        self.lstm = nn.LSTM(input_size = 4*self.embedding_dim, hidden_size = self.hidden_size,
                            num_layers = self.number_layers, dropout = self.dropout, bidirectional = self.bidirectional)
        #HOW DO THEY KNOW THE OUTPUT DIMENSION I WANT?
        self.fc = nn.Linear(in_features = self.hidden_size, out_features = self.out_features)#I'M NOT CLEAR ON HOW THIS ENDS
    
    def forward_once(self, x):
        x = self.elmo(x)#IS THIS IT? DOES THIS INCLUDE THE WEIGHTED AVERAGE THING? ARE WE GOING TO BE LEARNING THE WEIGHTED AVERAGE WITHOUT FINE TUNING THE 
                
        x, (hidden, cell) = self.lstm(x)
        x = torch.mean(x, dim = 1) #I'M NOT SURE IF I'M DOING THIS RIGHT
        x = self.fc(x) #WHAT IS THE ACTIVATION FUNCTION HERE? none?
        return x
    
    def forward(self ,x1, x2):
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        return F.cosine_similarity(x1, x2)#is this what I'm supposed to return?
#outcome is cosince similarity IS bounded between [0,1] which is obviously useful for us
