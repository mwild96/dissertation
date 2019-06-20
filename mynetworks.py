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


class HighwayReluNet(nn.Module):
    #two layer fully connected ReLU HighwayNet (I think two layers means two hidden layers?)
    #Adam optimizer
    #BCEWithLogitsLoss
    #I'm not sure what the size of our layers should be
    def __init__(self, input_size):
        super(HighwayReluNet, self).__init__()
        self.input_size = input_size
        self.plain1 = nn.Linear(self.input_size, self.input_size)#I'm completely guessing on the size of things right now...
        self.plain2 = nn.Linear(self.input_size,self.input_size)
        self.plain3 = nn.Linear(self.input_size,1)
        
        self.gate1 = nn.Linear(self.input_size,self.input_size) 
        #I think there's going to be an issue with the sizing here that I'm going to have to figure out
        self.gate2 = nn.Linear(self.input_size,self.input_size)
        #self.gate3 = nn.Linear(self.input_size,1)
    


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
        
        #I'M NOT SURE IF I'M DOING THIS RIGHT
        h1 = F.relu(self.plain1(x)) #F.relu what's the difference between these two ReLUs and which one should I use
        t1 = torch.sigmoid(self.gate1(x))
        o1 = torch.add(h1*t1, (1.0-t1)*x)
        
        h2 = F.relu(self.plain2(o1)) #F.relu what's the difference between these two ReLUs and which one should I use
        t2 = torch.sigmoid(self.gate2(o1))
        o2 = torch.add(h2*t2, (1.0-t2)*o1)
        
        
        #IS THE LAST LAYER JUST NORMAL FULLY CONNECTED HERE?
# =============================================================================
#         h3 = F.relu(self.plain3(o2)) #F.relu what's the difference between these two ReLUs and which one should I use
#         t3 = torch.sigmoid(self.gate3(o2))
#         print(h3.size())
#         print(t3.size())
#         o3 = torch.add(h3*t3, (1.0-t3)*o2)
#         print(o3.size())
# =============================================================================
        
        
        return torch.sigmoid(self.plain3(o2))#F.relu()?
        
        
# =============================================================================
#     
# #criterion
# criterion = nn.BCEWithLogitsLoss()#I don't think I need to specify any of the args at this time
# 
# 
# #optimizer
# optimizer = optim.Adam(instance.params())#I'm just going to use default hyperparams for right now but we should revisit this
#     
# 
# 
# =============================================================================
        
        
        
    
#weight (Tensor, optional) – a manual rescaling weight given to the loss of each batch element. 
#If given, has to be a Tensor of size nbatch.
#what if we give a weight proportional to the class ratio
        
        
        
        
        