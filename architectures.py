#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:51:16 2019

@author: s1834310
"""


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk.tokenize
import torch
#import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
#import torch.optim as optim
#import torch.nn.utils.rnn as rnn
import mynetworks
import embedder



##Dataset Classes


class SIFDataset(Dataset):
    
    def __init__(self, file_path, D):
        self.data = pd.read_csv(file_path, header = 0)
        self.D = D
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        embedding = np.asarray(self.data.iloc[index, 0:self.D])
        #SHOULD THIS BE HERE OR SHOULD THIS BE DONE ALL AT ONCE IN THE BEGINNING?
        label = np.asarray(self.data.iloc[index, self.D])
 
        return embedding, label
    
   
    
class RawTextDataset(Dataset):
    
    def __init__(self, file_path, text_column1, text_column2, vocab, max_tokens):
        self.data = pd.read_csv(file_path, header = 0)
        self.text_column1 = text_column1
        self.text_column2 = text_column2
        self.vocab = vocab
        self.max_tokens = max_tokens
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        text1 = nltk.tokenize.word_tokenize(pd.DataFrame(self.data.loc[self.data.index[index], self.text_column1]))
        text2 = nltk.tokenize.word_tokenize(pd.DataFrame(self.data.loc[self.data.index[index], self.text_column1]))
        
        text1 = text1 + ['<PAD>'] * (self.max_tokens - len(text1)) if len(text1) < self.max_tokens else text1[0:self.max_tokens]
        text2 = text2 + ['<PAD>'] * (self.max_tokens - len(text2)) if len(text2) < self.max_tokens else text2[0:self.max_tokens]
        
        
        text1 = embedder.word2index(text1, self.vocab)
        text2 = embedder.word2index(text2, self.vocab)
        
        label = self.data.iloc[index,self.data.shape[1]-1]
        
        return np.asarray(text1), np.asarray(text2), np.asarray(label)
    
    
    
    
def data_loaders_builder(dataset_class, batch_size, train_path = None, val_path = None, D = None,
                         text_column1 = None, text_column2 = None, vocab = None, max_tokens = None, vocab_size = None):
    
    if dataset_class == 'SIFDataset':
        
        train_set = eval(dataset_class + '(train_path, D)')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
        val_set = eval(dataset_class + '(val_path, D)')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    if dataset_class == 'RawTextDataset':
        train_set = eval(dataset_class + '(train_path, text_column1, text_column2, vocab, max_tokens)')#, vocab_size
        #idk if this is going to work with the padding thing we've done because
        #I think this loads it little by little? I can't tell if this only does it when DataLoader tells it to or if it does it all at once
        #and then DataLoader loads it little by little
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
        val_set = eval(dataset_class + '(val_path, text_column1, text_column2, vocab, max_tokens)')#, vocab_size
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader     
    
    


    
#https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/    
class ContrastiveLoss(nn.Module):

      def __init__(self, margin=2.0): #definitely just stole that margin = 2 thing from the link above
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, cosine_similarity, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            #cosine_similarity = F.cosine_similarity(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean(label * torch.pow(0.25*cosine_similarity, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2))

            return loss_contrastive    

    


#training
def train(epochs, batch_size, train_loader, val_loader, train_size, val_size, D, optimizer, net, criterion, dev, output_filename, train_dataset_name, val = True):
    
    train_loss_over_time = []
    val_loss_over_time = []
   

    for e in range(epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        running_loss_val = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.view(-1,D).float()
            labels = labels.view(-1,1).float()
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels).to(dev)
            loss.backward()
            optimizer.step()
            
            #append loss
            running_loss += loss.item()/batch_size #average loss per item 
           
        if i*batch_size <= train_size:

            train_loss_over_time.append(running_loss)

            if val == True: 
                net.eval()

                with torch.no_grad():

                    for data in val_loader:
                        inputs, labels = data
                        inputs = inputs.view(-1,D).float()#.cuda(async=True)
                        labels = labels.view(-1,1).float()#.cuda(async=True)
                        #inputs.cuda()
                        #labels.cuda()
                        inputs = inputs.to(dev)
                        labels = labels.to(dev)

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        running_loss_val += loss.item()


                val_loss_over_time.append(running_loss_val)

                net.train()

            torch.save({ #save model parameters
                 'epoch': e,
                 'model_state_dict': net.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': running_loss
                 }, output_filename + str(e) + '.pth')
    
    print('Finished Training')

    
    if val == True:
        p = plt.plot(np.arange(len(train_loss_over_time)), train_loss_over_time, np.arange(len(val_loss_over_time)), val_loss_over_time)
        plt.xlabel('batch number')
        plt.ylabel('loss')
        plt.gca().legend(('train','validation'))
        plt.savefig('TrainValLoss' + train_dataset_name + '.png')
    
    else:
        p = plt.plot(np.arange(len(train_loss_over_time)), train_loss_over_time)
        plt.xlabel('epoch*batch_size')
        plt.ylabel('loss')
        plt.gca().legend(('train','validation'))
        plt.savefig('TrainValLoss' + train_dataset_name + '.png')



#validation
def evaluate(val_loader, D, net, dev):    
    
    
    #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

    
    #validation
    correct = 0
    total = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    
    net.eval()#do I need to do this even if I don't have dropout in my model?
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.view(-1,D).float()
            labels = labels.view(-1,1).float()
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            
            outputs = net(inputs)
            predicted = torch.round(outputs)
            #_, predicted = torch.max(outputs.data, 1)
            #NEED TO SEE IF THIS IS THE RIGHT WAY TO DO THIS FOR MY LOSS FUNCTION
               
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
           
           
            fn += (np.logical_and(predicted == 0, labels == 1)).sum().item()
            fp += (np.logical_and(predicted == 1, labels == 0)).sum().item()
            tn += (np.logical_and(predicted == 0, labels == 0)).sum().item()
            tp += (np.logical_and(predicted == 1, labels == 1)).sum().item()
    
    print('Accuracy on validation set: %d %%' % (
        100 * correct / total))
    
    print('Precision on validation set: %d %%' % (
        100 * tp / (tp+fp)))
    
    print('Recall on validation set: %d %%' % (
        100 *  tp / (tp+fn)))
    
    print('F1 score on validation set: %d %%' % (
        100 *  (2*tp) / (2*tp + fp + fn)))
    
    net.train()
