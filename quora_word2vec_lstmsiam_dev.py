# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:17:58 2019

@author: Monica
"""

import os

os.chdir(r'C:\Users\Monica\Documents\ORwDS MSc\Dissertation')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
#from torch.utils.data import DataLoader, Dataset
#import torch.nn as nn
import torch.optim as optim
#import torch.nn.utils.rnn as rnn
import gensim.downloader as api
import embedder
import mynetworks
import architectures

#parameters
epochs = 15
batch_size = 16
path_root = 'Data\QuoraDuplicateQuestions'
embedding_dim = 300
max_tokens = 31
dataset_class = 'RawTextDataset'
train_path = 'Data\QuoraDuplicateQuestions\quora_train.csv'
val_path = 'Data\QuoraDuplicateQuestions\quora_train.csv'
D = embedding_dim*max_tokens
text_column1 = 'question1'
text_column2 = 'question2'
encoding = 'utf-8'

#from the paper: Learning Text Similarity with Siamese Recurrent Networks
'''
    hidden_size = 64
    num_layers = 4
    out_features = 128?????? <-I'm really unclear about this part from the paper
    bidirectional = True

'''


hidden_size = 64
num_layers = 4
out_features = 128 
dropout = 0.2 #THIS IS ARBITRARY AT THE MOMENT
bidirectional = True



#prepare data
train = pd.read_csv(train_path, header = 0, encoding = 'utf-8')
val = pd.read_csv(val_path, header = 0, encoding = 'utf-8')

train_size = train.shape[0]
val_size = val.shape[0]

#prepare embedding weights
vocab = api.load('word2vec-google-news-300')
embedder.expand_vocabulary(vocab, train, text_column1, text_column2, embedding_dim)
#add padding index
vocab.add('<PAD>', np.zeros(embedding_dim,))#but now I'm giving it a vector...
embedding_weights = torch.FloatTensor(vocab.vectors)
#vocab_size = embedding_weights.shape[0]

#train/val loaders


train_loader, val_loader = architectures.data_loaders_builder(dataset_class, batch_size, train_path = train_path, 
                                                              val_path = val_path, D = None, text_column1 = text_column1, 
                                                              text_column2 = text_column2, vocab = vocab, 
                                                              max_tokens = max_tokens)

#create an instance of the network
net = mynetworks.LSTMSiameseNet(embedding_dim, hidden_size, num_layers, out_features, dropout, bidirectional, 
                                embedding_weights, max_tokens, vocab.vocab.get('<PAD>').index)
#loss 
criterion = architectures.ContrastiveLoss()

#optimizer
optimizer = optim.Adam(net.parameters())#I'm just going to use default hyperparams for right now but we should revisit this



#training
train_loss_over_time = []
val_loss_over_time = []
#val_iter = iter(val_loader)


#net.double()
#net.train()
net.cpu()


#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
for e in range(epochs):  # loop over the dataset multiple times
    
    running_loss = 0.0
    running_loss_val = 0.0

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs1, inputs2, labels = data
        inputs1 = inputs1.squeeze().long()
        inputs2 = inputs2.squeeze().long()
        labels = labels.reshape(-1,1).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs1, inputs2)          
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #append loss
        running_loss += loss.item()/batch_size #average loss per item 
           
        if i*batch_size >= train_size:

            train_loss_over_time.append(running_loss)

            net.eval()

            with torch.no_grad():

                for j, data in enumerate(val_loader,0):
                    inputs1, inputs2, labels = data
                    inputs1 = inputs1.squeeze().long()
                    inputs2 = inputs2.squeeze().long()
                    labels = labels.squeeze().float()

                    outputs = net(inputs1, inputs2)
                    loss = criterion(outputs, labels)
                    running_loss_val += loss.item()


            val_loss_over_time.append(running_loss_val)

            net.train()

            torch.save({ #save model parameters
                 'epoch': e,
                 'model_state_dict': net.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': running_loss
                 }, 'Data/QuoraDuplicateQuestions/quora_word2vec_lstm_siam_epoch' + str(e) + '.pth')
    
    print('Finished Training')





len(train_loss_over_time)
len(val_loss_over_time)

plt.plot(np.arange(len(train_loss_over_time)), train_loss_over_time, np.arange(len(val_loss_over_time)), val_loss_over_time)
plt.xlabel('batch number')
plt.ylabel('loss')
plt.gca().legend(('train','validation'))
plt.show()




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
        inputs1, inputs2, labels = data
        inputs1 = inputs1.squeeze().float()
        inputs2 = inputs2.squeeze().float()
        labels = labels.reshape(-1,1).float()
        outputs = net(inputs1, inputs2)
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


