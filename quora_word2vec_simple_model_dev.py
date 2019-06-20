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
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import mynetworks




#***NEED TO MAKE SURE THAT WORD EMBEDDING VECTORS ARE NOT BEING TUNED



#parameters
epochs = 3#15 #THIS IS WHAT DESIGN SPACE EXPLORATION DOES
batch_size = 8 #BERT does 16 and 32 for training 
#(which is like an entirely different training scenario but I'm just playing it by ear right now)
path_root = 'Data\QuoraDuplicateQuestions' #SOME OUTPUT PATH TO SAVE OUR MODEL TO
embedding_dim = 300




    





#https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader


class SIFDataset(Dataset):
    
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        embedding = self.data[index, 0:embedding_dim]
        #SHOULD THIS BE HERE OR SHOULD THIS BE DONE ALL AT ONCE IN THE BEGINNING?
        label = self.data[index, embedding_dim]
 
        return embedding, label







#THIS MEANS I NEED TO CREATE 16 CLASSES AND DOES THAT REALLY MAKE SENSE?
# =============================================================================
# class QuoraWord2VecDataset(Dataset):
#     
#     def __init__(self, file_path):
#         self.data = pd.read_csv(file_path)
#         
#     def __len__(self):
#         return len(self.data)
#     
#     def __getitem__(self, index):
#         embedding = word2vec.weighted_average_word2vec_array(self.data.iloc[index,:], 'question1', 'question2', 25, 300)
#         #SHOULD THIS BE HERE OR SHOULD THIS BE DONE ALL AT ONCE IN THE BEGINNING?
#         label = np.asarray(self.data.loc[self.data.index[index], 'is_duplicate'])
#  
#         return embedding, label
# 
# 
# =============================================================================
# =============================================================================
# class EmbeddingDataset(Dataset):
#     
#     def __init__(self, file_path, text_field1, text_field2, labels, embedding_model, max_tokens, embedding_dim):
#         self.data = pd.read_csv(file_path)
#         self.text_field1 = text_field1
#         self.text_field2 = text_field2
#         self.labels = labels
#         self.embedding_model = embedding_model
#         self.max_tokens = max_tokens
#         self.embedding_dim = embedding_dim
#         
#     def __len__(self):
#         return len(self.data)
#     
#     def __getitem__(self, index):
#         embedding = self.embedding_model(self.data.iloc[index,:], self.text_field1, 
#                                          self.text_field2, self.max_tokens, self.embedding_dim)
#         #SHOULD THIS BE HERE OR SHOULD THIS BE DONE ALL AT ONCE IN THE BEGINNING?
#         label = self.data.loc[self.data.index[index], self.labels]
#  
#         
#         return embedding, label
# =============================================================================
    


train_set = SIFDataset('Data\QuoraDuplicateQuestions\quora_train_SIF.csv')#'Data\QuoraDuplicateQuestions\quora_train.csv'
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = SIFDataset('Data\QuoraDuplicateQuestions\quora_val_SIF.csv')
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)




#create an instance of the network
net = mynetworks.HighwayReluNet(input_size = embedding_dim)

#loss 
criterion = nn.BCELoss()#nn.BCEWithLogitsLoss()#I don't think I need to specify any of the args at this time
#I don't think we want to do the one with logits because we NEED sigmoid in our forward pass 
#(and I don't get how we'd get it in our forward pass if we don't add it like that...)

#optimizer
optimizer = optim.Adam(net.parameters())#I'm just going to use default hyperparams for right now but we should revisit this



#training
train_loss_over_time = []#I'M NOT QUITE SURE WHAT SIZE THIS WILL END UP BEING
val_loss_over_time = []
val_iter = iter(val_loader)


#net.double()
#net.train()
net.cpu()


#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
for e in range(epochs):  # loop over the dataset multiple times
    
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.view(-1,embedding_dim).float()
        labels = labels.reshape(-1,1).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        train_loss_over_time.append(loss.item())
        #print(loss.item())#it only prints one value
        #so is this like the cumulative 
        
        
        #check how we're doing with our validation error    
        #WE MIGHT NOT NEED TO CHECK THIS THIS OFTEN...
        net.eval()
        
        with torch.no_grad():
            inputs, labels = val_iter.next()
            inputs = inputs.reshape(-1,embedding_dim).float()
            labels = labels.reshape(-1,1).float()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss_over_time.append(loss.item())
            
        net.train()
        
        # model checkpoint & print statistics
        running_loss += loss.item()
        if i % 8 == 0:    # print every 1000 mini-batches
            
            torch.save({ #save model parameters
                'epoch': e,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss
                }, path_root + '\quora_word2vec_epoch' + str(e) + '_batch' + str(i) + '.pth')
            
            print('[%d, %5d] loss: %.3f' %
                  (epochs + 1, i + 1, running_loss / 8)) #1000
            #are we trying to print an estimate of the loss per batch then?
            #are we doing this right?
            running_loss = 0.0
            
            

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
        inputs, labels = data
        inputs = inputs.reshape(-1,embedding_dim).float()
        labels = labels.reshape(-1,1).float()
        outputs = net(inputs)
        print(outputs)
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











#RELOAD SAVEC (checkpoint-ed) MODEL
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
# =============================================================================
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
# 
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# 
# model.eval()
# # - or -
# model.train()
# 
# =============================================================================


train_iter = iter(train_loader)


for i, data in enumerate(train_loader, 0):
    print(i)
        # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data