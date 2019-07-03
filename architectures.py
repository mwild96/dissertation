#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:51:16 2019

@author: s1834310
"""


import arg_extractor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import mynetworks



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
    
    def __init__(self, file_path):
        #***FINISH THIS**
        
        

    
#https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/    
class ContrastiveLoss(nn.Module):

      def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            cosine_similarity = F.cosine_similarity(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean(label * torch.pow(0.25*cosine_similarity, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2))

            return loss_contrastive    
    
    
    
    
    
    
    
    
    
    
def data_loaders_builder(dataset_class, batch_size, train_path, val_path, D):
        
    train_set = eval(dataset_class + '(train_path, D)')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = eval(dataset_class + '(val_path, D)')
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader 
        


#training
def train(epochs, batch_size, train_loader, val_loader, val_size, D, optimizer, net, criterion, dev, output_filename, train_dataset_name, val = True):
    
    train_loss_over_time = []
    val_loss_over_time = []
   

    for e in range(epochs):  # loop over the dataset multiple times
    
        val_iter = iter(val_loader)
        val_counter = 0
        running_loss = 0.0  

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
    
    
            train_loss_over_time.append(loss.item())
            
            
            #check how we're doing with our validation error    
            #WE MIGHT NOT NEED TO CHECK THIS THIS OFTEN...
            
            
            if val == True: 
                net.eval()
                
                with torch.no_grad():
                    inputs, labels = val_iter.next()
                    inputs = inputs.view(-1,D).float()
                    labels = labels.view(-1,1).float()
                    inputs = inputs.to(dev)
                    labels = labels.to(dev)
        
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    val_loss_over_time.append(loss.item()/batch_size)#average loss per item
                    
                    val_counter += 1
                    
                    #reset the iterator if we've reached the end of it
                    if val_counter % np.floor(val_size/batch_size) == 0:
                        val_iter = iter(val_loader)
                        val_counter = 0
                    
                net.train()
            
            # model checkpoint & print statistics
            running_loss += loss.item()/batch_size #average loss per item 
            if i % 1000 == 0:    # print every 1000 mini-batches
                
                torch.save({ #save model parameters
                    'epoch': e,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss
                    }, output_filename + '_epoch' + str(e) + '_batch' + str(i) + '.pth')
                
                print('[%d, %5d] loss: %.3f' %
                      (e + 1, i + 1, running_loss / 1000))#the average loss over the previous 1000? 
                #are we trying to print an estimate of the loss per batch then?
                #are we doing this right?
                running_loss = 0.0
                
                
    
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
