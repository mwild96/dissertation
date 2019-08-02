#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:41:23 2019

@author: s1834310
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import mynetworks
import architectures





#RELOAD SAVEC (checkpoint-ed) MODEL
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
net = mynetworks.HighwayReluNet(768)
optimizer = optim.Adam(net.parameters())

checkpoint = torch.load('/afs/inf.ed.ac.uk/user/s18/s1834310/Documents/Dissertation/Models/Quora/BertSIF/quora_bert_SIF14.pth',map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

net.eval()


val_set = architectures.SIFDataset('Data/QuoraDuplicateQuestions/quora_val_bert_SIF.csv', 768)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)




##Validation Statistics##
outputs_all = []
labels_all = []

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
        inputs = inputs.squeeze().float()#.cuda(async=True)
        labels = labels.view(-1,1).float()#.cuda(async=True)
        labels_all.append(labels)
        
        outputs = net(inputs)
        outputs_all.append(outputs)
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



##Validation: ROC##

labels_all = [item for sublist in labels_all for item in sublist]
outputs_all = [item for sublist in outputs_all for item in sublist]

fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels_all, outputs_all, pos_label = 1)
auc = sklearn.metrics.auc(fpr, tpr)


plt.plot(fpr, tpr, color = 'darkorange', label='ROC curve (area = %0.2f)' % auc)
#plt.plot([0,1],[0,1], color = 'navy', linestyle = '--') <- doesn't make sense to plot this because we don't have balanced classes
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Quora BERT Simple Model ROC')
plt.legend(loc='lower right')
plt.show()





##ERROR ANALYSIS##


val = pd.read_csv('Data/QuoraDuplicateQuestions/quora_val_bert_SIF.csv', header = None)

net.eval()
with torch.no_grad():
    outputs = net(torch.tensor(val.iloc[:,0:768].values).float())
val['predicted'] = torch.round(outputs).numpy()

 
fn_indices = val.loc[val[768] != val['predicted'],:].loc[val['predicted']==0,:].index
fp_indices = val.loc[val[768] != val['predicted'],:].loc[val['predicted']==1,:].index
tp_indices = val.loc[val[768] == val['predicted'],:].loc[val['predicted']==1,:].index


val_raw = pd.read_csv('Data/QuoraDuplicateQuestions/quora_val.csv', header = 0)
val_raw_fn = val_raw.iloc[fn_indices,:]
val_raw_fp = val_raw.iloc[fp_indices,:]
val_raw_tp = val_raw.iloc[tp_indices,:]

for i in range(val_raw_fn.shape[0]):
    print('QUESTION 1: ')
    print(val_raw_fn.loc[val_raw_fn.index[i], 'question1'])
    print('QUESTION 2: ')
    print(val_raw_fn.loc[val_raw_fn.index[i], 'question2'])
    print(' ')
    print(' ')
    
val['outputs'] = outputs.numpy()
fn_confident_indices = val.loc[np.logical_and(val[768] != val['predicted'], val[768] == 1),:].loc[val['outputs']<=0.1,:].index
val_raw_fn_confident = val_raw.iloc[fn_confident_indices,:]

for i in range(val_raw_fn_confident.shape[0]):
    print('QUESTION 1: ')
    print(val_raw_fn_confident.loc[val_raw_fn_confident.index[i], 'question1'])
    print('QUESTION 2: ')
    print(val_raw_fn_confident.loc[val_raw_fn_confident.index[i], 'question2'])
    print(' ')
    print(' ')


    
for i in range(val_raw_tp.shape[0]):
    print('QUESTION 1: ')
    print(val_raw_fn.loc[val_raw_fn.index[i], 'question1'])
    print('QUESTION 2: ')
    print(val_raw_fn.loc[val_raw_fn.index[i], 'question2'])
    print(' ')
    print(' ')



