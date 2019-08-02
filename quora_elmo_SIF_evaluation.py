#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:59:59 2019

@author: s1834310
"""



import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import mynetworks
import architectures


net = mynetworks.HighwayReluNet(1024)
optimizer = optim.Adam(net.parameters())

checkpoint = torch.load('/afs/inf.ed.ac.uk/user/s18/s1834310/Documents/Dissertation/Models/Quora/ElmoSIF/quora_elmo_SIF14.pth',map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

net.eval()


val_set = architectures.SIFDataset('Data/QuoraDuplicateQuestions/quora_val_elmo_SIF.csv', 1024)
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
plt.title('Quora ELMo Simple Model ROC')
plt.legend(loc='lower right')
plt.show()
