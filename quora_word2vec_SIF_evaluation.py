#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:00:40 2019

@author: s1834310
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
import nltk.tokenize
import sklearn.metrics
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import mynetworks
import architectures
import embedder





#RELOAD SAVEC (checkpoint-ed) MODEL
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
net = mynetworks.HighwayReluNet(300)
optimizer = optim.Adam(net.parameters())

checkpoint = torch.load('/afs/inf.ed.ac.uk/user/s18/s1834310/Documents/Dissertation/Models/Quora/Word2VecSIF/quora_word2vec_SIF14.pth',map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

net.eval()


val_set = architectures.SIFDataset('Data/QuoraDuplicateQuestions/quora_val_word2vec_SIF.csv', 300)
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
plt.title('Quora Word2Vec Simple Model ROC')
plt.legend(loc='lower right')
plt.show()


##OOV Statistics##

word2vec = api.load('word2vec-google-news-300')


##Train Set##

#percentage of vocab oov
train = pd.read_csv('Data/QuoraDuplicateQuestions/quora_train.csv', header = 0, encoding = 'utf-8')

text1 = train['question1'].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
text2 = train['question2'].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))

column1_list = text1.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
column2_list = text2.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()

    
df_vocab = list(set([item for sublist in column1_list for item in sublist] + \
                        [item for sublist in column2_list for item in sublist]))
oov_words = [word for word in df_vocab if word not in word2vec.vocab]
len(oov_words)/len(df_vocab)#37.02%
#n oov_words: 39077
#n df_vocab: 105546

#percentage of corpus oov
text_complete1 = [item for sublist in column1_list for item in sublist]
text_complete2 = [item for sublist in column2_list for item in sublist]

text_complete = text_complete1 + text_complete2
text_complete_oov = [word for word in text_complete if word not in word2vec.vocab]
len(text_complete_oov)/len(text_complete)#20.4%






##Validation Set## 

#percentage of vocab oov
val = pd.read_csv('Data/QuoraDuplicateQuestions/quora_val.csv', header = 0, encoding = 'utf-8')

text1 = val['question1'].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
text2 = val['question2'].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))

column1_list = text1.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
column2_list = text2.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()

    
df_vocab = list(set([item for sublist in column1_list for item in sublist] + \
                        [item for sublist in column2_list for item in sublist]))
oov_words = [word for word in df_vocab if word not in word2vec.vocab]
len(oov_words)/len(df_vocab)#28%

#percentage of corpus oov
text_complete1 = [item for sublist in column1_list for item in sublist]
text_complete2 = [item for sublist in column2_list for item in sublist]

text_complete = text_complete1 + text_complete2
text_complete_oov = [word for word in text_complete if word not in word2vec.vocab]
len(text_complete_oov)/len(text_complete)#20.5%






embedder.expand_vocabulary(word2vec, train, 'question1', 'question2', 300)


#percentage oov after vocab expanded
expanded_oov_words = [word for word in df_vocab if word not in word2vec.vocab]
len(expanded_oov_words)/len(df_vocab)#16.01%

#percentage of corpus oov after vocab expanded
expanded_text_complete_oov = [word for word in text_complete if word not in word2vec.vocab]
len(expanded_text_complete_oov)/len(text_complete)#0.5%




