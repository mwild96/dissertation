#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:51:16 2019
@author: s1834310
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
import nltk.tokenize
import torch
#import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
#import pytorch_pretrained_bert as bert
import pytorch_transformers as bert
#import allennlp.modules.elmo as elmo
#import torch.optim as optim
#import torch.nn.utils.rnn as rnn
import mynetworks
import embedder


#bert reference
#https://huggingface.co/pytorch-transformers/model_doc/bert.html#bertmodel


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
    
    def __init__(self, file_path, text_column1, text_column2, max_tokens, word_embedding_type, vocab = None, unk_index = None):
        self.data = pd.read_csv(file_path, header = 0)
        self.text_column1 = text_column1
        self.text_column2 = text_column2
        self.max_tokens = max_tokens
        self.word_embedding_type = word_embedding_type
        self.vocab = vocab
        self.unk_index = unk_index
        if word_embedding_type == 'bert':
            #self.tokenizer = bert.BertTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer = bert.BertTokenizer.from_pretrained('/home/s1834310/Dissertation/PretrainedBert')
            
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        label = self.data.iloc[index,self.data.shape[1]-1]
        
        if self.word_embedding_type != 'bert':
            
            text1 = pd.Series(self.data.loc[self.data.index[index], self.text_column1]).apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
            text2 = pd.Series(self.data.loc[self.data.index[index], self.text_column2]).apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
            
            text1 = text1.apply(str).apply(nltk.tokenize.word_tokenize)
            text2 = text2.apply(str).apply(nltk.tokenize.word_tokenize)
            
            
            if self.word_embedding_type == 'fasttext' or self.word_embedding_type == 'word2vec':
            
                text1 = text1.apply(lambda x: x + ['[PAD]'] * (self.max_tokens - len(x)) if len(x) < self.max_tokens else x[0:self.max_tokens])
                text2 = text2.apply(lambda x: x + ['[PAD]'] * (self.max_tokens - len(x)) if len(x) < self.max_tokens else x[0:self.max_tokens])
                
                text1 = text1.apply(lambda x: embedder.word2index(x, self.vocab, self.unk_index))
                text2 = text2.apply(lambda x: embedder.word2index(x, self.vocab, self.unk_index))
                
                text1 = torch.tensor(text1)
                text2 = torch.tensor(text2)
                
                
            elif self.word_embedding_type == 'elmo':
                
                text1 = elmo.batch_to_ids(text1)
                text2 = elmo.batch_to_ids(text2)
                
                
                text1 = text1.apply(lambda x: x + [0] * (self.max_tokens - len(x)) if len(x) < self.max_tokens else x[0:self.max_tokens])
                text2 = text2.apply(lambda x: x + [0] * (self.max_tokens - len(x)) if len(x) < self.max_tokens else x[0:self.max_tokens])
            
            return text1, text2, torch.tensor(label)
        
        if self.word_embedding_type == 'bert':
            
                
                #***NOTICE IM CURRENTLY DOING NO PADDING***#
    
                #note: BERT is trained on "combined token length" of <= 512 tokens
                
                threshold = min(512, self.max_tokens)
                
                bert_df1 = pd.Series(self.data.loc[self.data.index[index], 
                                                   self.text_column1]).apply(lambda x: embedder.bert_input_builder(str(x), self.tokenizer, threshold))
                
                bert_df2 = pd.Series(self.data.loc[self.data.index[index], 
                                   self.text_column2]).apply(lambda x: embedder.bert_input_builder(str(x), self.tokenizer, threshold))
                
                
                bert_df1 = bert_df1.apply(np.stack).apply(lambda x: np.reshape(x, (1,3*(threshold+2))))
                bert_df2 = bert_df2.apply(np.stack).apply(lambda x: np.reshape(x, (1,3*(threshold+2))))
                
                bert_df1 = np.stack(bert_df1).squeeze()
                bert_df2 = np.stack(bert_df2).squeeze()
                
                
                if bert_df1.ndim == 1:
                    
                    indexed_tokens1 = bert_df1[0:33]
                    segments1 = bert_df1[33:66]
                    input_mask1 = bert_df1[66:]

                    indexed_tokens2 = bert_df2[0:33]
                    segments2 = bert_df2[33:66]
                    input_mask2 = bert_df2[66:]
                
                elif bert_df1.ndim == 2:
                
                    indexed_tokens1 = bert_df1[:,0:33]
                    segments1 = bert_df1[:,33:66]
                    input_mask1 = bert_df1[:,66:]

                    indexed_tokens2 = bert_df2[:,0:33]
                    segments2 = bert_df2[:,33:66]
                    input_mask2 = bert_df2[:,66:]

                
                return indexed_tokens1, segments1, input_mask1, indexed_tokens2, segments2, input_mask2, label
            #torch.tensor(indexed_tokens1), torch.tensor(segments1), torch.tensor(input_mask1), torch.tensor(indexed_tokens2), torch.tensor(segments2), torch.tensor(input_mask2), torch.tensor(label)

        

    
def data_loaders_builder(dataset_class, batch_size, word_embedding_type, train_path, val_path, D = None,
                         text_column1 = None, text_column2 = None, vocab = None, unk_index = None, max_tokens = None):
    
    if dataset_class == 'SIFDataset':
        
        train_set = eval(dataset_class + '(train_path, D)')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
        val_set = eval(dataset_class + '(val_path, D)')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    if dataset_class == 'RawTextDataset':
        
        train_set = eval(dataset_class + '(train_path, text_column1, text_column2, max_tokens, word_embedding_type, vocab, unk_index)')#, vocab_size
        #idk if this is going to work with the padding thing we've done because
        #I think this loads it little by little? I can't tell if this only does it when DataLoader tells it to or if it does it all at once
        #and then DataLoader loads it little by little
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
        val_set = eval(dataset_class + '(val_path, text_column1, text_column2, max_tokens, word_embedding_type, vocab, unk_index)')#, vocab_size
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader     
    
    


    
#https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/    
class ContrastiveLoss(nn.Module):
   
    def __init__(self, margin=0.5): #definitely just stole that margin = 2 thing from the link above
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

    def forward(self, cosine_similarity, labels):
        # Find the pairwise distance or eucledian distance of two output feature vectors
        #cosine_similarity = F.cosine_similarity(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean(labels * 0.25*torch.pow((1-cosine_similarity), 2) \
        + (1 - labels) * (torch.pow(cosine_similarity, 2)*(cosine_similarity<self.margin).float() \
        + 0*(cosine_similarity>=self.margin).float()))
 
        return loss_contrastive
   
#https://github.com/stytim/LFW_Siamese_Pytorch/blob/master/Contrastive-Loss.py
class ContrastiveLoss2(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss2, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    


#training
def train(epochs, batch_size, train_loader, val_loader, train_size, val_size, D, optimizer, net, criterion, dev, output_filename, train_dataset_name, embedding_type, val = True):
    
    train_loss_over_time = []
    val_loss_over_time = []
   

    for e in range(epochs):  # loop over the dataset multiple times
    
        print('EPOCH: ', e)
        running_loss = 0.0
        running_loss_val = 0.0

        for i, data in enumerate(train_loader, 0):
            
            if net.__class__.__name__ == 'HighwayReluNet':
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.squeeze().float()#.view(-1,D)
                labels = labels.reshape(-1,1).float()
                inputs = inputs.to(dev)
                labels = labels.to(dev)


    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = net(inputs)
                
                
            elif net.__class__.__name__ == 'LSTMSiameseNet' or net.__class__.__name__ == 'ElmoLSTMSiameseNet':

                inputs1, inputs2, labels = data
                inputs1 = inputs1.squeeze().long()
                inputs2 = inputs2.squeeze().long()
                labels = labels.reshape(-1,1).float()
                inputs1 = inputs1.to(dev)
                inputs2 = inputs2.to(dev)
                labels = labels.to(dev)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs1, inputs2) 
                print(outputs)
            elif net.__class__.__name__ == 'BertLSTMSiameseNet':
                
                tokens1, segments1, input_mask1, tokens2, segments2, input_mask2, labels = data
                tokens1 = tokens1.long()
                segments1 = segments1.long()
                input_mask1 = input_mask1.long()
                tokens2 = tokens2.long()
                segments2 = segments2.long()
                input_mask2 = input_mask2.long()
                labels = labels.reshape(-1,1).float()
                tokens1 = tokens1.to(dev)
                segments1 = segments1.to(dev)
                input_mask1 = input_mask1.to(dev)
                tokens2 = tokens2.to(dev)
                segments2 = segments2.to(dev)
                input_mask2 = input_mask2.to(dev)
                labels = labels.to(dev)
                
                
                optimizer.zero_grad()
                
                outputs = net(tokens1, segments1, input_mask1, tokens2, segments2, input_mask2)
                
                
            loss = criterion(outputs, labels).to(dev)
            loss.backward()
            optimizer.step()
            
            #append loss
            running_loss += loss.item()#average loss per item 
           
            #if (i+1)*batch_size >= train_size:
            if (i+1) >= train_loader.__len__():

                train_loss_over_time.append(running_loss/train_size)#do average per item so magnitude is comparable against validation

                if val == True: 
                    net.eval()

                    with torch.no_grad():

                        for data in val_loader:

                            if net.__class__.__name__ == 'HighwayReluNet':

                                inputs, labels = data
                                inputs = inputs.squeeze().float()
                                labels = labels.reshape(-1,1).float()
                                inputs = inputs.to(dev)
                                labels = labels.to(dev)

                                outputs = net(inputs)

                            elif net.__class__.__name__ == 'LSTMSiameseNet':

                                inputs1, inputs2, labels = data
                                inputs1 = inputs1.squeeze().long()
                                inputs2 = inputs2.squeeze().long()
                                labels = labels.reshape(-1,1).float()
                                inputs1 = inputs1.to(dev)
                                inputs2 = inputs2.to(dev)
                                labels = labels.to(dev)

                                outputs = net(inputs1, inputs2)


                            elif net.__class__.__name__ == 'BertLSTMSiameseNet':

                                tokens1, segments1, input_mask1, tokens2, segments2, input_mask2, labels = data
                                tokens1 = tokens1.long()
                                segments1 = segments1.long()
                                input_mask1 = input_mask1.long()
                                tokens2 = tokens2.long()
                                segments2 = segments2.long()
                                input_mask2 = input_mask2.long()
                                labels = labels.reshape(-1,1).float()
                                tokens1 = tokens1.to(dev)
                                segments1 = segments1.to(dev)
                                input_mask1 = input_mask1.to(dev)
                                tokens2 = tokens2.to(dev)
                                segments2 = segments2.to(dev)
                                input_mask2 = input_mask2.to(dev)
                                labels = labels.to(dev)
                                
                                outputs = net(tokens1, segments1, input_mask1, tokens2, segments2, input_mask2)


                            loss = criterion(outputs, labels)
                            running_loss_val += loss.item()


                    val_loss_over_time.append(running_loss_val/val_size)#look at the average loss so we can try to make it comparable magnitude wise to the train set

                    net.train()

                torch.save({ #save model parameters
                     'epoch': e,
                     'model_state_dict': net.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': running_loss
                     }, output_filename + str(e) + '.pth')
    
    print('Finished Training')

    print(train_loss_over_time)
    print(val_loss_over_time)
    
    if val == True:
        p = plt.plot(np.arange(len(train_loss_over_time)), train_loss_over_time, np.arange(len(val_loss_over_time)), val_loss_over_time)
        plt.xlabel('epoch number')
        plt.ylabel('loss')
        plt.gca().legend(('train','validation'))
        plt.savefig(train_dataset_name + '_' + net.__class__.__name__ + '_' + embedding_type + '_Train_Val_Loss' + '.png')
    
    else:
        p = plt.plot(np.arange(len(train_loss_over_time)), train_loss_over_time)
        plt.xlabel('epoch number')
        plt.ylabel('loss')
        plt.gca().legend(('train','validation'))
        plt.savefig(train_dataset_name + '_' + net.__class__.__name__ + '_' + embedding_type + '_Train_Val_Loss' + '.png')



#validation
def evaluate(val_loader, D, net, dev, train_dataset_name, embedding_type):    
    
    
    #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    
    #validation
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
            
            if net.__class__.__name__ == 'HighwayReluNet':

                inputs, labels = data
                inputs = inputs.squeeze().float()
                labels = labels.reshape(-1,1).float()
                inputs = inputs.to(dev)
                labels = labels.to(dev)

                outputs = net(inputs)

            elif net.__class__.__name__ == 'LSTMSiameseNet':

                inputs1, inputs2, labels = data
                inputs1 = inputs1.squeeze().long()
                inputs2 = inputs2.squeeze().long()
                labels = labels.reshape(-1,1).float()
                inputs1 = inputs1.to(dev)
                inputs2 = inputs2.to(dev)
                labels = labels.to(dev)

                outputs = net(inputs1, inputs2)
                
                
            elif net.__class__.__name__ == 'BertLSTMSiameseNet':
    
                tokens1, segments1, input_mask1, tokens2, segments2, input_mask2, labels = data
                tokens1 = tokens1.long()
                segments1 = segments1.long()
                input_mask1 = input_mask1.long()
                tokens2 = tokens2.long()
                segments2 = segments2.long()
                input_mask2 = input_mask2.long()
                labels = labels.reshape(-1,1).float()
                tokens1 = tokens1.to(dev)
                segments1 = segments1.to(dev)
                input_mask1 = input_mask1.to(dev)
                tokens2 = tokens2.to(dev)
                segments2 = segments2.to(dev)
                input_mask2 = input_mask2.to(dev)
                labels = labels.to(dev)

                outputs = net(tokens1, segments1, input_mask1, tokens2, segments2, input_mask2)
            
            outputs_all.append(outputs.cpu())
            labels_all.append(labels.cpu())
            
            outputs = outputs.cpu()
            labels = labels.cpu()
            
            predicted = torch.round(outputs)
            #_, predicted = torch.max(outputs.data, 1)
            #NEED TO SEE IF THIS IS THE RIGHT WAY TO DO THIS FOR MY LOSS FUNCTION
               
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
           
            #predicted = predicted.cpu()
            
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
    
    
    labels_all = [item for sublist in labels_all for item in sublist]
    outputs_all = [item for sublist in outputs_all for item in sublist]

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels_all, outputs_all, pos_label = 1)
    auc = sklearn.metrics.auc(fpr, tpr)


    p = plt.plot(fpr, tpr, color = 'darkorange', label='ROC curve (area = %0.2f)' % auc)
    #plt.plot([0,1],[0,1], color = 'navy', linestyle = '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Amazon Google Bert Simple Model ROC')
    plt.legend(loc='lower right')
    plt.savefig(train_dataset_name + '_' + embedding_type + '_' + net.__class__.__name__ + '_ROC' '.png')

    
    net.train()
