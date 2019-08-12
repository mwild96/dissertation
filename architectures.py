#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
import nltk.tokenize
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pytorch_transformers as bert
import allennlp.modules.elmo as elmo
import mynetworks
import embedder



##DATASET CLASSES##
class SIFDataset(Dataset):
    '''
    This dataset class is for the simple classifiers. The dataset itself should be pre-processed according to the SIF model
    in the paper "Deep Learning for Entity Matching: A Design Space Exploration" (that is, the rows in the .csv file should 
    be the absolute difference between the weighted averaged word embeddings of the two obervsations that constitute a pair).
    '''
    def __init__(self, file_path, D):
        self.data = pd.read_csv(file_path, header = 0)
        self.D = D
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        embedding = np.asarray(self.data.iloc[index, 0:self.D])
        label = np.asarray(self.data.iloc[index, self.D])
 
        return embedding, label
    
    
class RawTextDataset(Dataset):
    
    '''
    This dataset class is for the complex classifiers. The dataset itself should be the raw text of the observations that 
    constitute a pair. When getting an item, this class transforms the words of raw text into their associated indices 
    under the embedding model being used.
    '''
    
    def __init__(self, file_path, text_column1, text_column2, max_tokens, word_embedding_type, vocab = None, unk_index = None):
        self.data = pd.read_csv(file_path, header = 0)
        self.text_column1 = text_column1
        self.text_column2 = text_column2
        self.max_tokens = max_tokens
        self.word_embedding_type = word_embedding_type
        self.vocab = vocab
        self.unk_index = unk_index
        if word_embedding_type == 'bert':
            #bert reference
            #https://huggingface.co/pytorch-transformers/model_doc/bert.html#bertmodel
            self.tokenizer = bert.BertTokenizer.from_pretrained('bert-base-uncased')
            #self.tokenizer = bert.BertTokenizer.from_pretrained('/home/s1834310/Dissertation/PretrainedBert')
        
            
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        label = self.data.iloc[index,self.data.shape[1]-1]
        
        
        if self.word_embedding_type != 'bert':
            
            #remove contractions
            text1 = pd.Series(self.data.loc[self.data.index[index], self.text_column1]).apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
            text2 = pd.Series(self.data.loc[self.data.index[index], self.text_column2]).apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
            
            #tokenize
            text1 = text1.apply(str).apply(nltk.tokenize.word_tokenize)
            text2 = text2.apply(str).apply(nltk.tokenize.word_tokenize)
            
            
            if self.word_embedding_type == 'fasttext' or self.word_embedding_type == 'word2vec' or self.word_embedding_type == 'random':
            
                #padding
                text1 = text1.apply(lambda x: x + ['[PAD]'] * (self.max_tokens - len(x)) if len(x) < self.max_tokens else x[0:self.max_tokens])
                text2 = text2.apply(lambda x: x + ['[PAD]'] * (self.max_tokens - len(x)) if len(x) < self.max_tokens else x[0:self.max_tokens])
                
                #convert to indices
                text1 = text1.apply(lambda x: embedder.word2index(x, self.vocab, self.unk_index))
                text2 = text2.apply(lambda x: embedder.word2index(x, self.vocab, self.unk_index))
                
                text1 = torch.tensor(text1)
                text2 = torch.tensor(text2)
                label = torch.tensor(label)
                
                
            elif self.word_embedding_type == 'elmo':
                
                #convert to indices
                text1 = elmo.batch_to_ids(text1)
                text2 = elmo.batch_to_ids(text2)
                
                #padding
                if text1.shape[1] < self.max_tokens:
                    padded1 = torch.zeros(text1.shape[0], self.max_tokens, 50)
                    padded1[:, 0:text1.shape[1], :] = text1
                    
                elif text1.shape[1] > self.max_tokens:
                    padded1 = text1[:, 0:self.max_tokens, :]
                    
                elif text1.shape[1] == self.max_tokens:
                    padded1 = text1
                
                
                
                if text2.shape[1] < self.max_tokens:
                    padded2 = torch.zeros(text2.shape[0], self.max_tokens, 50)
                    padded2[:, 0:text2.shape[1], :] = text2
                    
                elif text2.shape[1] > self.max_tokens:
                    padded2 = text2[:, 0:self.max_tokens, :]
                    
                elif text2.shape[1] == self.max_tokens:
                    padded2 = text2  
                    
                
                
                text1 = padded1.long()
                text2 = padded2.long()
                label = torch.from_numpy(np.asarray(label)).float()
                
            return text1, text2, label
        
        if self.word_embedding_type == 'bert':
            
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
        

    
def data_loaders_builder(dataset_class, batch_size, word_embedding_type, train_path, val_path, D = None,
                         text_column1 = None, text_column2 = None, vocab = None, unk_index = None, max_tokens = None):
    '''
    This function returns the data loaders for the training and validation sets, to be used during training of the Networks.
    
    Inputs:
    dataset_class: a string, the dataset class to be used ('SIFDataset' for the simple classifiers, and 'RawTextDataset' for the complex
    classifiers
    batch_size: an int, the number of pairs to use in each mini-batch of training
    word_embedding_type: a string, the word embedding type being used
    train_path: a string, the path to the training dataset stored in a .csv file
    val_path: a string, the path to the validation dataset stored in a .csv file
    D: an int, the dimension of the word embeddings under the model being used, only necessary for 'SIFDataset'
    text_column1: a string, the name of the column that contains the text from the first dataset, only necessary for 
    'RawTextDataset'
    text_column2: a string, the name of the column that contains the text from the second dataset, only necessary for 
    'RawTextDataset'
    vocab: a gensim.KeyedVectors object, the vocabulary to be used for the random, word2vec, or fasttext embeddings, only 
    necessary for the RawTextDataset
    unk_index: an int, the index associated with unknown words when word2vec or fasttext embeddings are used, only necessary
    for the RawTextDataset
    max_tokens: an int, the maximum number of tokens to use when padding sequences, only necessary for the RawTextDataset
    
    Outputs:
    train_loader: torch DataLoader object, an incremental dataloader for the train set
    val_loader: torch DataLoader object, an incremental dataloader for the validation set
    '''
    
    
    if dataset_class == 'SIFDataset':
        
        train_set = eval(dataset_class + '(train_path, D)')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
        val_set = eval(dataset_class + '(val_path, D)')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    if dataset_class == 'RawTextDataset':
        
        train_set = eval(dataset_class + '(train_path, text_column1, text_column2, max_tokens, word_embedding_type, vocab, unk_index)')#, vocab_size
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
        val_set = eval(dataset_class + '(val_path, text_column1, text_column2, max_tokens, word_embedding_type, vocab, unk_index)')#, vocab_size
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader     
    
    

##LOSSES##
    
#https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/    
class ContrastiveLoss(nn.Module):
    '''
    This loss was designed for use with the complex classifier. It is based off the LSTM Siamese Network used in 
    "Learning Text Similarity with Siamese Recurrent Networks". In the forward method, it expects the cosine_similarity
    of the output of the two branches of the Siamese Network as input.
    '''
    def __init__(self, margin=0.5):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

    def forward(self, cosine_similarity, labels):
        loss_contrastive = torch.mean(labels * 0.25*torch.pow((1-cosine_similarity), 2) \
        + (1 - labels) * (torch.pow(cosine_similarity, 2)*(cosine_similarity<self.margin).float() \
        + 0*(cosine_similarity>=self.margin).float()))
 
        return loss_contrastive
   
#https://github.com/stytim/LFW_Siamese_Pytorch/blob/master/Contrastive-Loss.py
class ContrastiveLoss2(nn.Module):
    '''
    This loss is an alternative loss that can be used with the LSTM Siamese Network of the complex classifier.
    Instead of using the cosine similarity it uses the euclidean distance between the output of the two branches 
    of the Siamese Network as input to the forward method.
    '''
    def __init__(self, margin=1.0):
        super(ContrastiveLoss2, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) \
                                      + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    


##TRAINING##

def train(epochs, batch_size, train_loader, val_loader, train_size, val_size, D, optimizer, net, criterion, dev, output_filename, train_dataset_name, embedding_type, val = True):
    '''
    This function trains a network given a training dataset, an optimizer, and a loss function. It is designed for use with GPUs. 
    
    Inputs:
    epochs: an int, the number of times to pass over the full dataset during training
    batch_size: an int, the number of obervsations to use in each mini-batch of training
    train_loader: a torch DataLoader object, dataloader that loads training pairs by batch_size
    val_loader: a torch DataLoader object, dataloader that loads validation pairs by batch_size
    train_size: an int, the number of pairs in the training set
    val_size: an int, the number of pairs in the validation set
    D: an int, the dimension of the word embeddings
    optimizer: a torch Optimizer object
    net: a torch nn.Module object, the network to be trained
    criterion: a torch nn.Module object, the loss to be optimized in training
    dev: an int/string, the name of the GPU device to be used for training
    output_filename: a string, the base name of the file in which to output model checkpoints
    train_dataset_name: a string, the name of the training dataset
    embedding_type: a string, the name of the embedding type to be used
    val: a boolean, indicating whether or not to perform validation as training progresses (if True, a complete pass over the 
    validation set is performed at the end of every epoch; the average loss is saved for plotting)
    
    Outputs:
    p: a plot, of the training loss versus the epoch (if val == True, inclues the validation loss versus the epoch as well)
    '''
    
    train_loss_over_time = []
    val_loss_over_time = []
   

    for e in range(epochs):  # loop over the dataset multiple times
    
        print('EPOCH: ', e)
        running_loss = 0.0
        running_loss_val = 0.0

        for i, data in enumerate(train_loader, 0):
            
            if net.__class__.__name__ == 'HighwayReluNet':
                # get the inputs
                inputs, labels = data
                inputs = inputs.squeeze().float()
                labels = labels.reshape(-1,1).float()
                
                #synchronize the inputs to the GPU
                inputs = inputs.to(dev)
                labels = labels.to(dev)

    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
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

                # forward
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
                
                optimizer.zero_grad()
                
                outputs = net(tokens1, segments1, input_mask1, tokens2, segments2, input_mask2)
                
                
            loss = criterion(outputs, labels).to(dev)
            #backpropagation
            loss.backward()
            #update parameters
            optimizer.step()
            
            #append loss
            running_loss += loss.item()#average loss per item 
           
            if (i+1) >= train_loader.__len__():#at the end of every epoch
                
                #save average training loss for that epoch
                train_loss_over_time.append(running_loss/train_size)#average so magnitude is comparable against validation
                
                #get and save average validation loss for that epoch
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

                            elif net.__class__.__name__ == 'LSTMSiameseNet' or net.__class__.__name__ == 'ElmoLSTMSiameseNet':

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


                    val_loss_over_time.append(running_loss_val/val_size)#average loss so comparable magnitude to train set

                    net.train()
                    
                #model checkpoint
                torch.save({
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



##VALIDATION##

#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
def evaluate(val_loader, D, net, dev, train_dataset_name, embedding_type):   
    '''
    This function performs the validation for a network on a given validation set. It is designed for use with GPUs.
    
    Inputs:
    val_loader: a torch Dataloader object, a torch DataLoader object, dataloader that loads validation pairs
    D: an int, the dimension of the word embeddings used
    net: a torch nn.Module object, the network to be evaluated
    dev: an int/string, the name of the GPU to be used
    train_dataset_name: a string, the name of the dataset used to trian the network,
    embedding_type: a string, the name of the word embeddings used
    '''

    outputs_all = []
    labels_all = []
    
    correct = 0
    total = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    
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

            elif net.__class__.__name__ == 'LSTMSiameseNet' or net.__class__.__name__ == 'ElmoLSTMSiameseNet':

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
    
    
    labels_all = [item for sublist in labels_all for item in sublist]
    outputs_all = [item for sublist in outputs_all for item in sublist]

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels_all, outputs_all, pos_label = 1)
    auc = sklearn.metrics.auc(fpr, tpr)


    p = plt.plot(fpr, tpr, color = 'darkorange', label='ROC curve (area = %0.2f)' % auc)
    #plt.plot([0,1],[0,1], color = 'navy', linestyle = '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(train_dataset_name + '_' + embedding_type + '_' + net.__class__.__name__ + '_ROC' '.png')

    
    net.train()
