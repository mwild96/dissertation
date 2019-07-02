#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:52:10 2019
@author: s1834310
This file develops and outputs the word embedding arrays for the Amazon Google data set.
The process involves the following steps:
    original data import and merge
    blocking and undersampling of the negative class
    train/val/test split
    array of word2vec embeddings produced
    array of fasttext embeddings produced
    array of elmo embeddings produced
    array of bert embeddings produced
"""


import sys

sys.path.remove('C:\\Users\\Monica\\Anaconda3\\Lib\\site-packages')

import os

#os.chdir('/afs/inf.ed.ac.uk/user/s18/s1834310/Documents/Dissertation')
os.chdir(r'C:\Users\Monica\Documents\ORwDS MSc\Dissertation')


import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.utils
import gensim.downloader as api
import chardet
import blocking
import embedder


#import data
with open('Data/AmazonGoogle/Amazon.csv', 'rb') as f:
    result = chardet.detect(f.read())
    
amazon1 = pd.read_csv("Data/AmazonGoogle/Amazon.csv", header = 0, delimiter = ',', encoding = result['encoding'])
google = pd.read_csv("Data/AmazonGoogle/GoogleProducts.csv", header = 0, delimiter = ',', encoding = result['encoding'])
amaz_goog_matches = pd.read_csv("Data/AmazonGoogle/Amzon_GoogleProducts_perfectMapping.csv", header = 0, delimiter = ',')


#merge datasets: every possible combination of observations
amaz_goog = amazon1.assign(key=0).merge(google.assign(key=0), how='left', on = 'key', suffixes=('_az', '_gg'))
#should be 3226*1363 = 4397038 observations 
amaz_goog.drop(['key'], axis = 1, inplace = True)


#add labels 
amaz_goog = amaz_goog.merge(amaz_goog_matches.assign(match = 1), how = 'left', left_on = ['id_az', 'id_gg'], right_on=['idAmazon', 'idGoogleBase'] )
amaz_goog.drop(['idAmazon', 'idGoogleBase'], axis = 1, inplace = True)
amaz_goog.loc[amaz_goog['match'].isnull() == True,'match'] = 0


#combine title and description into a single text field
amaz_goog['description_az'] = amaz_goog.description_az.fillna('')
amaz_goog['description_gg'] = amaz_goog.description_gg.fillna('')

amaz_goog['text_az'] = amaz_goog['title'] + ' ' + amaz_goog['description_az']
amaz_goog['text_gg'] = amaz_goog['name'] + ' ' + amaz_goog['description_gg']


#remove null values (unusuable)
null_indices = np.asarray(amaz_goog.loc[np.logical_or(pd.isnull(amaz_goog['text_az']), pd.isnull(amaz_goog['text_gg'])),].index)
amaz_goog.drop(index = null_indices, inplace = True)




##BLOCKING##
amaz_goog = blocking.percentage_shared_tokens_blocking(amaz_goog, 
                                                       'text_az', 'text_gg', 0.05)



##TRAIN-TEST SPLIT##

#***MAKE SURE THESE 

#train-test split
amaz_goog_train, amaz_goog_test, amaz_goog_labels_train, amaz_goog_labels_test = sklearn.model_selection.train_test_split(amaz_goog.loc[:, amaz_goog.columns != 'match'], amaz_goog['match'], test_size = 0.2, random_state = 100)

#train-val split
amaz_goog_train, amaz_goog_val, amaz_goog_labels_train, amaz_goog_labels_val = sklearn.model_selection.train_test_split(amaz_goog_train, amaz_goog_labels_train, test_size = 0.25, random_state = 100)


#merge labels and data
amaz_goog_train = amaz_goog_train.merge(amaz_goog_labels_train, left_index = True, right_index = True)
amaz_goog_val = amaz_goog_val.merge(amaz_goog_labels_val, left_index = True, right_index = True)
amaz_goog_test = amaz_goog_test.merge(amaz_goog_labels_test, left_index = True, right_index = True)



##UPSAMPLING THE POSITIVE CLASS IN TRAIN##
#https://elitedatascience.com/imbalanced-classes
majority = amaz_goog_train.loc[amaz_goog_train['match']==0,:]
minority = amaz_goog_train.loc[amaz_goog_train['match']==1,:]
 
# Upsample minority class
minority_upsampled = sklearn.utils.resample(minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=amaz_goog.loc[amaz_goog['match']==0,:].shape[0],    # to match majority class
                                 random_state=100) # reproducible results
 
# Combine majority class with upsampled minority class
amaz_goog_train = pd.concat([majority, minority_upsampled])


#export
amaz_goog_train.to_csv('Data\AmazonGoogle\amaz_goog_train.csv', index = False)
amaz_goog_val.to_csv('Data\AmazonGoogle\amaz_goog_val.csv', index = False)
amaz_goog_test.to_csv('Data\AmazonGoogle\amaz_goog_test.csv', index = False)






##WORD2VEC SIF##
vocab = api.load('word2vec-google-news-300')

embedder.expand_vocabulary(vocab, amaz_goog_train, 'text_az', 'text_gg', 300)
text_az_weight_dict = embedder.SIF_weights(amaz_goog_train, 'text_az') #I THINK IT'S VERY IMPORTANT THAT TRAIN GOES THROUGH THIS
text_gg_weight_dict = embedder.SIF_weights(amaz_goog_train, 'text_gg') #BUT REALIZE THAT WILL CREATE PROBLEMS WITH VAL

#train
amaz_goog_train_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', amaz_goog_train, 'text_az', 'text_gg', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_train_word2vec_SIF).to_csv('Data\AmazonGoogle\amaz_goog_train_word2vec_SIF.csv', index = False)


#val
amaz_goog_val_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', amaz_goog_val, 'text_az', 'text_gg', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_val_word2vec_SIF).to_csv('Data\AmazonGoogle\amaz_goog_val_word2vec_SIF.csv', index = False)


#test
amaz_goog_test_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', amaz_goog_test, 'text_az', 'text_gg', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_test_word2vec_SIF).to_csv('Data\AmazonGoogle\amaz_goog_test_word2vec_SIF.csv', index = False)






##FASTTEXT SIF##
vocab = api.load('fasttext-wiki-news-subwords-300')

embedder.expand_vocabulary(vocab, amaz_goog_train, 'text_az', 'text_gg', 300)
text_az_weight_dict = embedder.SIF_weights(amaz_goog_train, 'text_az') #I THINK IT'S VERY IMPORTANT THAT TRAIN GOES THROUGH THIS
text_gg_weight_dict = embedder.SIF_weights(amaz_goog_train, 'text_gg') #BUT REALIZE THAT WILL CREATE PROBLEMS WITH VAL

#train
amaz_goog_train_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', amaz_goog_train, 'text_az', 'text_gg', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_train_fasttext_SIF).to_csv('Data\AmazonGoogle\amaz_goog_train_fasttext_SIF.csv', index = False)


#val
amaz_goog_val_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', amaz_goog_val, 'text_az', 'text_gg', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_val_fasttext_SIF).to_csv('Data\AmazonGoogle\amaz_goog_val_fasttext_SIF.csv', index = False)


#test
amaz_goog_test_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', amaz_goog_test, 'text_az', 'text_gg', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_test_fasttext_SIF).to_csv('Data\AmazonGoogle\amaz_goog_test_fasttext_SIF.csv', index = False)






