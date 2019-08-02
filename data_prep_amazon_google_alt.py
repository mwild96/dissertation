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


import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.utils
import gensim.downloader as api
import chardet
import blocking
import embedder


#import data
with open('Data/AmazonGoogle/amaz_goog_train.csv', 'rb') as f:
    result = chardet.detect(f.read())
    
amaz_goog_train = pd.read_csv('Data/AmazonGoogle/amaz_goog_train.csv', header = 0, encoding = result['encoding'])
amaz_goog_val = pd.read_csv('Data/AmazonGoogle/amaz_goog_val.csv', header = 0, encoding = result['encoding'])
#amaz_goog_test = pd.read_csv('Data/AmazonGoogle/amaz_goog_test.csv', header = 0, encoding = result['encoding'])

##WORD2VEC SIF##
vocab = api.load('word2vec-google-news-300')

embedder.expand_vocabulary(vocab, amaz_goog_train, 'title', 'name', 300)
text_az_weight_dict = embedder.SIF_weights(amaz_goog_train, 'title') #I THINK IT'S VERY IMPORTANT THAT TRAIN GOES THROUGH THIS
text_gg_weight_dict = embedder.SIF_weights(amaz_goog_train, 'name') #BUT REALIZE THAT WILL CREATE PROBLEMS WITH VAL

#train
amaz_goog_train_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', amaz_goog_train, 'title', 'name', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_train_word2vec_SIF).to_csv('Data/AmazonGoogle/amaz_goog_train_word2vec_SIF_alt.csv', index = False)


#val
amaz_goog_val_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', amaz_goog_val, 'title', 'name', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_val_word2vec_SIF).to_csv('Data/AmazonGoogle/amaz_goog_val_word2vec_SIF_alt.csv', index = False)


#test
#amaz_goog_test_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', amaz_goog_test, 'title', 'name', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
#pd.DataFrame(amaz_goog_test_word2vec_SIF).to_csv('Data/AmazonGoogle/amaz_goog_test_word2vec_SIF_alt.csv', index = False)






##FASTTEXT SIF##
vocab = api.load('fasttext-wiki-news-subwords-300')

embedder.expand_vocabulary(vocab, amaz_goog_train, 'title', 'text_gg', 300)
text_az_weight_dict = embedder.SIF_weights(amaz_goog_train, 'title') #I THINK IT'S VERY IMPORTANT THAT TRAIN GOES THROUGH THIS
text_gg_weight_dict = embedder.SIF_weights(amaz_goog_train, 'text_gg') #BUT REALIZE THAT WILL CREATE PROBLEMS WITH VAL

#train
amaz_goog_train_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', amaz_goog_train, 'title', 'name', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_train_fasttext_SIF).to_csv('Data/AmazonGoogle/amaz_goog_train_fasttext_SIF_alt.csv', index = False)


#val
amaz_goog_val_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', amaz_goog_val, 'title', 'name', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
pd.DataFrame(amaz_goog_val_fasttext_SIF).to_csv('Data/AmazonGoogle/amaz_goog_val_fasttext_SIF_alt.csv', index = False)


#test
#amaz_goog_test_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', amaz_goog_test, 'title', 'name', text_az_weight_dict, text_gg_weight_dict, vocabulary = vocab)
#pd.DataFrame(amaz_goog_test_fasttext_SIF).to_csv('Data/AmazonGoogle/amaz_goog_test_fasttext_SIF_alt.csv', index = False)



