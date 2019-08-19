# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:44:34 2019

@author: Monica
"""

import numpy as np
import pandas as pd
import sklearn.model_selection
import gensim.downloader as api
import embedder
#80870


##GENERAL DATA SET-UP##

#import data
quora = pd.read_csv('Data\QuoraDuplicateQuestions\questions.csv', header = 0, encoding = 'utf-8')


#for train-val-test split we use the ratio 3:1:1 according to the paper Design Space Exploration
#train-test split
quora_train, quora_test, quora_labels_train, quora_labels_test = sklearn.model_selection.train_test_split(quora.loc[:, quora.columns != 'is_duplicate'], quora['is_duplicate'], test_size = 0.2, random_state = 100)

#train-val split
quora_train, quora_val, quora_labels_train, quora_labels_val = sklearn.model_selection.train_test_split(quora_train, quora_labels_train, test_size = 0.25, random_state = 100)


#merge labels and data
quora_train = quora_train.merge(quora_labels_train, left_index = True, right_index = True)
quora_val = quora_val.merge(quora_labels_val, left_index = True, right_index = True)
quora_test = quora_test.merge(quora_labels_test, left_index = True, right_index = True)


#export
quora_train.to_csv('Data\QuoraDuplicateQuestions\quora_train.csv', index = False)
quora_val.to_csv('Data\QuoraDuplicateQuestions\quora_val.csv', index = False)
quora_test.to_csv('Data\QuoraDuplicateQuestions\quora_test.csv', index = False)



##BASELINE: RANDOMLY INITIALIZED WORD EMBEDDINGS##
vocab = embedder.build_vocabulary(quora_train, 'question1', 'question2', 300, 100)
question1_weight_dict = embedder.SIF_weights(quora_train, 'question1')
question2_weight_dict = embedder.SIF_weights(quora_train, 'question2')

#train#
quora_train_random_SIF = embedder.weighted_average_embedding_array('random', quora_train, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
quora_train_random_SIF = np.concatenate((quora_train_random_SIF, quora_labels_train.values.reshape(-1,1)), axis = 1)
pd.DataFrame(quora_train_random_SIF).to_csv('Data\QuoraDuplicateQuestions\quora_train_random_SIF.csv', index = False)

#val#
quora_val_random_SIF = embedder.weighted_average_embedding_array('random', quora_val, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
quora_val_random_SIF = np.concatenate((quora_val_random_SIF, quora_labels_val.values.reshape(-1,1)), axis = 1)
pd.DataFrame(quora_val_random_SIF).to_csv('Data\QuoraDuplicateQuestions\quora_val_random_SIF.csv', index = False)

#test#
quora_test_random_SIF = embedder.weighted_average_embedding_array('random', quora_test, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
quora_test_random_SIF = np.concatenate((quora_test_random_SIF, quora_labels_test.values.reshape(-1,1)), axis = 1)
pd.DataFrame(quora_test_random_SIF).to_csv('Data\QuoraDuplicateQuestions\quora_test_random_SIF.csv', index = False)






##WORD2VEC SIF##
vocab = api.load('word2vec-google-news-300')

#these steps should be done only on the training set
embedder.expand_vocabulary(vocab, quora_train, 'question1', 'question2', 300)
question1_weight_dict = embedder.SIF_weights(quora_train, 'question1')
question2_weight_dict = embedder.SIF_weights(quora_train, 'question2')

#now apply to all data
#train#
quora_train_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', quora_train, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
quora_train_word2vec_SIF.to_csv('Data\QuoraDuplicateQuestions\quora_train_word2vec_SIF.csv', index = False)

#val#
quora_val_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', quora_val, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_val_word2vec_SIF).to_csv('Data\QuoraDuplicateQuestions\quora_val_word2vec_SIF.csv', index = False)

#test#
quora_test_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', vocab, quora_test, 'question1', 'question2', question1_weight_dict, question1_weight_dict, 31, 300)
quora_test_word2vec_SIF.to_csv('Data\QuoraDuplicateQuestions\quora_test_word2vec_SIF.csv', index = False)









##FASTTEXT SIF##
vocab = api.load('fasttext-wiki-news-subwords-300')

#these steps should be done only on the training set
embedder.expand_vocabulary(vocab, quora_train, 'question1', 'question2', 300)
question1_weight_dict = embedder.SIF_weights(quora_train, 'question1')
question2_weight_dict = embedder.SIF_weights(quora_train, 'question2') 


#train#
quora_train_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', quora_train., 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_fasttext_SIF).to_csv('Data\QuoraDuplicateQuestions\quora_train_fasttext_SIF.csv', index = False)


#val#
quora_val_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', quora_val, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_val_fasttext_SIF).to_csv('Data\QuoraDuplicateQuestions\quora_val_fasttext_SIF.csv', index = False)

#test#
quora_test_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', vocab, quora_test, 'question1', 'question2', question1_weight_dict, question1_weight_dict, 31, 300)
quora_test_word2vec_SIF.to_csv('Data\QuoraDuplicateQuestions\quora_test_word2vec_SIF.csv', index = False)











