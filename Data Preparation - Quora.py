# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:44:34 2019

@author: Monica
"""

import os

os.chdir(r'C:\Users\Monica\Documents\ORwDS MSc\Dissertation')


import pandas as pd
import sklearn.model_selection
import gensim.downloader as api
import embedder
#80870

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
#actually like...do I even need a test set? Since I'm just doing this for model comparison/selection purposes?
#I mean maybe if we're going to do hyperparameter tuning on each data set-word embedding model combo?

#export
quora_train.to_csv('Data\QuoraDuplicateQuestions\quora_train.csv', index = False)
quora_val.to_csv('Data\QuoraDuplicateQuestions\quora_val.csv', index = False)
quora_test.to_csv('Data\QuoraDuplicateQuestions\quora_test.csv', index = False)



#word2vec SIF
vocab = api.load('word2vec-google-news-300')


#these steps should be done only on the training set
embedder.expand_vocabulary(vocab, quora_train, 'question1', 'question2', 300)
question1_weight_dict = embedder.SIF_weights(quora_train, 'question1') #I THINK IT'S VERY IMPORTANT THAT TRAIN GOES THROUGH THIS
question2_weight_dict = embedder.SIF_weights(quora_train, 'question2') #BUT REALIZE THAT WILL CREATE PROBLEMS WITH VAL

#now apply to all data
#choose 31 max tokens because that is the 99th percentile across the two fields
quora_train_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', quora_train, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
quora_train_word2vec_SIF.to_csv('Data\QuoraDuplicateQuestions\quora_train_word2vec_SIF.csv', index = False)


#but in reality we have to do it in four parts
quora_train_word2vec_SIF1 = embedder.weighted_average_embedding_array('word2vec', quora_train.iloc[0:60652,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_word2vec_SIF1).to_csv('Data\QuoraDuplicateQuestions\quora_train_word2vec_SIF1.csv', index = False)



quora_train_word2vec_SIF2 = embedder.weighted_average_embedding_array('word2vec', quora_train.iloc[60652:121305,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_word2vec_SIF2).to_csv('Data\QuoraDuplicateQuestions\quora_train_word2vec_SIF2.csv', index = False)



quora_train_word2vec_SIF3 = embedder.weighted_average_embedding_array('word2vec', quora_train.iloc[121305:181957,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_word2vec_SIF3).to_csv('Data\QuoraDuplicateQuestions\quora_train_word2vec_SIF3.csv', index = False)



quora_train_word2vec_SIF4 = embedder.weighted_average_embedding_array('word2vec', quora_train.iloc[181957:,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_word2vec_SIF4).to_csv('Data\QuoraDuplicateQuestions\quora_train_word2vec_SIF4.csv', index = False)




#DONE 
quora_val_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', quora_val, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_val_word2vec_SIF).to_csv('Data\QuoraDuplicateQuestions\quora_val_word2vec_SIF.csv', index = False)

#not done yet
quora_test_word2vec_SIF = embedder.weighted_average_embedding_array('word2vec', vocab, quora_test, 'question1', 'question2', question1_weight_dict, question1_weight_dict, 31, 300)
quora_test_word2vec_SIF.to_csv('Data\QuoraDuplicateQuestions\quora_test_word2vec_SIF.csv', index = False)









#fasttext SIF
vocab = api.load('fasttext-wiki-news-subwords-300')

#these steps should be done only on the training set
embedder.expand_vocabulary(vocab, quora_train, 'question1', 'question2', 300)
question1_weight_dict = embedder.SIF_weights(quora_train, 'question1')#I think this is the same regardless of the vocabulary right?
question2_weight_dict = embedder.SIF_weights(quora_train, 'question2') 


#TRAIN
quora_train_fasttext_SIF1 = embedder.weighted_average_embedding_array('fasttext', quora_train.iloc[0:60652,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_fasttext_SIF1).to_csv('Data\QuoraDuplicateQuestions\quora_train_fasttext_SIF1.csv', index = False)



quora_train_fasttext_SIF2 = embedder.weighted_average_embedding_array('fasttext', quora_train.iloc[60652:121305,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_fasttext_SIF2).to_csv('Data\QuoraDuplicateQuestions\quora_train_fasttext_SIF2.csv', index = False)



quora_train_fasttext_SIF3 = embedder.weighted_average_embedding_array('fasttext', quora_train.iloc[121305:181957,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_fasttext_SIF3).to_csv('Data\QuoraDuplicateQuestions\quora_fasttext_fasttext_SIF3.csv', index = False)



quora_train_fasttext_SIF4 = embedder.weighted_average_embedding_array('fasttext', quora_train.iloc[181957:,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_train_fasttext_SIF4).to_csv('Data\QuoraDuplicateQuestions\quora_train_fasttext_SIF4.csv', index = False)





#VAL
quora_val_fasttext_SIF = embedder.weighted_average_embedding_array('fasttext', quora_val, 'question1', 'question2', question1_weight_dict, question2_weight_dict, vocabulary = vocab)
pd.DataFrame(quora_val_fasttext_SIF).to_csv('Data\QuoraDuplicateQuestions\quora_val_fasttext_SIF.csv', index = False)












