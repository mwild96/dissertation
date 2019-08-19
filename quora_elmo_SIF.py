#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.model_selection
import embedder
import allennlp.commands.elmo as elmo


#import data
quora = pd.read_csv('Data/QuoraDuplicateQuestions/questions.csv', header = 0, encoding = 'utf-8')

#train-test split
quora_train, quora_test, quora_labels_train, quora_labels_test = sklearn.model_selection.train_test_split(quora.loc[:, quora.columns != 'is_duplicate'], quora['is_duplicate'], test_size = 0.2, random_state = 100)

#train-val split
quora_train, quora_val, quora_labels_train, quora_labels_val = sklearn.model_selection.train_test_split(quora_train, quora_labels_train, test_size = 0.25, random_state = 100)



#prep
elmo_model = elmo.ElmoEmbedder()
question1_weight_dict = embedder.SIF_weights(quora_train, 'question1') #I THINK IT'S VERY IMPORTANT THAT TRAIN GOES THROUGH THIS
question2_weight_dict = embedder.SIF_weights(quora_train, 'question2') #BUT REALIZE THAT WILL CREATE PROBLEMS WITH VAL



#TRAIN
#quora_train_elmo_SIF = embedder.weighted_average_embedding_array('elmo', quora_train.iloc[194080:,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, tokenizer = None, vocabulary = None, model = elmo_model)

#add labels
#quora_train_elmo_SIF = np.concatenate((quora_train_elmo_SIF, quora_labels_train.iloc[194080:].values.reshape(-1,1)), axis=1)

#export
#pd.DataFrame(quora_train_elmo_SIF).to_csv('Data/QuoraDuplicateQuestions/quora_train_elmo_SIF13.csv', header = False, index = False)



#VALIDATION
quora_val_elmo_SIF = embedder.weighted_average_embedding_array('elmo', quora_val.iloc[40000:,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, tokenizer = None, vocabulary = None, model = elmo_model)

#add labels
quora_val_elmo_SIF = np.concatenate((quora_val_elmo_SIF, quora_labels_val.iloc[40000:].values.reshape(-1,1)), axis=1)

#export
pd.DataFrame(quora_val_elmo_SIF).to_csv('Data/QuoraDuplicateQuestions/quora_val_elmo_SIF2.csv', header = False, index = False)
