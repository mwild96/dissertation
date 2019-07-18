#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.model_selection
import embedder
#import pytorch_pretrained_bert as bert
import pytorch_transformers as bert

#import data
quora = pd.read_csv('Data/QuoraDuplicateQuestions/questions.csv', header = 0, encoding = 'utf-8')

#train-test split
quora_train, quora_test, quora_labels_train, quora_labels_test = sklearn.model_selection.train_test_split(quora.loc[:, quora.columns != 'is_duplicate'], quora['is_duplicate'], test_size = 0.2, random_state = 100)

#train-val split
quora_train, quora_val, quora_labels_train, quora_labels_val = sklearn.model_selection.train_test_split(quora_train, quora_labels_train, test_size = 0.25, random_state = 100)




#prep
bert_tokenizer = bert.BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = bert.BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)    
question1_weight_dict = embedder.SIF_weights(quora_train, 'question1')
question2_weight_dict = embedder.SIF_weights(quora_train, 'question2') 



#TRAIN
quora_train_bert_SIF = embedder.weighted_average_embedding_array('bert', quora_train.iloc[0:24261,:], 'question1', 'question2', question1_weight_dict, question2_weight_dict, tokenizer = bert_tokenizer, vocabulary = None, model = bert_model)

#add labels
quora_train_bert_SIF = np.concatenate((quora_train_bert_SIF, quora_labels_train.iloc[0:24261].values.reshape(-1,1)), axis=1)

#export
pd.DataFrame(quora_train_bert_SIF).to_csv('Data/QuoraDuplicateQuestions/quora_train_bert_SIF1.csv', header = False, index = False)



#VAL
#quora_val_bert_SIF = embedder.weighted_average_embedding_array('bert', quora_val, 'question1', 'question2', question1_weight_dict, question2_weight_dict, tokenizer = bert_tokenizer, vocabulary = None, model = bert_model)

#add labels
#quora_val_bert_SIF = np.concatenate((quora_val_bert_SIF, quora_labels_val.values.reshape(-1,1)), axis=1)

#export
#pd.DataFrame(quora_val_bert_SIF).to_csv('Data/QuoraDuplicateQuestions/quora_val_bert_SIF.csv', header = False, index = False)


