# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:16:41 2019

@author: Monica
"""

import tqdm
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader as api
import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import nltk.tokenize
import torch
import mynetworks



val_raw = pd.read_csv('Data/QuoraDuplicateQuestions/quora_val.csv')



#RUN MODELS AND GET STATS##


#word2vec#
word2vec_net = mynetworks.HighwayReluNet(300)
checkpoint = torch.load('Models/quora_word2vec_SIF14.pth',map_location='cpu')
word2vec_net.load_state_dict(checkpoint['model_state_dict'])
word2vec_net.eval()

word2vec_val = pd.read_csv('Data/QuoraDuplicateQuestions/quora_val_word2vec_SIF.csv', header = None)

with torch.no_grad():
    word2vec_val['outputs'] = word2vec_net(torch.tensor(word2vec_val.iloc[:,0:300].values).float()).numpy()

word2vec_val['predicted'] = torch.round(torch.tensor(word2vec_val['outputs'].values)).numpy()

 
word2vec_fn_indices = word2vec_val.loc[word2vec_val[300] != word2vec_val['predicted'],:].loc[word2vec_val['predicted']==0,:].index
word2vec_fp_indices = word2vec_val.loc[word2vec_val[300] != word2vec_val['predicted'],:].loc[word2vec_val['predicted']==1,:].index
word2vec_tp_indices = word2vec_val.loc[word2vec_val[300] == word2vec_val['predicted'],:].loc[word2vec_val['predicted']==1,:].index


word2vec_cfn_indices = word2vec_val.loc[np.logical_and(word2vec_val[300] != word2vec_val['predicted'], word2vec_val[300] == 1),:].loc[word2vec_val['outputs']<=0.1,:].index
word2vec_cfp_indices = word2vec_val.loc[np.logical_and(word2vec_val[300] != word2vec_val['predicted'], word2vec_val[300] == 0),:].loc[word2vec_val['outputs']>=0.9,:].index





#fasttext#
fasttext_net = mynetworks.HighwayReluNet(300)
checkpoint = torch.load('Models/quora_fastext_SIF14.pth',map_location='cpu')
fasttext_net.load_state_dict(checkpoint['model_state_dict'])
fasttext_net.eval()

fasttext_val = pd.read_csv('Data/QuoraDuplicateQuestions/quora_val_fasttext_SIF.csv', header = None)

with torch.no_grad():
    fasttext_val['outputs'] = fasttext_net(torch.tensor(fasttext_val.iloc[:,0:300].values).float()).numpy()

fasttext_val['predicted'] = torch.round(torch.tensor(fasttext_val['outputs'].values)).numpy()

 
fasttext_fn_indices = fasttext_val.loc[fasttext_val[300] != fasttext_val['predicted'],:].loc[fasttext_val['predicted']==0,:].index
fasttext_fp_indices = fasttext_val.loc[fasttext_val[300] != fasttext_val['predicted'],:].loc[fasttext_val['predicted']==1,:].index
fasttext_tp_indices = fasttext_val.loc[fasttext_val[300] == fasttext_val['predicted'],:].loc[fasttext_val['predicted']==1,:].index


fasttext_cfn_indices = fasttext_val.loc[np.logical_and(fasttext_val[300] != fasttext_val['predicted'], fasttext_val[300] == 1),:].loc[fasttext_val['outputs']<=0.1,:].index
fasttext_cfp_indices = fasttext_val.loc[np.logical_and(fasttext_val[300] != fasttext_val['predicted'], fasttext_val[300] == 0),:].loc[fasttext_val['outputs']>=0.9,:].index







#elmo#
elmo_net = mynetworks.HighwayReluNet(1024)
checkpoint = torch.load('Models/quora_elmo_SIF14.pth',map_location='cpu')
elmo_net.load_state_dict(checkpoint['model_state_dict'])
elmo_net.eval()


elmo_val = pd.read_csv('Data/QuoraDuplicateQuestions/quora_val_elmo_SIF.csv', header = 0)


with torch.no_grad():
    elmo_val['outputs'] = elmo_net(torch.tensor(elmo_val.iloc[:,0:1024].values).float()).numpy()

elmo_val['predicted'] = torch.round(torch.tensor(elmo_val['outputs'].values)).numpy()


elmo_fn_indices = elmo_val.loc[elmo_val['1024'] != elmo_val['predicted'],:].loc[elmo_val['predicted']==0,:].index
elmo_fp_indices = elmo_val.loc[elmo_val['1024'] != elmo_val['predicted'],:].loc[elmo_val['predicted']==1,:].index
elmo_tp_indices = elmo_val.loc[elmo_val['1024'] == elmo_val['predicted'],:].loc[elmo_val['predicted']==1,:].index

elmo_cfn_indices = elmo_val.loc[np.logical_and(elmo_val['1024'] != elmo_val['predicted'], elmo_val['1024'] == 1),:].loc[elmo_val['outputs']<=0.1,:].index
elmo_cfp_indices = elmo_val.loc[np.logical_and(elmo_val['1024'] != elmo_val['predicted'], elmo_val['1024'] == 0),:].loc[elmo_val['outputs']>=0.9,:].index





#bert#
bert_net = mynetworks.HighwayReluNet(768)
checkpoint = torch.load('Models/quora_bert_SIFregularized00114.pth',map_location='cpu')#regularized001
bert_net.load_state_dict(checkpoint['model_state_dict'])
bert_net.eval()

bert_val = pd.read_csv('Data/QuoraDuplicateQuestions/quora_val_bert_SIF.csv', header = None)

with torch.no_grad():
    bert_val['outputs'] = bert_net(torch.tensor(bert_val.iloc[:,0:768].values).float()).numpy()

bert_val['predicted'] = torch.round(torch.tensor(bert_val['outputs'].values)).numpy()


bert_fn_indices = bert_val.loc[bert_val[768] != bert_val['predicted'],:].loc[bert_val['predicted']==0,:].index
bert_fp_indices = bert_val.loc[bert_val[768] != bert_val['predicted'],:].loc[bert_val['predicted']==1,:].index
bert_tp_indices = bert_val.loc[bert_val[768] == bert_val['predicted'],:].loc[bert_val['predicted']==1,:].index

bert_cfn_indices = bert_val.loc[np.logical_and(bert_val[768] != bert_val['predicted'], bert_val[768] == 1),:].loc[bert_val['outputs']<=0.1,:].index
bert_cfp_indices = bert_val.loc[np.logical_and(bert_val[768] != bert_val['predicted'], bert_val[768] == 0),:].loc[bert_val['outputs']>=0.9,:].index










#PERCENTAGE OOV#

def percentage_oov(string, vocab):
    lst = nltk.tokenize.word_tokenize(re.sub('n\'t', ' not', str(string)))
    oov = [word for word in lst if word not in vocab.vocab]
    return len(oov)/len(lst)


val_raw['complete_text'] = val_raw['question1'] + ' ' + val_raw['question2'] 


tqdm.tqdm.pandas()

#word2vec#
word2vec = api.load('word2vec-google-news-300')

val_raw['word2vec_perct_oov_complete'] = val_raw['complete_text'].progress_apply(lambda x: percentage_oov(x, word2vec))

sns.kdeplot(val_raw.loc[~val_raw.index.isin(word2vec_fn_indices.tolist() + word2vec_fp_indices.tolist()),'word2vec_perct_oov_complete'], color = 'b', label = 'correct predictions')
sns.kdeplot(val_raw.loc[val_raw.index.isin(word2vec_fn_indices.tolist() + word2vec_fp_indices.tolist()),'word2vec_perct_oov_complete'], color = 'r', label = 'incorrect predictions')
plt.xlabel('Text Out-of-Vocabulary Percentage (Both Questions)')
plt.ylabel('Density')
plt.show()


#fasttext#
fasttext = api.load('fasttext-wiki-news-subwords-300')

val_raw['fasttext_perct_oov_complete'] = val_raw['complete_text'].progress_apply(lambda x: percentage_oov(x, word2vec))

sns.kdeplot(val_raw.loc[~val_raw.index.isin(fasttext_fn_indices.tolist() + fasttext_fp_indices.tolist()),'fasttext_perct_oov_complete'], color = 'b', label = 'correct predictions')
sns.kdeplot(val_raw.loc[val_raw.index.isin(fasttext_fn_indices.tolist() + fasttext_fp_indices.tolist()),'fasttext_perct_oov_complete'], color = 'r', label = 'incorrect predictions')
plt.xlabel('Text Out-of-Vocabulary Percentage (Both Questions)')
plt.ylabel('Density')
plt.show()





wn.synsets('unusual')

#MEANING CONFLATION DEFICIENCY#
#https://stackoverflow.com/questions/22016273/list-of-polysemy-words
def polysemous_count(lst):
    count = 0
    for word in lst:
        if(len(wn.synsets(word)) > 1):
            count += 1
    return count 


tqdm.tqdm.pandas()

val_raw.dropna(subset = ['question2'], inplace = True)
val_raw['polysemous_q1'] = val_raw['question1'].str.split().progress_apply(polysemous_count)
val_raw['polysemous_q2'] = val_raw['question2'].str.split().progress_apply(polysemous_count)
val_raw['polysemous_total'] = val_raw['polysemous_q1'] + val_raw['polysemous_q2']

context_wrong_indices = list(set((bert_fn_indices.tolist() + elmo_fn_indices.tolist() + bert_fp_indices.tolist() + elmo_fp_indices.tolist() )))
non_context_wrong_indices = list(set((word2vec_fn_indices.tolist() + fasttext_fn_indices.tolist() + word2vec_fp_indices.tolist() + fasttext_fp_indices.tolist() )))
context_only_wrong_indices = [idx for idx in context_wrong_indices if idx not in non_context_wrong_indices]
non_context_only_wrong_indices = [idx for idx in non_context_wrong_indices if idx not in context_wrong_indices]

sns.kdeplot(val_raw.iloc[context_only_wrong_indices,:]['polysemous_total'], color = 'r', label = 'context wrong')
sns.kdeplot(val_raw.iloc[non_context_only_wrong_indices,:]['polysemous_total'], color = 'b', label = 'non-context wrong')
plt.xlabel('Number of Polysemous Words (Both Questions)')
plt.ylabel('Density')
plt.show()



context_cwrong_indices = list(set((bert_cfn_indices.tolist() + elmo_cfn_indices.tolist() + bert_cfp_indices.tolist() + elmo_cfp_indices.tolist() )))
non_context_cwrong_indices = list(set((word2vec_cfn_indices.tolist() + fasttext_cfn_indices.tolist() + word2vec_cfp_indices.tolist() + fasttext_cfp_indices.tolist() )))
context_only_cwrong_indices = [idx for idx in context_cwrong_indices if idx not in non_context_wrong_indices]
non_context_only_cwrong_indices = [idx for idx in non_context_cwrong_indices if idx not in context_wrong_indices]

sns.kdeplot(val_raw.iloc[context_only_cwrong_indices,:]['polysemous_total'], color = 'r', label = 'context confident & wrong')
sns.kdeplot(val_raw.iloc[non_context_only_cwrong_indices,:]['polysemous_total'], color = 'b', label = 'non-context confident & wrong')
plt.xlabel('Number of Polysemous Words (Both Questions)')
plt.ylabel('Density')
plt.show()




