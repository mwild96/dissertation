#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:52:10 2019

@author: s1834310


This file develops and outputs the word embedding arrays for the Amazon Google data set.

The process involves the following steps:
    original data import and merge
    blocking and undersampling of the negative class
    array of word2vec embeddings produced
    array of fasttext embeddings produced
    array of elmo embeddings produced
    array of bert embeddings produced


"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#nltk.download('stopwords')
import nltk.corpus
import chardet
import os
import tqdm


os.chdir('/afs/inf.ed.ac.uk/user/s18/s1834310/Documents/Dissertation')
#os.chdir(r'C:\Users\Monica\Documents\ORwDS MSc\Dissertation')


#import data
with open('Data/AmazonGoogle/Amazon.csv', 'rb') as f:
    result = chardet.detect(f.read())
    
amazon1 = pd.read_csv("Data/AmazonGoogle/Amazon.csv", header = 0, delimiter = ',', encoding = result['encoding'])
google = pd.read_csv("Data/AmazonGoogle/GoogleProducts.csv", header = 0, delimiter = ',', encoding = result['encoding'])
amaz_goog_matches = pd.read_csv("Data/AmazonGoogle/Amzon_GoogleProducts_perfectMapping.csv", header = 0, delimiter = ',')


#merge datasets: every possible combination of observations

#https://stackoverflow.com/questions/53907526/merge-dataframes-with-the-all-combinations-of-pks
amaz_goog = amazon1.assign(key=0).merge(google.assign(key=0), how='left', on = 'key', suffixes=('_az', '_gg'))
#should be 3226*1363 = 4397038 observations 
amaz_goog.drop(['key'], axis = 1, inplace = True)


#add labels 
amaz_goog = amaz_goog.merge(amaz_goog_matches.assign(match = 1), how = 'left', left_on = ['id_az', 'id_gg'], right_on=['idAmazon', 'idGoogleBase'] )
amaz_goog.drop(['idAmazon', 'idGoogleBase'], axis = 1, inplace = True)
amaz_goog.loc[amaz_goog['match'].isnull() == True,'match'] = 0


#check join
amaz_goog.shape[0] == amazon1.shape[0]*google.shape[0]
amaz_goog.loc[amaz_goog['match']==1,:].shape
amaz_goog_matches.shape
#good


#null values
amaz_goog.loc[np.logical_or(pd.isnull(amaz_goog['description_az']), pd.isnull(amaz_goog['description_gg'])),].shape[0]/amaz_goog.shape[0]
#13.9% of the data


#remove (unusuable)
null_indices = np.asarray(amaz_goog.loc[np.logical_or(pd.isnull(amaz_goog['description_az']), pd.isnull(amaz_goog['description_gg'])),].index)
amaz_goog.drop(index = null_indices, inplace = True)


#concatenate title and description together 
amaz_goog['text_az'] = amaz_goog['title'] + ' ' + amaz_goog['description_az']
amaz_goog['text_gg'] = amaz_goog['name'] + ' ' + amaz_goog['description_gg']






#Blocking#


##############################################################################################################
#THIS STUFF RIGHT HERE (all the deciding on the blocking, etc.) SHOULD REALLY BE PART OF EXPLORATORY ANALYSIS#
#THIS SHOULD ONLY CONTAIN THE IMPLEMENTATION OF THE BLOCKING SCHEME#

amaz_goog.loc[amaz_goog['match']==1,:].shape[0]/amaz_goog.shape[0]#0.00030652008617412243


#what is the capitalization like?






stopwords = nltk.corpus.stopwords.words('english')


def filter_stopwords(row):
    return list(filter(lambda word: word not in stopwords, row))


def number_shared_tokens(row):
    '''
    this is really a very poorly designed function because it's very specific...
    '''
    intersection = set(row['text_az']).intersection(set(row['text_gg']))
    return len(list(intersection))



def percentage_shared_tokens(row):
    joint = list(set(row['text_az'])) + list(set(row['text_gg'])) #should this be sets as well
    return row['number_shared_tokens']/len(joint)


# #***TEST THIS FUNCTION**#
# sample = amaz_goog.sample(n = 10)
# sample['text_az'] = sample['text_az'].str.split()
# sample['text_gg'] = sample['text_gg'].str.split()

# sample['text_gg'] = sample['text_gg'].apply(filter_stopwords)
# sample['text_az'] = sample['text_az'].apply(filter_stopwords)


# sample.apply(number_shared_tokens, axis = 1)
# #***********************************************************#



amaz_goog = amaz_goog.sample(frac = 0.5)


##APPLY FUNCTIONS##




amaz_goog['text_az'] = amaz_goog['text_az'].str.split()
amaz_goog['text_gg'] = amaz_goog['text_gg'].str.split()


tqdm.tqdm.pandas()


amaz_goog['text_az'] = amaz_goog['text_az'].progress_apply(filter_stopwords)
amaz_goog['text_gg'] = amaz_goog['text_gg'].progress_apply(filter_stopwords)


amaz_goog['number_shared_tokens'] = amaz_goog.progress_apply(number_shared_tokens, axis = 1)
amaz_goog['percentage_share_tokens'] = amaz_goog.progress_apply(percentage_shared_tokens, axis = 1)




##EXAMINE##


#observations with zero shared tokens
amaz_goog.loc[amaz_goog['number_shared_tokens']==0,:].shape[0]/amaz_goog.shape[0]#0.4806129345668061

amaz_goog.loc[np.logical_and(amaz_goog['match']==0, amaz_goog['number_shared_tokens']==0),:].shape[0]#910204

amaz_goog.loc[np.logical_and(amaz_goog['match']==1, amaz_goog['number_shared_tokens']==0),:].shape[0]#0


    
    


#remove observations with zero shared tokens
amaz_goog.drop(index = amaz_goog.loc[amaz_goog['number_shared_tokens']==0,:].index, inplace = True)

#new class balance
amaz_goog.loc[amaz_goog['match'] == 1, :].shape[0]/amaz_goog.shape[0]#0.0005967654701535934 
#we're making almost non-existent improvements


sns.kdeplot(amaz_goog.loc[amaz_goog['match'] == 1, 'number_shared_tokens'], color = 'r')
sns.kdeplot(amaz_goog.loc[amaz_goog['match'] == 0, 'number_shared_tokens'], color = 'b')
plt.show()




#observations with one shared token
amaz_goog.loc[np.logical_and(amaz_goog['match'] == 1, amaz_goog['number_shared_tokens'] == 1), :].count()
#there's only 6 rows that have one shared token and are also a match

amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens'] == 1), :].count()
#471659

#remove observations with one shared token
amaz_goog.drop(index = amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens']==1),:].index, inplace = True)
#new class balance
amaz_goog.loc[amaz_goog['match'] == 1, :].shape[0]/amaz_goog.shape[0]#0.0011011452691753515




#observations with two shared tokens
amaz_goog.loc[np.logical_and(amaz_goog['match'] == 1, amaz_goog['number_shared_tokens'] == 2), :].count()
#19

amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens'] == 2), :].count()
#226690

#remove observations with one shared token
amaz_goog.drop(index = amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens']==2),:].index, inplace = True)
amaz_goog.loc[amaz_goog['match'] == 1, :].shape[0]/amaz_goog.shape[0]#0.0019754539340954944





#observations with three shared tokens
amaz_goog.loc[np.logical_and(amaz_goog['match'] == 1, amaz_goog['number_shared_tokens'] == 3), :].count()#41
#there's only 6 rows that have one shared token and are also a match

amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens'] == 3), :].count()#117638


#remove observations with three shared tokens
amaz_goog.drop(index = amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens']==3),:].index, inplace = True)
amaz_goog.loc[amaz_goog['match'] == 1, :].shape[0]/amaz_goog.shape[0]#0.0033598227157375527





#observations with four shared tokens
amaz_goog.loc[np.logical_and(amaz_goog['match'] == 1, amaz_goog['number_shared_tokens'] == 4), :].count()#57


amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens'] == 4), :].count()#64710


#remove observations with four shared tokens
amaz_goog.drop(index = amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens']==4),:].index, inplace = True)
amaz_goog.loc[amaz_goog['match'] == 1, :].shape[0]/amaz_goog.shape[0]#0.005467447361278064



#observations with five shared tokens
amaz_goog.loc[np.logical_and(amaz_goog['match'] == 1, amaz_goog['number_shared_tokens'] == 5), :].count()
#there's only 6 rows that have one shared token and are also a match

amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens'] == 5), :].count()


#remove observations with four shared tokens
amaz_goog.drop(index = amaz_goog.loc[np.logical_and(amaz_goog['match'] == 0, amaz_goog['number_shared_tokens']==5),:].index, inplace = True)
amaz_goog.loc[amaz_goog['match'] == 1, :].shape[0]/amaz_goog.shape[0]





#look at the resulting distributions
sns.kdeplot(amaz_goog.loc[amaz_goog['match'] == 1, 'number_shared_tokens'], color = 'r')
sns.kdeplot(amaz_goog.loc[amaz_goog['match'] == 0, 'number_shared_tokens'], color = 'b')
plt.show()


#is this a reasonable thing to do? is it reasonable to eliminate only the negative?
#I'm like completely hand crafting this data set...







#what percentage of total tokens are shared
sns.kdeplot(amaz_goog['percentage_share_tokens'])
plt.show()

np.median(amaz_goog['percentage_share_tokens'])#0.03875968992248062
#percentage of shared tokens is small in general



sns.kdeplot(amaz_goog.loc[amaz_goog['match']==1,'percentage_share_tokens'], color = 'r')
sns.kdeplot(amaz_goog.loc[amaz_goog['match']==0,'percentage_share_tokens'], color = 'b')
plt.show()

#little bit more spread out for mathces (tends to be higher) but still a lot of matches with almost zero percent shared tokens


test = amaz_goog.loc[np.logical_and(amaz_goog['match']==1, amaz_goog['number_shared_tokens'] == 1),]

for i in range(test.shape[0]):
    print(test.loc[test.index[i], 'text_az'])
    print(' ')
    print(test.loc[test.index[i], 'text_gg'])
    print('.')
    print('.')
    
    
    
    
    
#look at oov statistics
az_vocab = list(set([item for sublist in description_az.values.tolist() for item in sublist]))
oov = [x for x in az_vocab if x not in list(word2vec.vocab)]

len(oov)/len(az_vocab)


#look at oov at the obersvation level
def percentage_oov(row):
    return len([x for x in row if x in oov])/len(row)

description_az_perct_oov = description_az.apply(percentage_oov)

sns.kdeplot(description_az_perct_oov)
plt.show()

np.median(description_az_perct_oov)



sns.kdeplot(description_az_perct_oov*description_az.str.len())
plt.show()

np.median(description_az_perct_oov*description_az.str.len())


##############################################################################################################



#so what if we concatenate description and title? then maybe we can eliminate this problem of minimal shared tokens?

#another significant advantage with gensim is: it lets you handle large text files without having to load the entire file in memory.


































#filter out out of vocabulary words
def filter_oov(list_a):
    return [x for x in list_a if x[0] in list(word2vec.vocab)]


decription_az = description_az.apply(filter_oov)
#it's taking over an hour to run this btw











#get word embeddings for all words
description_az.apply(lambda x: word2vec[x])








import word2vec







#word2vec_df(df, text_column1, text_column2, max_tokens)
word2vec.word2vec_array()








##OTHER PREPROCESSING##


import re
re.findall('([A-Z][a-z]+)', ' '.join(amaz_goog['text_az']))#no capitalization?
re.findall('([A-Z][a-z]+)', ' '.join(amaz_goog['text_gg']))#no capitalization...

















