# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:15:02 2019
@author: Monica
"""

import tqdm
import nltk
import nltk.corpus
import pandas as pd

stopwords = nltk.corpus.stopwords.words('english')


def filter_stopwords(row):    
    return list(filter(lambda word: word not in stopwords, row))


def number_shared_tokens(row, text_column1, text_column2):
    
    intersection = set(row[text_column1]).intersection(set(row[text_column2]))
    return len(list(intersection))



def percentage_shared_tokens(row, text_column1, text_column2):
    
    intersection = set(row[text_column1]).intersection(set(row[text_column2]))
    joint = list(set(row[text_column1])) + list(set(row[text_column2])) #should this be sets as well
    return len(list(intersection))/len(joint)


def shared_tokens_blocking(df, text_column1, text_column2, shared_tokens_threshold):
    
    text1 = df[text_column1].str.split()
    text2 = df[text_column2].str.split()
    
    text1.dropna(inplace = True)
    text2.dropna(inplace = True)

    tqdm.tqdm.pandas()
    
    text1 = text1.progress_apply(lambda x: [item for item in x if item not in stopwords])
    text2 = text2.progress_apply(lambda x: [item for item in x if item not in stopwords])
    #text1 = text1.progress_apply(filter_stopwords)
    #text2 = text2.progress_apply(filter_stopwords)

    text = pd.concat([text1, text2], axis = 1)
    #SHOULD WE ALSO TOKENIZE? OR NO BECAUSE THAT'S SPECIFIC TO THE MODEL WE'RE USING
    
    
    text['shared_tokens'] = text.progress_apply(lambda x: number_shared_tokens(x, text_column1, text_column2), axis = 1)
    df = df.loc[text['shared_tokens']>shared_tokens_threshold,:]
    
    return df



def percentage_shared_tokens_blocking(df, text_column1, text_column2, shared_tokens_threshold):
    
    text1 = df[text_column1].str.split()
    text2 = df[text_column2].str.split()
    
    text1.dropna(inplace=True)
    text2.dropna(inplace=True)
    
    tqdm.tqdm.pandas()
    
    text1 = text1.progress_apply(lambda x: [item for item in x if item not in stopwords])
    text2 = text2.progress_apply(lambda x: [item for item in x if item not in stopwords])

    text = pd.concat([text1, text2], axis = 1)
    #SHOULD WE ALSO TOKENIZE? OR NO BECAUSE THAT'S SPECIFIC TO THE MODEL WE'RE USING
    
    
    text['shared_tokens'] = text.progress_apply(lambda x: percentage_shared_tokens(x, text_column1, text_column2), axis = 1)
    #amaz_goog['percentage_share_tokens'] = amaz_goog.progress_apply(percentage_shared_tokens, axis = 1)
    df = df.loc[text['shared_tokens']>shared_tokens_threshold,:]
    
    return df