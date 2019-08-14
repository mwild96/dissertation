#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This file contains functions designed for 'blocking' - typically used in record linkage to group similar observations that may
constitute pairs together; here, used mainly to decrease the size of the dataset by eliminating pairs that need to be
evaluated.
'''

import tqdm
import nltk
import nltk.corpus
import pandas as pd

stopwords = nltk.corpus.stopwords.words('english')

def number_shared_tokens(row, text_column1, text_column2):
    '''
    This function is designed to determine the number of tokens shared between two lists of words contained in the 
    columns of a dataframe.
    
    Inputs:
    row: the row of the dataframe whose columns are to be evaluted
    text_column1: a string, the name of the column containing lists of words from the first dataset
    text_column2: a string, the name of the column containing lists of words from the second dataset
    
    Ouputs:
    l: the number of tokens shared between the two lists of words 
    '''
    
    intersection = set(row[text_column1]).intersection(set(row[text_column2]))
    return len(list(intersection))



def percentage_shared_tokens(row, text_column1, text_column2):
    '''
    This function is designed to determine the percentage of tokens that are shared between two lists of words contained in the 
    columns of a dataframe.
    
    Inputs:
    row: the row of the dataframe whose columns are to be evaluted
    text_column1: a string, the name of the column containing lists of words from the first dataset
    text_column2: a string, the name of the column containing lists of words from the second dataset
    
    Ouputs:
    p: the percentage of tokens shared between the two lists of words 
    
    '''
    intersection = set(row[text_column1]).intersection(set(row[text_column2]))
    joint = list(set(row[text_column1])) + list(set(row[text_column2]))
    return len(list(intersection))/len(joint)


def shared_tokens_blocking(df, text_column1, text_column2, shared_tokens_threshold):
    '''
    This function is designed to reduce the size of a dataframe by eliminating pairs of observations that have fewer than
    or equal to a certain number of shared tokens.
    
    Inputs:
    df: a Pandas dataframe, the dataframe to be reduced
    text_column1: a string, the name of the column containing text from the first dataset 
    text_column2: a string, the name of the column containing text from the second dataset
    shared_tokens_threshold: an int; pairs that share <= this value tokens will be removed from the dataset
    
    Outputs:
    df: a Pandas dataframe object, the new dataframe in which pairs with <= shared_tokens_threshold tokens shared have been
    eliminated
    '''
    
    
    text1 = df[text_column1].str.split()
    text2 = df[text_column2].str.split()
    
    text1.dropna(inplace = True)
    text2.dropna(inplace = True)

    tqdm.tqdm.pandas()
    
    text1 = text1.progress_apply(lambda x: [item for item in x if item not in stopwords])
    text2 = text2.progress_apply(lambda x: [item for item in x if item not in stopwords])

    text = pd.concat([text1, text2], axis = 1)
    
    
    text['shared_tokens'] = text.progress_apply(lambda x: number_shared_tokens(x, text_column1, text_column2), axis = 1)
    df = df.loc[text['shared_tokens']>shared_tokens_threshold,:]
    
    return df



def percentage_shared_tokens_blocking(df, text_column1, text_column2, shared_tokens_threshold):
    '''
    This function is designed to reduce the size of a dataframe by eliminating pairs of observations that have less than
    or equal to a certain percentage of their tokens shared.
    
    Inputs:
    df: a Pandas dataframe, the dataframe to be reduced
    text_column1: a string, the name of the column containing text from the first dataset 
    text_column2: a string, the name of the column containing text from the second dataset
    shared_tokens_threshold: a double; pairs that share <= this percentage tokens will be removed from the dataset
    
    Outputs:
    df: a Pandas dataframe object, the new dataframe in which pairs with <= shared_tokens_threshold % of tokens shared 
    have been eliminated
    '''
    
    
    text1 = df[text_column1].str.split()
    text2 = df[text_column2].str.split()
    
    text1.dropna(inplace=True)
    text2.dropna(inplace=True)
    
    tqdm.tqdm.pandas()
    
    text1 = text1.progress_apply(lambda x: [item for item in x if item not in stopwords])
    text2 = text2.progress_apply(lambda x: [item for item in x if item not in stopwords])

    text = pd.concat([text1, text2], axis = 1)
    
    
    text['shared_tokens'] = text.progress_apply(lambda x: percentage_shared_tokens(x, text_column1, text_column2), axis = 1)
    df = df.loc[text['shared_tokens']>shared_tokens_threshold,:]
    
    return df
