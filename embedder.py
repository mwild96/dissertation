#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:14:47 2019

@author: s1834310
"""

import numpy as np
#import pandas as pd
import nltk.tokenize 
import collections
import tqdm
#nltk.download('punkt')



def expand_vocabulary(original_vocab, df, text_column1, text_column2, embedding_dim):
    '''
    Add out-of-vocabulary words to the vocabulary of the pre-trained word embeddings
    (right now this function is designed specifically to work with KeyedVectors objects from the
    Gensim package for the original vocabulary)
    
    HAPPENS IN PLACE - NO RETURN
    
    '''
    
    column1_list = df[text_column1].apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
    column2_list = df[text_column2].apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
    
    df_vocab = list(set([item for sublist in column1_list for item in sublist] + \
                        [item for sublist in column2_list for item in sublist]))
    oov_words = [word for word in df_vocab if word not in original_vocab.vocab]

    oov_word_embeddings = []
    for i in tqdm.tqdm(range(len(oov_words))):
        oov_word_embeddings.append(np.random.rand(embedding_dim,))#less than a second

    original_vocab.add(oov_words, oov_word_embeddings)


def SIF_weights(df, text_column, a=1):
    '''
    NOTE THAT THIS IS SORT OF SPECIFIC TO WORD2VEC BECAUSE
    returns a dictionary of the weights for each 
    '''
    
    text_vocab = df[text_column].apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
    text_vocab = [item for sublist in text_vocab for item in sublist]
    
    freq_dict = dict(collections.Counter(text_vocab))
    n = len(list(set(text_vocab)))
    freq_dict.update({k:  a/(a + (freq_dict[k]/n)) for k in freq_dict.keys()})
    
    return freq_dict



def weighted_embedding_lookup(vocabulary, lst, weight_dict, embedding_dim):
    '''
    (currently designed to work with KeyedVector objects from the Gensim package for the vocabulary)
    
    words that do not appear in the vocabulary are effectively filtered out at validation/test time
    '''
    
    pw_list = []
    embeddings_list = []
    for i in range(len(lst)):
        if lst[i] == 0:
            pw_list.append(0)
            embeddings_list.append(np.zeros((embedding_dim,)))
        else:
            if lst[i] in weight_dict.keys(): 
                pw_list.append(weight_dict[lst[i]])
                embeddings_list.append(vocabulary[lst[i]])
                
            elif lst[i] not in weight_dict.keys() and lst[i] in vocabulary.vocab:
                #it could be in the overall vocab but not in the  vocab for that individual dataset
                #we can still give it a word embedding because the word embedding vectors are the same across datasets anyway
                #and we give it a weight of one because it's clearly rare for that dataset
                pw_list.append(1)
                embeddings_list.append(vocabulary[lst[i]])
                #note this we are effectively filtering out oov words (for validation purposes)
                
                
    return np.multiply(np.stack(embeddings_list).T, np.stack(pw_list))


    



def weighted_average_embedding_array(embed_type, vocabulary, df, text_column1, text_column2, weight_dict1, weight_dict2, max_tokens, embedding_dim, a=1):
    '''
    This function takes turns the text columns of a record linkage dataframe into an array of word embedding
    vectors from the pre-trained word2vec model through gensim, weighted according to the weighting scheme
    of the SIF model in the Design Space Exploration paper

    
    Input: 
    df: a dataframe (N,D) whose observations are pairs of observations from two separate original datasets that are 
    either a non-match or a match
    text_column1(2): the name, in string format, of the first (second) text column in df, to be converted to word
    embedding vectors
    a: the hyperparameter for the weighting scheme according to Design Space Exploration ("specifically, 
    the weights used to compute the average over the word embeddings for an input sequence are as follows: 
    given a word w the corresponding embedding is weighted by a weight f(w) = a/(a + p(w)) 
    where a is a hyperparameter and p(w) the normalized unigram frequency of w in the input corpus")
    max_tokens: for padding
    embedding_dim: the number of word embedding dimensions to retain
    
    Output:
    an array (N,2*embedding_dim)
    
    '''
    

    tqdm.tqdm.pandas()
    #tokenize
    if embed_type == 'word2vec':
        text1 = df[text_column1].apply(str).apply(nltk.tokenize.word_tokenize)
        text2 = df[text_column2].apply(str).apply(nltk.tokenize.word_tokenize)
    
    
    #padding
    text1 = text1.apply(lambda x: x + ['0'] * 
                        (max_tokens - len(x)) if len(x) < max_tokens else x[0:max_tokens])
    text2 = text2.apply(lambda x: x + ['0'] * 
                        (max_tokens - len(x)) if len(x) < max_tokens else x[0:max_tokens])
    
    
    #get weighted embeddings
    text1 = text1.progress_apply(lambda x: weighted_embedding_lookup(vocabulary, x, weight_dict1, 300))
    text2 = text2.apply(lambda x: weighted_embedding_lookup(vocabulary, x, weight_dict2, 300))
    
    
    #weighted average
    text1 = text1.progress_apply(np.mean, axis=1) 
    text2 = text2.apply(np.mean, axis=1)
    
    #format arrays
    array1 = np.stack(text1.values)
    array2 = np.stack(text2.values)
        
    
    return np.abs(array1-array2)











# =============================================================================
# word2vec_vocab = list(word2vec.vocab)
# word2vec_vocab.sort()
# for i in range(len(word2vec_vocab)):
#     if '_' not in word2vec_vocab[i]:
#         print(word2vec_vocab[i])
# 
# #looks like they did almost no pre-processing
# #replace groups of numbers with groups of hashtags
# #no lower casing even...
#         
#         
#         
# #check with the capitalization is like
# import re
# flat_list = [item for sublist in word2vec_vocab for item in sublist]
# len(re.findall('([A-Z][a-z]+)', ' '.join(word2vec_vocab)))/len(word2vec_vocab)
# re.findall('([a-z]+)', flat_list)
# len(re.findall('(_)', ' '.join(word2vec_vocab)))/len(word2vec_vocab) 
# re.match(r'\_', ' '.join(word2vec_vocab))        
# =============================================================================


#I guess I'm not worrying about the numbers that become # at the moment


# =============================================================================
# def filter_oov_word2vec(row):
#     return list(filter(lambda x: x in list(word2vec.vocab), row))
# 
    
# =============================================================================
# def word2vec_lookup(lst, embedding_dim):
#     word2vec_list = []
#     for i in range(len(lst)):
#         if lst[i] == 0:
#             word2vec_list.append(np.zeros((embedding_dim,)))
#         else:
#             word2vec_list.append(word2vec[lst[i]])
#     return np.stack(word2vec_list).T
# =============================================================================



# 
# 
# 
# def word2vec_array(df, text_column1, text_column2, max_tokens, embedding_dim):
# 
#     '''
#     This function turns the text columns of a record linkage dataframe into an array of word embedding
#     vectors from the pre-trained word2vec model through gensim
#     
#     Input: 
#     df: a dataframe (N,D) whose observations are pairs of observations from two separate original datasets that are 
#     either a non-match or a match
#     text_column1(2): the name, in string format, of the first (second) text column in df, to be converted to word
#     embedding vectors
#     max_tokens: for padding
#     embedding_dim: the number of word embedding dimensions to retain
#     
#     Output:
#     an array (N,2*embedding_dim)
#     
#     '''
#     
#     
#     #tokenize
#     text1 = df[text_column1].apply(nltk.tokenize.word_tokenize)
#     text2 = df[text_column2].apply(nltk.tokenize.word_tokenize)
#     
#     
#     #filter out oov words        
#     text1 = text1.apply(filter_oov_word2vec)
#     text2 = text2.apply(filter_oov_word2vec)
#     #^assign a random vector to oov
#     
#     #padding
#     text1 = text1.apply(lambda x: x + ['0'] * 
#                         (max_tokens - len(x)) if len(x) < max_tokens else x[0:max_tokens])
#     text2 = text2.apply(lambda x: x + ['0'] * 
#                         (max_tokens - len(x)) if len(x) < max_tokens else x[0:max_tokens])
#     
#     #get embeddings
#     text1 = text1.apply(lambda x: word2vec_lookup(x, 300))
#     text2 = text2.apply(lambda x: word2vec_lookup(x, 300))
#     
#     
#     #format array
#     array1 = np.stack(text1.apply(lambda x: 
#         np.reshape(x,(max_tokens*embedding_dim))).values)
#         
#     array2 = np.stack(text2.apply(lambda x: 
#         np.reshape(x,(max_tokens*embedding_dim))).values)
#     
#     return np.concatenate((array1, array2), axis = 1)

# 
# =============================================================================





























