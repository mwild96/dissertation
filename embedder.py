#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:14:47 2019
@author: s1834310
"""

import numpy as np
import nltk.tokenize 
import collections
import tqdm
import re
import torch
#nltk.download('punkt')


def build_vocabulary(df, text_column1, text_column2, embedding_dim):
    '''
    This function builds a vocabulary and corresponding (randomly initialized) word embeddings for a pair of 
    record linkage datasets, given the text field from each dataset.
    '''
    
    text1 = df[text_column1].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
    text2 = df[text_column2].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
    
    column1_list = text1.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
    column2_list = text2.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
    
    df_vocab = list(set([item for sublist in column1_list for item in sublist] + \
                        [item for sublist in column2_list for item in sublist]))

    word_embeddings = []
    for i in tqdm.tqdm(range(len(words))):
        word_embeddings.append(np.random.rand(embedding_dim,))
        
    vocab = zip(words, word_embeddings)
    
    return vocab
    
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



def weighted_embedding_lookup(embed_type, lst, weight_dict, tokenizer = None, vocabulary=None,  model=None):
    '''
    this function is designed to operate on a single list of words to be embedded (so over a dataframe, it should be applied rowwise)
    
    for vocabulary based models: words that do not appear in the vocabulary are effectively filtered out at validation/test time
    '''
    
    pw_list = []
    embeddings_list = []
    
    if vocabulary is not None:
        for i in range(len(lst)):
            #we are eliminating this because we are deciding that we're not going to do padding
# =============================================================================
#             if lst[i] == 0:
#                 pw_list.append(0)
#                 embeddings_list.append(np.zeros((embedding_dim,)))
#             else:
# =============================================================================
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
        
    if model is not None:
        if embed_type == "elmo":
            ''' wait how will I deal with the fact that an entire setence gets input here, in terms of the padding we did?
            it looks like zero always gets the vector on each layer regardless of where the zero turns up (which makes sense
            because there is no polysemy with )
            '''
            
            for i in range(len(lst)):
                if lst[i] in weight_dict.keys():
                    pw_list.append(weight_dict[lst[i]])
                else: 
                    pw_list.append(1)
                    
            embeddings = model.embed_sentence(lst)
            embeddings = np.mean(embeddings, axis=0)
            
            return np.multiply(embeddings.T, np.stack(pw_list))
        
        if embed_type == 'bert':
            
            for i in range(len(lst)):
                if lst[i] in weight_dict.keys():
                    pw_list.append(weight_dict[lst[i]])
                else: 
                    pw_list.append(1)            
            
            
            indexed_tokens = tokenizer.convert_tokens_to_ids(lst)
        
            #note: BERT is trained on "combined token length" of <= 512 tokens
            if len(indexed_tokens) > 512:
                indexed_tokens = indexed_tokens[0:513]
                
                
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segments_ids = [1]*len(indexed_tokens)
            
            # Convert inputs to PyTorch tensors
            segments_tensors = torch.tensor([segments_ids])
            tokens_tensor = torch.tensor([indexed_tokens])
            
            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)
            
            #we'll take the top four layers because that's what they suggest to do in the paper
            embeddings = np.stack(encoded_layers).squeeze()[8:,:,:] 
            embeddings = np.mean(embeddings, axis=0)
            
            
            return np.multiply(embeddings.T, np.stack(pw_list))
            


def weighted_average_embedding_array(embed_type, df, text_column1, text_column2, weight_dict1, weight_dict2, tokenizer = None, vocabulary = None, model = None, a=1):
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
    an array (N,embedding_dim)
    
    '''
    

    tqdm.tqdm.pandas()
    #tokenize
    if embed_type != 'bert':
        text1 = df[text_column1].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
        text2 = df[text_column2].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
        
        text1 = text1.apply(str).apply(nltk.tokenize.word_tokenize)
        text2 = text2.apply(str).apply(nltk.tokenize.word_tokenize)
        
    elif embed_type == 'bert':
        
        text1 = df[text_column1].progress_apply(lambda x: "[CLS] " + str(x) + " [SEP]").apply(tokenizer.tokenize)
        text2 = df[text_column2].progress_apply(lambda x: "[CLS] " + str(x) + " [SEP]").apply(tokenizer.tokenize)
    
    
    #NO PADDING IS NECESSARY WHEN WE'RE JUST TAKING THE AVERAGE 
# =============================================================================
#     text1 = text1.apply(lambda x: x + ['0'] * 
#                         (max_tokens - len(x)) if len(x) < max_tokens else x[0:max_tokens])
#     text2 = text2.apply(lambda x: x + ['0'] * 
#                         (max_tokens - len(x)) if len(x) < max_tokens else x[0:max_tokens])
#     
# =============================================================================
    
    
    #get weighted embeddings
    text1 = text1.progress_apply(lambda x: weighted_embedding_lookup(embed_type, x, weight_dict1, tokenizer, vocabulary, model))
    text2 = text2.progress_apply(lambda x: weighted_embedding_lookup(embed_type, x, weight_dict2, tokenizer, vocabulary, model))
    
    
    #weighted average
    text1 = text1.apply(np.mean, axis=1) 
    text2 = text2.apply(np.mean, axis=1)
    
    #format arrays
    array1 = np.stack(text1.values)
    array2 = np.stack(text2.values)
        
    
    return np.abs(array1-array2)


def word2index(lst, vocab):
    for i in range(len(lst)):
        lst[i] = vocab.vocab.get(lst[i]).index
    return lst
