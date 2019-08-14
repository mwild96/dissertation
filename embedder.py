#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains functions for manipulating word embeddings for the simple and complex classifier input representations.
"""

import numpy as np
import nltk.tokenize 
import gensim.utils
import gensim.models
import collections
import tqdm
import re
import torch
#nltk.download('punkt')


def build_vocabulary(df, text_column1, text_column2, embedding_dim, seed):
    '''
    This function builds a vocabulary and assigns corresponding (randomly initialized) word embeddings for a pair of 
    record linkage datasets, given the text field from each dataset.
    
    Inputs:
    df: a Pandas dataframe, containing pairs of observations from the two datasets
    text_column1: a string, the name of the column containing the raw text from the first dataset
    text_column2: a stirng, the name of the column containing the raw text from the second dataset
    embedding_dim: an int, the desired dimension of the randomized word vectors
    seed: an int, the RNG seed to be used for creating the randomized vectors
    
    Outputs:
    vocab: a dictionary of the words and their randomized embeddings
    '''    
    
    np.random.seed(seed)

    text1 = df[text_column1].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
    text2 = df[text_column2].apply(str).apply(lambda x: re.sub('n\'t', ' not', x))
    
    column1_list = text1.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
    column2_list = text2.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
    
    words = list(set([item for sublist in column1_list for item in sublist] + \
                        [item for sublist in column2_list for item in sublist]))

    word_embeddings = []
    for i in tqdm.tqdm(range(len(words))):
        word_embeddings.append(np.random.rand(embedding_dim,))
        
    vocab = dict(zip(words, word_embeddings))
    
    return vocab
    
def expand_vocabulary(original_vocab, df, text_column1, text_column2, embedding_dim):
    '''
    This function adds out-of-vocabulary words to the vocabulary of the pre-trained word embeddings
    
    **Note, this functions happens in place**
    
    Inputs:
    original_vocab: a gensim KeyedVectors object of the pre-trained embeddings
    df: a Pandas dataframe, containing pairs of observations from the two datasets
    text_column1: a string, the name of the column containing the raw text from the first dataset
    text_column2: a stirng, the name of the column containing the raw text from the second dataset
    embedding_dim: an int, the dimension of the pre-trained word embeddings
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
    This function determines the weights for the words in a corpus based on the paper 'Deep Learning for Entity Matching:
    A Design Space Exploration'
    
    "specifically, 
    the weights used to compute the average over the word embeddings for an input sequence are as follows: 
    given a word w the corresponding embedding is weighted by a weight f(w) = a/(a + p(w)) 
    where a is a hyperparameter and p(w) the normalized unigram frequency of w in the input corpus"

    Inputs:
    df: a Pandas dataframe, containing pairs of observations from the two datasets
    text_column: a string, the name of the text column whose entries collectively define the corpus
    a: the weighting parameter, default = 1
    
    Ouputs:
    freq_dict: a dictionary, containing all the words in the corpus and their associated weights
    '''
    
    text_vocab = df[text_column].apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
    text_vocab = [item for sublist in text_vocab for item in sublist]
    
    freq_dict = dict(collections.Counter(text_vocab))
    n = len(list(set(text_vocab)))
    freq_dict.update({k:  a/(a + (freq_dict[k]/n)) for k in freq_dict.keys()})
    
    return freq_dict



def weighted_embedding_lookup(embed_type, lst, weight_dict, tokenizer = None, vocabulary=None,  model=None):
    '''
    This function returns the weighted average of the word embeddings for a list of words. It is designed to operate 
    operate rowwise on a dataframe.
    
    Inputs:
    embed_type: a string, the type of word embedding being used
    lst: a list, containing the words to be transformed into the weighted average of word vectors
    weight_dict: a dictionary (created with the SIF_weights function), containing the weights for each word in the corpus
    tokenizer: when not None, a BERT Tokenizer object, only necessary when embed_type = 'bert'
    vocabulary: a gensim KeyedVectors object, the pretrained word embeddings, only necessary when embed_type = 'word2vec'
    or embed_type = 'fasttext' 
    model: a pytorch model object, either BERT and ELMo, containing the pre-trained embeddings of these models, 
    only necessary when embed_type = 'elmo' or embed_type = 'bert'
    
    Outputs:
    V: a numpy array, the weighted average over the word embeddings of the lst
    
    
    Note:
    for vocabulary based models: words that do not appear in the vocabulary are effectively filtered 
    out at validation/test time
    '''
    
    pw_list = []
    embeddings_list = []
    
    if vocabulary is not None:
        for i in range(len(lst)):
            if lst[i] in weight_dict.keys(): 
                pw_list.append(weight_dict[lst[i]])
                embeddings_list.append(vocabulary[lst[i]])
                
            elif lst[i] not in weight_dict.keys() and lst[i] in vocabulary.vocab:
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
            
            #note: BERT is trained on "combined token length" of <= 512 tokens
            if len(lst) > 512:
                lst = lst[0:512]
            
            for i in range(len(lst)):
                if lst[i] in weight_dict.keys():
                    pw_list.append(weight_dict[lst[i]])
                else: 
                    pw_list.append(1)            
            
            
            indexed_tokens = tokenizer.convert_tokens_to_ids(lst)

                
                
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segments_ids = [1]*len(indexed_tokens)
            
            # Convert inputs to PyTorch tensors
            segments_tensors = torch.tensor([segments_ids])
            tokens_tensor = torch.tensor([indexed_tokens])
            
            model.eval()

            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)[-2:]
            
            #we'll take the top four layers because that's what they suggest to do in the paper
            embeddings = np.stack(encoded_layers).squeeze()[8:12,:,:] 
            embeddings = np.mean(embeddings, axis=0)
            
            
            return np.multiply(embeddings.T, np.stack(pw_list))
            


def weighted_average_embedding_array(embed_type, df, text_column1, text_column2, weight_dict1, weight_dict2, tokenizer = None, vocabulary = None, model = None):
    '''
    This function turns the text columns of a record linkage dataframe into an array of word embedding
    vectors from one of four pre-trained models (word2vec, fasttext, elmo, bert), weighted according 
    to the weighting scheme of the SIF model in the paper 'Deep Learning for Entity Matching: A Design Space Exploration'
    
    Input: 
    embed_type: a string, the name of the embedding type to be used
    df: a Pandas dataframe, (N,D) whose observations are pairs of observations from two separate original datasets that are 
    either a non-match or a match
    text_column1: a string, the name of the first text column in df, to be converted to word embedding vectors
    text_column2: a string, the name of the second text column in df, to be converted to word embedding vectors
    weight_dict1: a dictionary (created by the function SIF_weights()), containing each word in the corpus defined by 
    text_column1 and its corresponding weight
    weight_dict2: a dictionary (created by the function SIF_weights()), containing each word in the corpus defined by 
    text_column2 and its corresponding weight
    tokenizer: when not None, a BERT Tokenizer object, only necessary when embed_type = 'bert'
    vocabulary: a gensim KeyedVectors object, the pretrained word embeddings, only necessary when embed_type = 'word2vec'
    or embed_type = 'fasttext' 
    model: a pytorch model object, either BERT and ELMo, containing the pre-trained embeddings of these models, 
    only necessary when embed_type = 'elmo' or embed_type = 'bert'
    
    Output:
    A: a numpy array of dimension (N,embedding_dim)
    
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
    
    
    #get weighted embeddings
    text1 = text1.progress_apply(lambda x: weighted_embedding_lookup(embed_type, x, weight_dict1, 
                                                                     tokenizer, vocabulary, model))
    text2 = text2.progress_apply(lambda x: weighted_embedding_lookup(embed_type, x, weight_dict2, 
                                                                     tokenizer, vocabulary, model))
    
    
    #weighted average
    text1 = text1.apply(np.mean, axis=1) 
    text2 = text2.apply(np.mean, axis=1)
    
    #format arrays
    array1 = np.stack(text1.values)
    array2 = np.stack(text2.values)
        
    
    return np.abs(array1-array2)



def word2index(lst, vocab, unk_index):
    '''
    This functions turn a list of words into a list of their corresponding indices for a pre-trained set of 
    word embedding vectors. (It is designed specifically for use with gensim KeyedVector objects, and for use with the
    complex classifier).
    
    Inputs:
    lst: a list, the words to be converted to their word embedding indices
    vocab: a gensim KeyedVectors object, containing the pre-trained word embeddings (from either fasttext, word2vec, or 
    the randomly initialized embeddings)
    unk_index: an int, the index to assign to unknown words
    '''
    
    for i in range(len(lst)):
        if lst[i] in vocab.vocab:
            lst[i] = vocab.vocab.get(lst[i]).index
        else:
            lst[i] = unk_index
    return lst



def bert_input_builder(string, tokenizer, max_tokens, padding = True):
    '''
    Inspired by: https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
    
    This function turns a string into the input required to retrieve the corresponding embeddings from the BERT model.
    
    Inputs:
    string: a string, text to be converted the BERT model input
    tokenizer: a BERT tokenizer object
    max_tokens: the maximum number of tokens to return as input to the BERT model
    padding: a boolean, if True, truncates or pads with zeros sequences to match the final length of max_tokens
    
    Outputs:
    indexed_tokens: a list, containing the BERT embedding indices of all tokens in string
    segments: a list, the same length as indexed_tokens, indicating which sentence, 1 or 2, the tokens correspond
    (for the next sentence prediction task, not really relevant here)
    input_mask: a list, the same length as indexed_tokens, with 0 for every occurence of the padding index (0) and 1, otherwise
    '''
    
    tokens = tokenizer.tokenize(string)
    if len(tokens) > max_tokens:
        tokens = tokens[0:max_tokens]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    segments = [0] * len(tokens) #<- everything should have the same segment id
    input_mask = [1] * len(tokens)
    
    
    if padding == True:
        padding = [0]*(max_tokens + 2 - len(tokens)) #don't include cls and sep as tokens
        indexed_tokens += padding
        segments += padding
        input_mask += padding
        
    return indexed_tokens, segments, input_mask




def restrict_w2v(w2v, restricted_word_set):
    '''
    This function was taken from 
    https://stackoverflow.com/questions/50914729/gensim-word2vec-select-minor-set-of-word-vectors-from-pretrained-model
    It is used to reduce gensim KeyedVectors objects to only the words that appear in the copora of the relevant 
    record linkage datasets (this was done purely because of RAM limitations on the GPU cluster used to train all 
    classifiers)
    
    **Note: this function operates in place**
    
    Inputs:
    w2v: a gensim KeyedVectors object, the set of word vectors to be reduced
    restricted_word_set: a list, the word and their associated vectors that should be retained
    '''
    
    
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    #new_vectors_norm = []

    for i in tqdm.tqdm(range(len(w2v.vocab))):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        #vec_norm = w2v.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            #new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = new_vectors
    w2v.index2entity = new_index2entity
    w2v.index2word = new_index2entity
    #w2v.vectors_norm = new_vectors_norm



#
def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    """
    This function was taken from https://stackoverflow.com/questions/45981305/convert-python-dictionary-to-word2vec-object
    It is used to turn a set of words and their vectors into a gensim KeyedVectors object. It is used here specifically 
    to create a gesim KeyedVectors object for the randomly initialized word vectors.
    
    Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.
    
    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).
    
    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with gensim.utils.smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype('float32')
                fout.write(gensim.utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(gensim.utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))
