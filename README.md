# Contextualized Word Embeddings in the Context of Record Linkage

This repository contains code for my dissertation project, which compared the use of contextualized (ELMo, BERT) and non-contextualized (Word2Vec, Fasttext) embeddings for the Record Linkage matching task in the presence of text data. The goal of Record Linkage is to join two of more datasets by connecting the entries of these datasets that pertain to the same real world entity. For this project, it was treated as a classification problem, in which a pair of observations, one from each dataset, has a corresponding label, match or non-match, that needs to be predicted.

The word in this project was inspired by the success of the contextualized word embeddings BERT (Google) and ELMo (AllenNLP) in other fields. This study conducted experiments to determine the under what data conditions and modeling architectures contextualized embeddings might be preferred to non-contextualized embeddings, like Fasttext and Word2Vec.

This repository contains the following files:

###
