# Contextualized Word Embeddings in the Context of Record Linkage

This repository contains code for a dissertation project, which compared the use of contextualized (ELMo, BERT) and non-contextualized (Word2Vec, Fasttext) embeddings for the Record Linkage matching task in the presence of text data. The goal of Record Linkage is to join two of more datasets by connecting the entries of these datasets that pertain to the same real world entity. For this project, it was treated as a classification problem, in which a pair of observations, one from each dataset, has a corresponding label, match or non-match, that needs to be predicted.

The word in this project was inspired by the success of the contextualized word embeddings BERT (Google) and ELMo (AllenNLP) in other fields. This study conducted experiments to determine the under what data conditions and modeling architectures contextualized embeddings might be preferred to non-contextualized embeddings, like Fasttext and Word2Vec. Specifically, two different classifiers were used with input from the four different word embeddings models. One classifier relies on a weighted average over word embeddings, and uses a two layer Multi-Layer Perceptron with Highway (referred to as simple or SIF); the architecture was taken from the paper 'Deep Learning for Entity Matching: A Design Space Exploration'. The other takes the raw word embeddings as input to an LSTM Siamese Network (referred to as complex), and was inspired by the paper 'Learning Text Similarity with Siamese Recurrent Networks'. This project tested these models on two different datasets, Amazon-Google Products (https://dbs.uni-leipzig.de/de/research/projects/object_matching/benchmark_datasets_for_entity_resolution), and Quora Duplicate Questions (https://www.kaggle.com/c/quora-question-pairs/data).

This repository contains the following files:

### mynetworks.py
This file contains the Pytorch implementation of the two different classifier structures used.

### architectures.py
This file contains necessary functions and classes for training, including Pytorch Dataset classes for each type of classifier, alternative loss functions, and training and validation loops.

### embedder.py
This file contains helper functions for converting raw text into the inputs required of the two different classifiers.

### blocking.py
This file contains some very basic helping functions for tackling class imbalance problems (in this project, used only on the Amazon-Google dataset).

### quora_SIF_data_prep.py
This file is an example for how to produce the input representation for the two layer MLP with Highway.

### quora_SIF_evaluation.py
This file contains some sample analysis that was conducted on the results of the two layer MLP with Highway classifiers.

### experiment_simple.py, experiment_complex.py, training_simple.sh, training_complex.sh, arg_extractor.py
These files were used simply to train models on the cluster of GPUs that was used in this project.



