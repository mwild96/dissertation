# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:38:29 2019

@author: Monica
"""

import gensim.downloader as api
import nltk.tokenize
import numpy as np
import random
import chardet
import pandas as pd
import collections
import re
import tqdm

fasttext = api.load('fasttext-wiki-news-subwords-300')
ft_vocab = list(fasttext.vocab)

quora = pd.read_csv('Data\QuoraDuplicateQuestions\questions.csv', header = 0, encoding = 'utf-8')

#do character ngrams help us at all in this fasttext implementation?
question1_list = quora['question1'].apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
question1_list = [item for sublist in question1_list for item in sublist]
question1_list = list(set(question1_list))
question1_list.sort()


question1_list.remove('Cheapism.com')
question1_list.remove('NitroFlare')
question1_list.remove('stock/real')
question1_list.remove('Sayl')
question1_list.remove('half-angel')
question1_list.remove('TubeMate')
question1_list.remove('12/20/16')
question1_list.remove('Aryans/Out')
question1_list.remove('Arzaylea')
question1_list.remove('Aschermann')
question1_list.remove('Asend')
question1_list.remove('Ashbys')



oov_counter = 0
oov_words = []
oov_ngram_counter = 0
oov_ngram_words = []

for i in tqdm.tqdm(range(len(question1_list))):
    oov_counter += 1
    oov_words.append(question1_list[i])
    if question1_list[i] not in ft_vocab:
        if question1_list[i] in fasttext.wv:
            oov_ngram_counter += 1
            oov_ngram_words.append(question1_list[i])
        

ft_vocab = list(fasttext.vocab)

random.shuffle(ft_vocab)

for i in range(10000):
    print(ft_vocab[i])
    


#again we have that thing where we have different embeddings for the capitalized and lowercase versions of the words
#one issue is with Google/Amazon/Walmart datasets I think all capitalization has already been eliminated
    
    
#example words
#over-valued
#зона  
#40-34
#Anna-Karin
#release.
#to--but
#2,354
#-small
#Asia
#Asia-
#Asia.
    
    
#check with the capitalization is like
import re
#flat_list = [item for sublist in ft_vocab for item in sublist]
re.findall('([A-Z][a-z]+)', ' '.join(ft_vocab))#it seems like capitalized words are largely name related but it's hard to tell for sure
re.findall('([a-z]+)', ' '.join(ft_vocab))

  
re.findall('\w+-\w+', ' '.join(ft_vocab))#there's lots os hyphened words in here    
re.findall('\w+_\w+', ' '.join(ft_vocab))#no words with underscores
#is there a way we can look at ngrams? how do I tell it to look for tokens that aren't words...
#I DON'T THINK USING THE SUBWORD VERSION OF THIS IS REALLY GOING TO HELP FOR US
#should we maybe try to turn oov words in subwords from the fasttext vocab?



ft_vocab.sort()
for i in range(10000):
    j = i+100000
    print(ft_vocab[-j])





   


#let's just take a look at what some oov words look like for fasttext


df = quora
text_column1 = 'question1'
text_column2 = 'question2'

tqdm.tqdm.pandas()

#take out the contractions ahead of time since we've started to do this
text1 = df[text_column1].apply(str).progress_apply(lambda x: re.sub('n\'t', ' not', x))
text2 = df[text_column2].apply(str).progress_apply(lambda x: re.sub('n\'t', ' not', x))

column1_list = text1.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
column2_list = text2.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()

    
df_vocab = list(set([item for sublist in column1_list for item in sublist] + \
                        [item for sublist in column2_list for item in sublist]))
oov_words = [word for word in df_vocab if word not in fasttext.vocab]#why is this taking so long for fasttext?
len(oov_words)/len(df_vocab)#~32% of the vocabulary of Quora is oov




#find the most frequently occuring words that are oov
text1 = df[text_column1].apply(str).progress_apply(lambda x: re.sub('n\'t', ' not', x))
text_complete1 = text1.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
text_complete1 = [item for sublist in text_complete1 for item in sublist]

text2 = df[text_column2].apply(str).progress_apply(lambda x: re.sub('n\'t', ' not', x))
text_complete2 = text2.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
text_complete2 = [item for sublist in text_complete2 for item in sublist]

text_complete = text_complete1 + text_complete2
text_complete_oov = [word for word in text_complete if word not in fasttext.vocab]
len(text_complete_oov)/len(text_complete)#about 1% of our corpus isn't being used...which really isn't bad



oov_freq = collections.Counter(text_complete_oov)
oov_freq.most_common(100)



#this is the most common one
#("n't", 21409)
#('``', 18507),
#("''", 18406),
#('/math', 1420)
21409/len(text_complete_oov)# 14.7% of the oov words (not that much...)
#this is definitely significant because nltk tokenizer tokenizes 'didn't' into 'did' & ''nt'
#and obviously only turning did into a word vector is problematic because didn't and did are opposites
#but then again if''nt' gets assigned it's own random word vector every time anyway...how much does it matter?

import re
len(re.findall('[a-z]+n\'t', ' '.join(text_complete), flags = re.I))
len(re.findall('[a-z]+n\'t ', ' '.join(text_complete), flags = re.I))
len(re.findall(' n\'t', ' '.join(text_complete), flags = re.I))
len(re.findall(' n\'t ', ' '.join(text_complete), flags = re.I))

import tqdm
tqdm.tqdm.pandas()

test = quora['question1'].apply(str).progress_apply(lambda x: re.sub('n\'t', ' not', x))#, count=0, flags=0

#check if it worked
test = test.apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
test = [item for sublist in test for item in sublist]
len(re.findall(' n\'t', ' '.join(test), flags = re.I))
#there's still 9945 left for some reason...
re.findall('\w+ n\'t \w+', ' '.join(test), flags = re.I)




re.sub(' n\'t', ' not', "ca n't get")
re.sub(' n\'t ', ' not', "ca n't get")
re.sub('(\w+ )n\'t( \w+)', '.\1not.\3', "ca n't get")














#NOW LET'S DO THE SAME THING BUT WITH AMAZON GOOGLE


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


#remove (unusuable)
null_indices = np.asarray(amaz_goog.loc[np.logical_or(pd.isnull(amaz_goog['description_az']), pd.isnull(amaz_goog['description_gg'])),].index)
amaz_goog.drop(index = null_indices, inplace = True)


#concatenate title and description together 
amaz_goog['text_az'] = amaz_goog['title'] + ' ' + amaz_goog['description_az']
amaz_goog['text_gg'] = amaz_goog['name'] + ' ' + amaz_goog['description_gg']



tqdm.tqdm.pandas()



tc1 = amaz_goog['text_az'].progress_apply(str).progress_apply(nltk.tokenize.word_tokenize).values.tolist()
#um well now this is saying it's going to take 4 hours...
tc1 = [item for sublist in tc1 for item in sublist]

tc2 = amaz_goog['text_gg'].apply(str).apply(nltk.tokenize.word_tokenize).values.tolist()
tc2 = [item for sublist in tc2 for item in sublist]

tc = tc1 + tc2
tc_oov = [word for word in tc if word not in fasttext.vocab]
len(tc_oov)/len(tc)#about 1% of our corpus isn't being used...which really isn't bad



oov_freq = collections.Counter(text_complete_oov)
oov_freq.most_common(100)





















def fasttext_preprocessing():
    


#oov_word_embeddings = []
#for i in tqdm.tqdm(range(len(oov_words))):
#    oov_word_embeddings.append(np.random.rand(embedding_dim,))#less than a second




















#https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
#THIS COULD HELP WITH word2vec

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
    
#he also defines a function that has to do with misspelled words
    
    
    
    
    
    
    
    
    



def filter_oov_fasttext(row):
    return list(filter(lambda x: x in list(fasttext.vocab), row))



def fasttext_array(df, text_column1, text_column2, max_tokens, embedding_dim):
    #tokenize
    text1 = df[text_column1].apply(nltk.tokenize.word_tokenize)
    text2 = df[text_column2].apply(nltk.tokenize.word_tokenize)
    
    
    
    #**************************************************************************
    #SHOULD WE ALSO TRY TO EXTRACT CHARACTER NGRAMS FROM THE DATA?
    #(tbh idk how many character ngrams this version of fasttext actually has...)
    #**************************************************************************
    
    
    
    #filter out oov words        
    text1 = text1.apply(filter_oov_fasttext)
    text2 = text2.apply(filter_oov_fasttext)

    
    #padding
    text1 = text1.apply(lambda x: x + ['0'] * 
      (max_tokens - len(x)) if len(x) < max_tokens else x[0:max_tokens])
    text2 = text2.apply(lambda x: x + ['0'] * 
      (max_tokens - len(x)) if len(x) < max_tokens else x[0:max_tokens])
    
    #get embeddings
    text1 = text1.apply(lambda x: fasttext[x])
    text2 = text2.apply(lambda x: fasttext[x])
    
    #format array
    array1 = np.stack(text1.apply(lambda x: 
        np.reshape(x,(max_tokens*embedding_dim))).values)
        
    array2 = np.stack(text2.apply(lambda x: 
        np.reshape(x,(max_tokens*embedding_dim))).values)
    
    return np.concatenate((array1, array2), axis = 1)























#some general exploration#




# =============================================================================
# fasttext_vocab = list(fasttext.vocab)
# fasttext_vocab.sort()
# #vocab of 999999 words
# 
# 
# for i in range(100000):
#      #if '_' not in fasttext_vocab[i]:
#      print(fasttext_vocab[i])
#  
# =============================================================================

#the character level of the fasttext model will not help us unless we're either
     #a. training with the fasttext model from scratch
     #b. separating our tokens into the appropriate character n_grams of this vocabulary
     #   ...which I have literally no idea how to even go about doing


#looks like they have both lower case and capitals in here  
    #e.g. fasttext['Ancient'] & fasttext['ancient']
    #but how we know for what words that's true and for what words it isn't I have no idea

#but this is contrary to what you would imagine to say the least

# =============================================================================
# np.mean(fasttext['Apple'] - fasttext['apple'])
# Out[122]: 0.0053287465
# 
# np.mean(fasttext['Apple'] - fasttext['car'])
# Out[123]: 0.00025482418
# =============================================================================
#you would expect the same words but with different capitalization to have 
#essentially the same vector...wouldn't you?


#Webster's Third New International Dictionary, Unabridged, together with its 1993 
#Addenda Section, includes some 470,000 entries. The Oxford English Dictionary, 
#Second Edition, reports that it includes a similar number.
#so both word2vec and fasttext have significantly more words than the standard english dictionary






# =============================================================================
# import re
# re.findall('(_)', ' '.join(fasttext_vocab)) #so it looks like FastText doesn't have words like that
# re.findall('([A-Z][a-z]+)', ' '.join(fasttext_vocab))
# re.findall('([a-z]+)', )
# 
# re.match(r'\_', ' '.join(fasttext_vocab))   
# 
# new = re.findall(r'\bNew\w+', ' '.join(fasttext_vocab))
# for i in range(200):
#     print(new[i])
# #we have some concatenations without underscores NewYork
# #idk how to really adjust our own corpus to account for ALL these types of things
# #maybe that's just one of the unavoidable issues with using pre-trained vectors
# #your stuck with the vocabulary they set up
#     
#     
# len(re.findall(r'\b[a-z]\w+', ' '.join(fasttext_vocab)))#426886
# len(re.findall(r'\b[A-Z]\w+', ' '.join(fasttext_vocab)))#608816
# 
# 
# =============================================================================

