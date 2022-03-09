import pandas as pd
import nltk
import re
import numpy as np
import heapq
from nltk.corpus import stopwords
import PyPDF2
import pdfplumber
from sklearn.metrics import f1_score
import os
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


def load_pos(positive_wordbank):
    df_pos = pd.read_csv(positive_wordbank, header=None)
    df_pos = df_pos.drop(1, axis=1)
    df_pos.columns = ['positive_words']
    df_pos['positive_words'] = df_pos['positive_words'].apply(lambda x: x.lower())
    return df_pos

def load_neg(negative_wordbank):
    df_neg = pd.read_csv(negative_wordbank, header=None)
    df_neg = df_neg.drop(1, axis=1)
    df_neg.columns = ['negative_words']
    df_neg['negative_words'] = df_neg['negative_words'].apply(lambda x: x.lower())
    return df_neg

def bag_of_words_prediction(filepath, n):
    pos_words_list = list(load_pos()['positive_words'])
    neg_words_list = list(load_neg()['negative_words']) 

    f = open(filepath)
    text_string = f.read()
    
    dataset = nltk.sent_tokenize(text_string)
    for i in range(len(dataset)):
        dataset[i] = dataset[i].lower()
        dataset[i] = re.sub(r'\W', ' ', dataset[i])
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])
        dataset[i] = dataset[i].split()
        dataset[i] = [word for word in dataset[i] if not word in set(stopwords.words('english'))]
        dataset[i] = ' '.join(dataset[i])
        dataset[i] = ''.join([j for j in dataset[i] if not j.isdigit()])
        
    word2count = {}
    for data in dataset:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1
                
    freq_words = heapq.nlargest(n, word2count, key=word2count.get)
    count_pos = 0
    for i in freq_words:
        if i in pos_words_list:
            count_pos += word2count[i]

    count_neg = 0
    for i in freq_words:
        if i in neg_words_list:
            count_neg += word2count[i]

    if count_pos > count_neg:
        return 1
    else:
        return 0

