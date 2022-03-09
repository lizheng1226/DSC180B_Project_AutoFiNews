import pandas as pd
import nltk
import re
import numpy as np
import heapq
from nltk.corpus import stopwords
import PyPDF2
import pdfplumber
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import os
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm



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

def add_coefficient(word):
    pos_words_list = list(load_pos()['positive_words'])
    neg_words_list = list(load_neg()['negative_words']) 

    word_list = word.split(" ")
    if any(i in pos_words_list for i in word_list):
        return 1
    elif any(i in neg_words_list for i in word_list):
        return -1
    else:
        return 0

def autophrase_model_train(fps, stock):
    big_df = pd.DataFrame()
    for i in range(len(fps)):
        df = pd.read_fwf(fps[i], header=None)
        df.columns = ['score', 'phrase']
        df['coefficient'] = df['phrase'].apply(add_coefficient)
        df['weighted_score'] = df['score'] * df['coefficient']
        big_df[i] = df['weighted_score']
    
    clf = svm.SVC()
    clf.fit(big_df, stock)
    return clf

def autophrase_model_test(fps, clf):
    big_df = pd.DataFrame()
    for i in range(len(fps)):
        df = pd.read_fwf(fps[i], header=None)
        df.columns = ['score', 'phrase']
        df['coefficient'] = df['phrase'].apply(add_coefficient)
        df['weighted_score'] = df['score'] * df['coefficient']
        big_df[i] = df['weighted_score']
    
    return clf.predict(big_df)




    