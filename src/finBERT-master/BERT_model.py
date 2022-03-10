import pandas as pd
import os
import numpy as np
from sklearn import svm
from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import os
import torch

# iterate over files in
# that directory
def run_pred(model):
    directory = 'apple_news_text_1'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if (os.path.isfile(f) and f.endswith('.txt')):
            text_path = f
            output_dir = "output/"
            output_path = filename
            with open(text_path,'r') as f:
                text = f.read()
            predict(text,model,write_to_csv=True,path=os.path.join(output_dir,output))

def get_label(day, date,label):
    x = label[str(day)+' days label'][label.date == date].values[0]
    if x == 'True':return 1
    else: return 0

def load_data():
    df1 = pd.read_csv('apple_news_text_1/dates.csv')
    label1 = pd.read_csv('AAPL_labels_1.csv',header = None).T
    #df2 = pd.read_csv('apple_news_text_2/dates.csv')
    #label2 = pd.read_csv('AAPL_labels_2.csv',header = None).T

    col = ['date']+list(label1.iloc[0][1:])
    label1.columns = col
    label1 = label1.iloc[1:]
    label1.index = label1['date']

    df1['label_5'] = [get_label(5,x,label1) for x in df1['date']]
    return df1

def process_data(df1):
    list_score = []
    #list_at = []
    directory = 'output'
    for i in range(len(df1)):
        path = df1['name'][i]
        f_path = os.path.join(directory,path)
        d = pd.read_csv(f_path)
        score = np.mean(d['sentiment_score'])
        #at = int(sum(d['prediction'].apply(to_num))>8)
        list_score.append(score)
        #list_at.append(at)

    df1['score'] = list_score
    #df1['sentiment_label'] = list_at
    df1= df1[['name','date','label_5','score']]
    return df1

def fit(df1):
    X = df1[['score']]
    y = df1.label_5
    clf = svm.SVC()
    clf.fit(X, y)
    return clf

def run_all():
    model_path = "models/classifier_model/finbert-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)
    run_pred(model)
    df1 = load_data()
    df1 = process_data(df1)
    clf = fit(df1)
    return clf.predict(df1[['score']])
