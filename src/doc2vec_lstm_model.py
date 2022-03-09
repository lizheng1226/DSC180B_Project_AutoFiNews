import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import glob
import os
import csv
from datetime import datetime, timedelta, date
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import json
from sklearn.metrics import confusion_matrix





def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


def tokenize_text_output(real_train_df, MAX_FEATURES, MAX_SEQUENCE_LENGTH, TEST_SIZE, RANDOM_STATE, df):
    new_df = pd.read_csv(real_train_df)
    new_df['content'] = new_df['content'].apply(cleanText)
    train, test = train_test_split(new_df, test_size=TEST_SIZE , random_state=RANDOM_STATE)
    new_df.to_csv(df)

    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                #if len(word) < 0:
                if len(word) <= 0:
                    continue
                tokens.append(word.lower())
        return tokens


    train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=[r["5 days label"]]), axis=1)
    test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=[r["5 days label"]]), axis=1)
    tokenizer = Tokenizer(num_words=MAX_FEATURES, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(new_df['content'].values)
    X = tokenizer.texts_to_sequences(new_df['content'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(X))

    # new_df.to_csv(df)
    return (train_tagged, X)

def create_d2v_model(DM, DM_MEAN, VECTOR_SIZE, WINDOW, MIN_COUNT, WORKERS, ALPHA, MIN_ALPHA,
                    real_train_df, MAX_FEATURES, MAX_SEQUENCE_LENGTH, TEST_SIZE, RANDOM_STATE,
                    output_d2v_model, df):
    d2v_model = Doc2Vec(dm=DM, dm_mean=DM_MEAN, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, alpha=ALPHA, min_alpha=MIN_ALPHA)
    tokenize_text = tokenize_text_output(real_train_df, MAX_FEATURES, MAX_SEQUENCE_LENGTH, TEST_SIZE, RANDOM_STATE, df)
    train_tagged = tokenize_text[0]
    X = tokenize_text[1]
    print(X)
    cwd = os.getcwd() + "/config/lstm_model-params.json"
    with open(cwd, "r") as jsonfile:
        data = json.load(jsonfile)
        data["X"] = X.tolist()

    with open(cwd, "w") as jsonfile:
        json.dump(data, jsonfile)


    d2v_model.build_vocab([x for x in tqdm(train_tagged.values)])

    # %%time
    for epoch in range(30):
        d2v_model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        d2v_model.alpha -= 0.002
        d2v_model.min_alpha = d2v_model.alpha

    print(d2v_model)
    print(len(d2v_model.wv.index_to_key))

    # embedding_matrix = np.zeros((len(d2v_model.wv.key_to_index) + 1, 20))

    d2v_model.save(output_d2v_model)



    return d2v_model



def create_lstm_model(output_d2v_model, batch_size, epochs, verbose, df, X, model_accuracy, model_loss, confusion_matrix_1):
    df = pd.read_csv(df)
    d2v_model = Doc2Vec.load(output_d2v_model)
    X = np.array(X)
    embedding_matrix = np.zeros((len(d2v_model.wv.key_to_index) + 1, 20))
    model = Sequential()
    model.add(Embedding(len(d2v_model.wv.key_to_index)+1,20,input_length=X.shape[1], weights=[embedding_matrix], trainable=True))
    # def split_input(sequence):
    #     return sequence[:-1], tf.reshape(sequence[1:], (-1,1))
    model.add(LSTM(50,return_sequences=False))
    model.add(Dense(2,activation="softmax"))

    model.summary()
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])


    Y = pd.get_dummies(df['5 days label']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.95, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    history = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, verbose = verbose)

    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_accuracy)
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_loss)
    plt.show()

    _, train_acc = model.evaluate(X_train, Y_train, verbose=2)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))


    yhat_probs = model.predict(X_test, verbose=0)
    print(yhat_probs)
    yhat_classes = np.argmax(yhat_probs,axis=1)
    print(yhat_classes)

    rounded_labels=np.argmax(Y_test, axis=1)
    cm = confusion_matrix(rounded_labels, yhat_classes)


    lstm_val = confusion_matrix(rounded_labels, yhat_classes)
    f, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(lstm_val, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
    plt.title('LSTM Classification Confusion Matrix for stock price change after 5 days')
    plt.xlabel('Y predict')
    plt.ylabel('Y test')
    plt.savefig(confusion_matrix_1)
    plt.show()


    return



