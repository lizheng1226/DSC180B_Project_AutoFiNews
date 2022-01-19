import io
import re
import string
import tqdm
import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers
from sklearn.metrics import precision_score, recall_score, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import time




def generate_right_text(original_text, right_text):
    with open(original_text,encoding='utf-8') as f:
        line = f.read()
    def func(x):
        start, end = x.span()
        s = line[start:end]
        s = s.replace('<phrase>', '').replace('</phrase>', '').replace(' ', '_')
        return s
    s = re.sub("<phrase>.*?</phrase>", func, line)
    # print(s)
    with open(right_text,'w',encoding='utf-8') as r:
        r.write(s)

    return r


def percentage_high_quality_phrase(text_path, output_100):
    autophrase = pd.read_csv(filepath_or_buffer = text_path, header = None, sep = "\t")
    autophrase_100 = autophrase[autophrase[0] > 0.5].sample(100)
    autophrase_100.to_csv(output_100)
    high_quality = len(autophrase_100[autophrase_100[0] > 0.9]) / len(autophrase_100)
    print("percentage of high-quality phrases is " , high_quality)
    return autophrase_100


def call_precision_curve(df, output_new_100):
    df_output = pd.read_csv(filepath_or_buffer = df).copy()
    df_output = df_output.rename(columns={'0': "score", '1': "multi_phrase"})
    # print(df_output.columns)
    df_output['high_quality'] = df_output['score'].apply(lambda x: 1 if x > 0.9 else 0)
    # autophrase_100.head(20)
    df_output["cum_sum"] = df_output["high_quality"].cumsum()
    df_output["cum_index"] = list(range(1, len(df_output)+1))
    df_output["recall"] = df_output["cum_sum"] / df_output["high_quality"].sum()
    df_output["precision"] = df_output["cum_sum"] / df_output["cum_index"]
    print("autophrase_100 now contains call and precision columns: \n\n", df_output.head())
    df_output.to_csv(output_new_100)
    return df_output



def draw_curve(df_curve, output_fig):
    df_draw = pd.read_csv(df_curve).copy()
    fig = plt.figure()
    plt.plot(df_draw["recall"], df_draw["precision"])
    fig.suptitle('Precision-Recall Curve', fontsize=20)
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=16)
    fig.savefig(output_fig)
    return fig




def create_word2vec_model(textfile, output_model):
    text_ds = tf.data.TextLineDataset(textfile).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


    # Define the vocabulary size and number of words in a sequence.
    vocab_size = 10000
    sequence_length = 5

    # Use the TextVectorization layer to normalize, split, and map strings to
    # integers. Set output_sequence_length length to pad all samples to same length.
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)
    vectorize_layer.adapt(text_ds.batch(1024))
    inverse_vocab = vectorize_layer.get_vocabulary()
    print(inverse_vocab[:20])
    len(inverse_vocab)
    text_vector_ds = text_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())
    for seq in sequences[:5]:
        print(f"{seq} => {[inverse_vocab[i] for i in seq]}")
    list_text = []
    for seq in sequences:
        list_text.append([inverse_vocab[i] for i in seq])
        # print(f"{seq} => {[inverse_vocab[i] for i in seq]}")
    model = gensim.models.Word2Vec(sentences=list_text, vector_size=100, window=10, min_count=1, workers=5)
    print(len(model.wv.index_to_key))
    model.save(output_model)
    return model






def similarity_search(output_model, df_multiwords, compare_autophrase):
    # time.sleep(10)
    # dff = pd.read_csv(compare_autophrase, header = None, sep = "\t").copy()
    # dff = dff.rename(columns={0: "score", 1: "multi_phrase"})
    model = Word2Vec.load(output_model)
    df_100 = pd.read_csv(filepath_or_buffer = df_multiwords).copy()
    autophrase_compare = pd.read_csv(filepath_or_buffer = compare_autophrase, header = None, sep = "\t").copy()
    autophrase_compare = autophrase_compare.rename(columns={0: "score", 1: "multi_phrase"})
    # high_quality3 = df_100[df_100["score"] > 0.9].sample(3)
    def remove(string):
        return string.replace(" ", "").replace("'","")
    autophrase_compare["multi_phrase"] = autophrase_compare["multi_phrase"].apply(remove)
    lst_words = []
    for index, word in enumerate(model.wv.index_to_key):
        lst_words.append(word)
    # dff[1] = dff[1].apply(remove)
    dfff = pd.DataFrame(lst_words)
    # print(dfff)
    autophrase_compare = autophrase_compare.merge(dfff, left_on="multi_phrase", right_on=0)
    # print(autophrase_compare)
    
    
    
    
    
    def insert_value_multidatabase(x):
        try:
            output = model.wv.similarity(x, 'multidatabase')
        except KeyError:
            return 0
        else:
            return output


    def insert_value_hypertext(x):
        try:
            output = model.wv.similarity(x, 'hypertext')
        except KeyError:
            return 0
        else:
            return output

    def insert_value_parallelcomputing(x):
        try:
            output = model.wv.similarity(x, 'parallelcomputing')
        except KeyError:
            return 0
        else:
            return output
        
        
        

    autophrase_compare["sim_multidatabase"] = autophrase_compare["multi_phrase"].apply(insert_value_multidatabase)
    autophrase_compare["sim_hypertext"] = autophrase_compare["multi_phrase"].apply(insert_value_hypertext)
    autophrase_compare["sim_parallelcomputing"] = autophrase_compare["multi_phrase"].apply(insert_value_parallelcomputing)
    autophrase_compare = autophrase_compare[["score", "multi_phrase", "sim_multidatabase", "sim_hypertext", "sim_parallelcomputing"]]
    
    print(autophrase_compare)
    
    
    
    
    
    multidatabase = autophrase_compare.sort_values(by=['sim_multidatabase'], ascending = False)[:5]
    result_multidatabase = dict(zip(multidatabase.multi_phrase, multidatabase.sim_multidatabase))
    print("top-5 results for multidatabase is ", result_multidatabase,  "\n")
    
    hypertext = autophrase_compare.sort_values(by=['sim_hypertext'], ascending = False)[:5]
    result_hypertext = dict(zip(hypertext.multi_phrase, hypertext.sim_hypertext))
    print("top-5 results for hypertext is ", result_hypertext, "\n")
    
    parallelcomputing = autophrase_compare.sort_values(by=['sim_parallelcomputing'], ascending = False)[:5]
    result_parallelcomputing = dict(zip(parallelcomputing.multi_phrase, parallelcomputing.sim_parallelcomputing))
    print("top-5 results for parallelcomputing is ", result_parallelcomputing, "\n")
    
    return autophrase_compare




