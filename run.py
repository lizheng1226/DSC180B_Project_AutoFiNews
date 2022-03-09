import sys
import json
import os
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
from datetime import datetime as ddt


sys.path.insert(0, 'src')
# from test import generate_right_text, percentage_high_quality_phrase, call_precision_curve, draw_curve, create_word2vec_model, similarity_search
from test import *
from etl import *
from doc2vec_lstm_model import *
from doc2vec_lstm_model2 import *

from baseline_model import *
# from autophrase_model import *
# from eda import *



def main(targets):

    test_config = json.load(open('config/test-params.json'))
    high_config = json.load(open('config/high_quality-params.json'))
    call_config = json.load(open('config/call_precision-params.json'))
    draw_config = json.load(open('config/draw_curve-params.json'))
    word2vec_config = json.load(open('config/word2vec-params.json'))
    similarity_config = json.load(open('config/similarity-params.json'))
    etl_config = json.load(open('config/etl-params.json'))
    doc2vec_lstm_config = json.load(open('config/lstm_model-params.json'))
    doc2vec_lstm_config2 = json.load(open('config/lstm_model2-params.json'))

    baseline_config = json.load(open('config/baseline-params.json'))
    # autophrase_config = json.load(open('config/autophrase-params.json'))
    # eda_config = json.load(open('config/eda-params.json'))
    
    
    

    if 'test' in targets:
        generate_right_text(**test_config)
        percentage_high_quality_phrase(**high_config)
        call_precision_curve(**call_config)
        draw_curve(**draw_config)
        create_word2vec_model(**word2vec_config)
        similarity_search(**similarity_config)


    if 'etl' in targets:
        set_data(**{k:etl_config[k] for k in ('apple_csv','output_apple') if k in etl_config})
        get_stock_price_df(**{k:etl_config[k] for k in ('data','output_AAPL') if k in etl_config})
        get_train_df(**{k:etl_config[k] for k in ('path','output_df', "dates") if k in etl_config})
        get_df_prices(**{k:etl_config[k] for k in ('output_AAPL','output_df', "intermediate_df", "real_train_df") if k in etl_config})


    if "doc2vec_lstm_model" in targets:
        # tokenize_text(**doc2vec_lstm_config)
        create_d2v_model(**{k:doc2vec_lstm_config[k] for k in ("DM", "DM_MEAN", "VECTOR_SIZE", "WINDOW", "MIN_COUNT", "WORKERS", "ALPHA", "MIN_ALPHA",
                    "real_train_df", "MAX_FEATURES", "MAX_SEQUENCE_LENGTH", "TEST_SIZE", "RANDOM_STATE",
                    "output_d2v_model", "df") if k in doc2vec_lstm_config})
        


        create_lstm_model(**{k:doc2vec_lstm_config[k] for k in ("output_d2v_model", "batch_size", "epochs", "verbose", "df", "X", "model_accuracy", "model_loss", "confusion_matrix_1") if k in doc2vec_lstm_config})


    if "doc2vec_lstm_model2" in targets:

        create_d2v_model2(**{k:doc2vec_lstm_config2[k] for k in ("DM", "DM_MEAN", "VECTOR_SIZE", "WINDOW", "MIN_COUNT", "WORKERS", "ALPHA", "MIN_ALPHA",
                    "real_train_df", "MAX_FEATURES", "MAX_SEQUENCE_LENGTH", "TEST_SIZE", "RANDOM_STATE",
                    "output_d2v_model", "df1") if k in doc2vec_lstm_config2})



        create_lstm_model2(**{k:doc2vec_lstm_config2[k] for k in ("output_d2v_model", "batch_size", "epochs", "verbose", "df1", "X", "model_accuracy", "model_loss", "confusion_matrix_1") if k in doc2vec_lstm_config2})



    if "baseline_model" in targets:
        # bag_of_words_prediction(**baseline_config)
        run_all()


    # if autophrase_model in targets:
    #     add_coefficient(...)
    #     autophrase_model_train(...)
    #     autophrase_model_test(...)


    # if eda in targets:
    #     ...
    #     ...
    #     ...







if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)

