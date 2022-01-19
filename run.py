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


sys.path.insert(0, 'src')
from test import generate_right_text, percentage_high_quality_phrase, call_precision_curve, draw_curve, create_word2vec_model, similarity_search



def main(targets):

    test_config = json.load(open('config/test-params.json'))
    high_config = json.load(open('config/high_quality-params.json'))
    call_config = json.load(open('config/call_precision-params.json'))
    draw_config = json.load(open('config/draw_curve-params.json'))
    word2vec_config = json.load(open('config/word2vec-params.json'))
    similarity_config = json.load(open('config/similarity-params.json'))

    if 'test' in targets:
        generate_right_text(**test_config)
        percentage_high_quality_phrase(**high_config)
        call_precision_curve(**call_config)
        draw_curve(**draw_config)
        create_word2vec_model(**word2vec_config)
        similarity_search(**similarity_config)




if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)

