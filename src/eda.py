import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def pos_words():
    df_pos = pd.read_csv('LoughranMcDonald_Positive.csv', header=None)
    df_pos = df_pos.drop(1, axis=1)
    df_pos.columns = ['positive_words']
    df_pos['positive_words'] = df_pos['positive_words'].apply(lambda x: x.lower())
    return df_pos

def neg_words():
    df_neg = pd.read_csv('LoughranMcDonald_Negative.csv', header=None)
    df_neg = df_neg.drop(1, axis=1)
    df_neg.columns = ['negative_words']
    df_neg['negative_words'] = df_neg['negative_words'].apply(lambda x: x.lower())
    return df_neg

def word_cloud_pos():
    text = " ".join(cat for cat in pos_words().positive_words)
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def word_cloud_neg():
    text2 = " ".join(cat for cat in neg_words().negative_words)
    word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(text2)
    plt.imshow(word_cloud2, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def get_stock(fp):
    stock = pd.read_csv(fp)
    return stock

def plot_stock(fp):
    df = get_stock(fp)
    for i in [0, 1, 3, 5, 7]:
        plt.plot(df.columns[1:], [float(i) for i in df.iloc[i, 1:]])
        plt.show()
