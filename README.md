# Model Analysis of Stock Price Trend Predictions based on Financial News

This is a data analysis project that aims to use different methods to predict the change of stock price from financial news and compare the performances of those methods.

We build following models:
- Baseline model: use Bag-of-Words to extract frequent words in news and determine their attitudes.
- AutoPhrase model: use AutoPhrase to extract high quality phrases and determine their attitudes.
- Doc2vec and LSTM model: use Doc2vec to create numberical representation of documents and use LSTM to predict the result.
- BERT model: use BERT to make prediction.

The code and results of different methods can be found in this repository.

## Local Run
```
$ python run.py [test] [eda] [etl] [doc2vec_lstm_model] [doc2vec_lstm_model2] [baseline_model] [autophrase_model] [bert_model]
```

### `test` target
This target runs a subset of pre-trained AutoPhrase dataset on the Word2vec model using the data in `test/testdata` and test the configurations in `config`.
### `eda` target
This target generate the EDA analysis and wordclouds of the positive and negative word bank and on the Apple stock price, and the results are saved to `data/eda_data`
### `etl` target
This target cleans and outputs the training dataset from the Apple stock price and corresponding financial news release dates, and the results are saved to `data/etl_data`
### `doc2vec_lstm_model` target
This target runs the doc2vec and lstm model to give prediction on Apple's stock price movement, and the model and results are saved to `data/d2v_lstm_model_data`
### `doc2vec_lstm_model2` target
This target runs the doc2vec and lstm model to give the sentiment analysis labels prediction based on the financial news title. The model and results are saved to `data/d2v_lstm_model_data2`
### `baseline_model` target
This target runs the baseline_model using the Bag-of-Words model.
### `autophrase_model` target
This target runs the Autophrase model to predict the stock price changes.
### `bert_model` target
This target runs the BERT model to predict the stock price changes.


## Docker
- The docker repository is `lizheng1226/dsc180_autofinews`.
- Please use the following command to run a DSMLP container using docker:
```
launch.sh -c 2 -m 4 -g 1 -i lizheng1226/dsc180_autofinews:latest
```

## Webpage
* https://aponyua991.github.io/AutoFiNews/


## Group Members
- Liuyang Zheng
- Mingjia Zhu
- Yunhan Zhang