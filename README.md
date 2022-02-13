# DSC180B_Project_AutoFiNews

This is a data analysis project that aims to use different methods to predict the change of stock price from financial news and compare the performances of those methods.

We build following models:
- Baseline model: use Bag-of-Words to extract frequent words in news and determine their attitudes.
- AutoPhrase model: use AutoPhrase to extract high quality phrases and determine their attitudes.
- Doc2vec and LSTM model: use Doc2vec to create numberical representation of documents and use LSTM to predict the result.
- BERT model: use BERT to make prediction.

The code and results of different methods can be found in this repository.

## Run
```
$ python run.py [test]
```

### `test` target
Runs a subset of pre-trained AutoPhrase dataset on the Word2vec model using the data in `test/testdata` and test the configurations in `config`.



## Webpage
* https://aponyua991.github.io/AutoFiNews/
