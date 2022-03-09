import sys
import json
import os
import io
import re
import string
import tqdm
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as ddt


def set_data(apple_csv, output_apple):
    data = pd.read_csv(apple_csv)
    data = data[['Date','Adjusted Close']]
    convert = lambda x : datetime.datetime.strptime(x, '%d-%m-%Y').strftime('%Y-%m-%d')
    data['Date'] = data['Date'].apply(convert)
    data.to_csv(output_apple)
    return data
    

def fetch_means(date, data, t_range_list):
    def find_ind(data, date):
        return data.index[data['Date'] == date][0]
    idx_s = find_ind(data, date)
    p = data['Adjusted Close'][idx_s]
    d = dict()
    d['Initial Price'] = p
    for i in t_range_list[1:]:
        t = i
        idx_e = idx_s + t
        data_ = data[idx_s:idx_e]
        mean_ = np.mean(data_['Adjusted Close'])
        d[str(t)+' days'] = mean_
        if mean_ > p :
            d[str(t)+' days label'] = True
        else:
            d[str(t)+' days label'] = False
    return d


def get_stock_price_df(data, output_AAPL):
    data = pd.read_csv(data)
    # t_range_list = [1,5,10,20,60]
    time_list = ['2021-11-23', '2021-04-20', '2021-08-26', '2021-12-21', '2021-05-17', '2021-10-18',
                 '2021-09-01', '2021-10-18', '2021-09-28', '2021-09-01', '2022-02-08', '2021-09-14', '2021-10-18', '2021-06-07', '2021-11-09',
                 '2021-11-30', '2021-10-27', '2021-11-10', '2021-10-18', '2021-10-04', '2021-10-05', '2021-09-14', '2021-06-07', '2022-02-08',
                 '2021-09-19', '2022-01-27', '2021-06-22', '2021-06-07', '2021-09-14']
    
    aapl = dict()
    for i in time_list:
        try:
            aapl[i] = fetch_means(i,data,[1,5,10,20,60])
        except IndexError:
            aapl[i] = None
    AAPL_df = pd.DataFrame(aapl)
    # AAPL_df.T
    AAPL_df.to_csv(output_AAPL)
    return AAPL_df
            
            
def get_filename(path):
    filenames = []
    files = [i.path for i in os.scandir(path) if i.is_file()]
    for filename in files:
        filename = os.path.basename(filename)
        filenames.append(filename)
    return filenames                      
            


def get_train_df(path, output_df, dates):
    files = get_filename(path)
    lst_documents = []
    for _file in files:
        #lst_documents = []
        file_name = _file
        lst = []
        with open(path + _file,'r') as f:
            for lines in f:
                lst.append(lines.replace("\n", "").strip().replace("•",""))
                # text = f.read()
                # writer = csv.writer(csv_file)
                # writer.writerow([file_name, lst])
            lst = " ".join(lst)
        lst_documents.append(lst)

    df = pd.DataFrame(data = {"Report": files, "content": lst_documents})
    df_date = pd.read_csv(dates, index_col=[0]).rename(columns = {"name": "Report"})
    df_date["Report"] = df_date["Report"].apply(lambda x: x + ".txt")
    df = pd.merge(df, df_date, on = "Report")
    df.to_csv(output_df)

    return df



def get_df_prices(output_AAPL, output_df, intermediate_df, real_train_df):
    df_apple = pd.read_csv(output_AAPL).T
    df_apple.columns = df_apple.iloc[0]
    df_apple = df_apple[1:]
    df_apple.reset_index().to_csv(intermediate_df)
    df_apple = pd.read_csv(intermediate_df)
    df_apple = df_apple.rename(columns = {"index": "date"})
    df_apple["date"] = df_apple["date"].apply(lambda x: ddt.strptime(str(x), "%Y-%m-%d"))
    df = pd.read_csv(output_df, index_col = 0)
    df["date"] = df["date"].apply(lambda x: ddt.strptime(str(x), "%Y-%m-%d"))
    new_df = pd.merge(df, df_apple, how='outer', on='date').dropna()
    new_df["5 days label"] = new_df["5 days label"].astype(int)
    new_df["10 days label"] = new_df["10 days label"].astype(int)
    new_df["20 days label"] = new_df["20 days label"].astype(int)
    new_df["60 days label"] = new_df["60 days label"].astype(int)
    new_df.to_csv(real_train_df)
    print(new_df.head())


    return new_df



    