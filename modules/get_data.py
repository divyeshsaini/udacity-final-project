import os
import pandas as pd

# Get data from csv file to pandas Dataframe
def data(symbol, dates, base_dir='data'):
    df = pd.DataFrame(index=dates)
    path_csv = os.path.join(base_dir, '{}.csv'.format(str(symbol)))
    df_temp = pd.read_csv(path_csv, index_col='Date', parse_dates=True)
    df = df.join(df_temp, how='inner')

    return df

def convert_label(df):
    df_temp = df.copy()
    df_temp[1:] = (df_temp[1:]-df[:-1].values)
    df_temp.ix[0,0] = 1
    for i, row in df_temp.iteritems():
        if df_temp[i] > 0:
            df_temp[i] = 1
        elif df_temp[i] < 0:
            df_temp[i] = -1
        else:
            df_temp[i] = 0
    return df_temp

def get_feature_label(start_date, end_date):
    #start_date = '2010-01-01'
    #end_date = '2017-06-30'
    dates = pd.date_range(start_date, end_date)
    df_amzn = data('AMZN', dates)
    df_spy = data('SPY', dates).rename(columns={'Adj Close':'SPY'}) 
    df_amzn = df_amzn.join(df_spy['SPY'])
    features = df_amzn.drop('Close', axis=1)
    label = df_amzn['Close']
    return features, label

