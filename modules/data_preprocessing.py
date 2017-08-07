# Data preprocessing functions for feature engineering
import pandas as pd
# Calculate daily returns (price[t]/price[t-1] - 1)
def compute_daily_return(df):
    daily_return = df.copy()
    daily_return[1:] = (daily_return[1:]/df[:-1].values) - 1
    daily_return.ix[0,0] = 0
    return daily_return

# Calculate Net Price Change
def net_change(series, days):
    net_change = series.pct_change(days)
    net_change = net_change.fillna(value=0)
    return net_change

# Calculate rolling mean/moving average
def compute_rolling_mean(df,window):
    rolling_mean = df.rolling(window,center=False).mean()
    rolling_mean = rolling_mean.fillna(method='bfill')
    return rolling_mean

# Calculate rolling standard deviation
def compute_rolling_std(df, window):
    rolling_std = df.rolling(window, center=False).std()
    rolling_std = rolling_std.fillna(method='bfill')
    return rolling_std

# Calculate 10 day volatility using daily returns
def compute_volatility(df, window):
    daily_return = compute_daily_return(df)
    volatility = compute_rolling_std(daily_return, window)
    volatility = volatility.fillna(method='bfill')
    return volatility

# Calculate Bollinger bands(upper and lower)
def compute_bollinger_bands(df, window, flag):
    roll_mean = compute_rolling_mean(df, window)
    roll_std = compute_rolling_std(df, window)
    if flag == 'upper':
        band = roll_mean + 2*roll_std
        band = band.fillna(method='bfill')
    elif flag == 'lower':
        band = roll_mean - 2*roll_std
        band = band.fillna(method='bfill')
    else:
        print('Error: Value for flag is either "upper" or "lower"')
    return band

# Calculate commodity channel oscillator
def compute_cci(df, window):
    typical_price = (df['Low'] + df['Close'] + df['High'])/3
    moving_average = compute_rolling_mean(typical_price, window)
    standard_average = compute_rolling_std(typical_price, window)
    cci = (typical_price - moving_average)/(0.015*standard_average)
    cci = cci.fillna(method='bfill')
    return cci

# Calculate ease of movement
def compute_eom(df):
    distance_moved = ((df['High'] + df['Low'])/2) - ((df['High'].shift(1) + df['Low'].shift(1))/2)
    box_ratio = (df['Volume']/100000000)/(df['High'] - df['Low'])
    eom = distance_moved/box_ratio
    eom = eom.fillna(value=0)
    return eom

# Calculate exponential weighted moving average
def compute_ewma(df, span):
    average_price = (df['Open'] + df['Low'] + df['High'])/3
    ema_span = average_price.ewm(span=span).mean()
    return ema_span
