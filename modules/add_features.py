# Add features to dataset from data.preprocessing module 
# based on list(names of features)

import pandas as pd
import modules.data_preprocessing as dp

def addFeatures(features, price, list_):
    df = features.copy()
    if 'daily_return' in list_:
        daily_return = dp.compute_daily_return(price)
        daily_return = daily_return.to_frame(name='Daily Return')
        features = features.join(daily_return)
    if 'net_price_3day' in list_:
        net_price_3day = dp.net_change(price, 3)
        net_price_3day = net_price_3day.to_frame(name='3 Day Net Change')
        features = features.join(net_price_3day)
    if '10_day_volatility' in list_:
        volatility_10 = dp.compute_volatility(price, 10)
        volatility_10 = volatility_10.to_frame(name='10 Day Volatility')
        features = features.join(volatility_10)
    if '20_day_moving_average' in list_:
        moving_average_20 = dp.compute_rolling_mean(price, 20)
        moving_average_20 = moving_average_20.to_frame(name='20 Day Moving Avg')
        features = features.join(moving_average_20)
    if '50_day_moving_average' in list_:
        moving_average_50 = dp.compute_rolling_mean(price, 50)
        moving_average_50 = moving_average_50.to_frame(name='50 Day Moving Avg')
        features = features.join(moving_average_50)
    if 'upper_bollinger_band' in list_:
        upper_bollinger_band = dp.compute_bollinger_bands(price, 20, flag='upper')
        upper_bollinger_band = upper_bollinger_band.to_frame(name='Upper Bollinger')
        features = features.join(upper_bollinger_band)
    if 'lower_bollinger_band' in list_:
        lower_bollinger_band = dp.compute_bollinger_bands(price, 20, flag='lower')
        lower_bollinger_band = lower_bollinger_band.to_frame(name='Lower Bollinger')
        features = features.join(lower_bollinger_band)
    if 'CCI' in list_:
        cci = dp.compute_cci(df, 10)
        cci = cci.to_frame(name='CCI')
        features = features.join(cci)
    if 'EOM' in list_:
        eom = dp.compute_eom(df)
        eom = eom.to_frame(name='EOM')
        features = features.join(eom)
    if 'EWM' in list_:
        ewm = dp.compute_ewma(df, span=20)
        ewm = ewm.to_frame(name='EWMA')
        features = features.join(ewm)
    return features