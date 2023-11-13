# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:19:39 2023

@author: jy298
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from itertools import permutations
from NN_utils import *
from tensorflow.keras.models import load_model

doing_CV = False
ticker = 'SPX'

options = pd.read_csv(ticker + '_options_panel.csv')
options = options.loc[options.am_settlement == 1]

options['date'] = pd.to_datetime(options['date'], format = '%Y-%m-%d')
options['exdate'] = pd.to_datetime(options['exdate'], format = '%Y-%m-%d')
options['maturity'] = ((options['exdate'] - options['date']).dt.days)/365

options = options.loc[options.cp_flag == 'C'].reset_index(drop = True)
options['moneyness'] = options['strike_price']/options['forward_price']
options['option_price'] = (options['best_bid'] + options['best_offer'])/2

T = 30
options = filter_by_dte(options, T, 400)
options = filter_by_moneyness(options, 0.8, 1.2)
options = add_time_T_price(options, T)

options['option_price'] = 100*options['option_price']/options['settle']
options['option_price_T'] = 100*options['option_price_T']/options['settle']

# cross-validation block (comment out if not using CV)
# start year refers to start year of the test set
# test set runs from start_year to start_year + 5
# doing_CV = True
# start_year = 2001
# options = options.loc[(options.date.dt.year >= start_year) & (options.date.dt.year < start_year+5)]

df = options

# get closest maturities for each expiration-date pair

maturities = options[['date', 'maturity']].drop_duplicates(['date', 'maturity']).sort_values(['date', 'maturity']).reset_index(drop = True)
maturities_1 = pd.DataFrame({'date_1': maturities['date'].shift(1), 'maturity_1': maturities['maturity'].shift(1)})
maturities_2 = pd.DataFrame({'date_2': maturities['date'].shift(2), 'maturity_2': maturities['maturity'].shift(2)})
maturities = pd.concat([maturities, maturities_1, maturities_2], axis = 1)

maturities = maturities.loc[(maturities.date == maturities.date_1) & (maturities.date == maturities.date_2)]
maturities = maturities.dropna().drop(columns = ['date_1', 'date_2'], axis = 1).reset_index(drop = True)

# model evaluation (ATM options)
ATM_options = options.loc[np.abs(options.moneyness - 1) <= 0.25]
ATM_options = ATM_options.merge(maturities, on = ['date', 'maturity'])
ATM_options['maturity_3'] = ATM_options['maturity_1']

if doing_CV:
    model_policy = load_model('./CV_models/'  + ticker + '/' + ticker +  '_policy_' + str(start_year) + '.keras')
else:
    model_policy = load_model(ticker + '_policy.keras')

X_policy = np.array(ATM_options[['moneyness', 'maturity', 'maturity_1', 'maturity_2', 'maturity_3']])
policy_output = model_policy.predict(X_policy)

# build test dataframe
testing_df = ATM_options[['date', 'date_T', 'maturity', 'moneyness', 'option_price', 'maturity_1', 'maturity_2', 'maturity_3', 'option_price_T']]
testing_df.columns = ['date', 'date_T', 'maturity_0', 'moneyness_0', 'option_price_0', 'maturity_1', 'maturity_2', 'maturity_3', 'option_price_T_0']

testing_df['moneyness_1'] = np.array(policy_output[:, 0], dtype = 'float64')
testing_df['moneyness_2'] = np.array(policy_output[:, 1], dtype = 'float64')
testing_df['moneyness_3'] = np.array(policy_output[:, 2], dtype = 'float64')

# map to closest availible moneyness
for i in [1,2,3]:
    testing_df = testing_df.sort_values('moneyness_' + str(i))
    temp_df = options[['date', 'maturity','moneyness', 'optionid']]
    temp_df.columns = ['date', 'maturity_' + str(i),'moneyness_'  + str(i) + str('_available'), 'optionid_' + str(i)]
    temp_df = temp_df.sort_values('moneyness_'  + str(i) + str('_available'))
    testing_df = pd.merge_asof(testing_df, temp_df, by = ['date', 'maturity_' + str(i)],
       left_on = 'moneyness_' + str(i), right_on = 'moneyness_'  + str(i) + str('_available'), direction = 'nearest')
    testing_df['moneyness_' + str(i)] = testing_df['moneyness_'  + str(i) + str('_available')]
    testing_df.drop(columns = ['moneyness_'  + str(i) + str('_available')], inplace = True)

    temp_df = options[['date', 'optionid', 'option_price']]
    temp_df.columns = ['date', 'optionid_' + str(i), 'option_price_' + str(i)]
    testing_df = testing_df.merge(temp_df, on = ['date',  'optionid_' + str(i)])
    
    temp_df = options[['date_T', 'optionid', 'option_price_T']].drop_duplicates(['optionid','date_T'])
    temp_df.columns = ['date_T', 'optionid_' + str(i), 'option_price_T_' + str(i)]
    testing_df = testing_df.merge(temp_df, on = ['date_T',  'optionid_' + str(i)])
    
testing_df = testing_df.drop(columns=['optionid_1', 'optionid_2', 'optionid_3', 'date', 'date_T'])
time_T_options = testing_df[['option_price_T_0', 'option_price_T_1','option_price_T_2', 'option_price_T_3']]
testing_df = testing_df[['maturity_0', 'moneyness_0', 'option_price_0', 'maturity_1',
       'moneyness_1', 'option_price_1', 'maturity_2', 'moneyness_2',
       'option_price_2', 'maturity_3', 'moneyness_3', 'option_price_3']]

if doing_CV:
    model_eval = load_model('./CV_models/' + ticker + '/' + ticker + '_evaluation_' + str(start_year) + '.keras')
else:
    model_eval = load_model(ticker + '_evaluation.keras')
    
X_eval = np.array(testing_df)
eval_output = model_eval.predict(X_eval)

time_T_options['w1'] = eval_output[:, 0]
time_T_options['w2'] = eval_output[:, 1]
time_T_options['w3'] = eval_output[:, 2]

time_T_options['hedge_portfolio_T'] = time_T_options['option_price_T_1']*time_T_options['w1'] 
time_T_options['hedge_portfolio_T'] = time_T_options['hedge_portfolio_T'] + time_T_options['option_price_T_2']*time_T_options['w2'] 
time_T_options['hedge_portfolio_T'] = time_T_options['hedge_portfolio_T'] + time_T_options['option_price_T_3']*time_T_options['w3'] 
time_T_options['Y'] = np.abs(time_T_options['option_price_T_0'] - time_T_options['hedge_portfolio_T'])
time_T_options['MAPE'] = np.abs(time_T_options['option_price_T_0'] - time_T_options['hedge_portfolio_T'])/time_T_options['option_price_T_0'] 

quick_RMSE(time_T_options['Y'])

testing_df = pd.concat([testing_df, time_T_options], axis = 1)
testing_df['K'] = np.round(testing_df['moneyness_0'], 2)
testing_df['T-Th'] = np.round(testing_df['maturity_0'] - 2*testing_df['maturity_1']/3 - testing_df['maturity_3']/3, 1)
testing_df['Y2'] =  np.power(testing_df['Y'],2)
testing_df['months_to_expiration'] = np.floor(testing_df['maturity_0']*12)

testing_results = testing_df.groupby(['K', 'months_to_expiration']).mean()
testing_results['RMSE'] = np.sqrt(testing_results['Y2'])
testing_results = testing_results.drop(columns = ['Y2'], axis = 1)
