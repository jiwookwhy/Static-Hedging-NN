# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:40:47 2023

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
# given a tensor with three times to expiration (could be all the same)
# want to compute optimal strikes
# input layer (T_0, K_0 T_1,T_2,T_3) with output layer (K_1,K_2,K_3)

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
# start_year = 2016
# options = options.loc[(options.date.dt.year < start_year) | (options.date.dt.year >= start_year+8)]

df = options

if doing_CV:
    model_eval = load_model('./CV_models/' + ticker + '/' + ticker + '_evaluation_' + str(start_year) + '.keras')
else:
    model_eval = load_model(ticker + '_evaluation.keras')

non_training_cols = ['date', 'option_price_T_0', 'option_price_T_1', 'option_price_T_2', 'option_price_T_3', 'hedge_portfolio_T', 'w1', 'w2', 'w3', 'Y']
training_df = training_data_sampler(df, False, True)
weights = model_eval.predict(np.array(training_df.drop(columns = non_training_cols, axis = 1)))
training_df = append_weights(training_df, weights)

training_df = training_df[['date', 'moneyness_0', 'maturity_0', 'maturity_1', 'maturity_2', 'maturity_3', 'Y_hat', 'moneyness_1', 'moneyness_2', 'moneyness_3']]
training_df_grouped = training_df.loc[training_df.groupby(['date', 'moneyness_0', 'maturity_0'])['Y_hat'].idxmin()]

quick_RMSE(training_df.Y_hat)
quick_RMSE(training_df_grouped.Y_hat)

X = np.array(training_df_grouped[['moneyness_0', 'maturity_0', 'maturity_1', 'maturity_2', 'maturity_3']])
y = np.array(training_df_grouped[['moneyness_1', 'moneyness_2', 'moneyness_3']])

plt.hist(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(32, input_shape=(5,), activation='relu'))  # First hidden layer with 16 units and ReLU activation
model.add(Dense(8, activation='relu'))  # Second hidden layer with 8 units and ReLU activation
model.add(Dense(3))  # Output layer with 1 unit (length 1 tensor)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using mean squared error as the loss function

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)  # Adjust epochs and batch_size as needed

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print("Test loss:", loss)

if doing_CV:
    model.save('./CV_models/'  + ticker + '/' + ticker +  '_policy_' + str(start_year) + '.keras')
else:
    model.save(ticker + '_policy.keras')

