#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 00:28:16 2023

@author: jiwooky
"""

import numpy as np
import pandas as pd
from math import sqrt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from static_hedging_utils import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

options = pd.read_csv('SPX_options_panel.csv')
options['date'] = pd.to_datetime(options['date'], format = '%Y-%m-%d')
options['exdate'] = pd.to_datetime(options['exdate'], format = '%Y-%m-%d')

option_risk_factors = pd.read_csv('option_risk_factors.csv')
option_risk_factors['date'] = pd.to_datetime(option_risk_factors['date'], format = '%Y-%m-%d')
option_risk_factors['exdate'] = pd.to_datetime(option_risk_factors['exdate'], format = '%Y-%m-%d')


T = 30
df = add_time_T_price(options, T)
df = filter_by_dte(df, 60, 400)
df = filter_by_moneyness(df, 0.8, 1.2)
df = add_risk_factors(df, option_risk_factors)
df = df.drop_duplicates(['date', 'optionid'])

df = df.dropna()

df['del_price'] = (df['settle_T'] - df['settle'])/df['settle']
df['del_SVIX'] = df['SVIX_T'] - df['SVIX']
df['del_LT'] = df['LT_T'] - df['LT']
df['del_RT'] = df['RT_T'] - df['RT']
df['maturity'] = df['dte']/365
df['moneyness'] = df['strike_price']/df['forward_price']

df = df.loc[(df['del_SVIX'] != np.inf) & (df['del_SVIX'] != -np.inf)]
df = df.loc[df.cp_flag == 'C']

############################ train neural network ############################
# Generate train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['del_price', 'maturity', 'moneyness', 'del_SVIX', 'SVIX', 'del_RT', 'RT']],
                    (df['option_price_T'] - df['option_price'])/df['settle'], test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(32, input_shape=(7,), activation='relu'))  # First hidden layer with 16 units and ReLU activation
model.add(Dense(8, activation='relu'))  # Second hidden layer with 8 units and ReLU activation
model.add(Dense(1))  # Output layer with 1 unit (length 1 tensor)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using mean squared error as the loss function

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)  # Adjust epochs and batch_size as needed

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print("Test loss:", loss)
model.save('SPX_NN')

####################### End training block #######################


# load model (if a pre-trained one is available)
model = load_model('SPX_NN')

# compute numerical derivative
df['del_option_price'] = (df['option_price_T'] - df['option_price'])/df['settle']
df_1mo = df[['del_price', 'maturity', 'moneyness', 'del_SVIX', 'SVIX', 'del_RT', 'RT', 'del_option_price']]
df_1mo = df_1mo.reset_index(drop = True)

def add_partial_derivatives(df, model):
    partial_price = []
    partial_SVIX = []
    partial_RT = []
    partial_arrays = [np.array([0.01, 0, 0, 0, 0, 0, 0]), 
                      np.array([0, 0, 0, 0, 0.01, 0, 0]),
                      np.array([0, 0, 0, 0, 0, 0, 0.025])]
    df = df.reset_index(drop = True)
    X = df[['del_price', 'maturity', 'moneyness', 'del_SVIX', 'SVIX', 'del_RT', 'RT']]
    Y = df['del_option_price']
    df['est_del_option_price'] = model.predict(np.array(X))
    df['error'] = np.abs(df['est_del_option_price'] - Y)
    
    X_price = np.array(X) + np.array(partial_arrays[0])
    X_price_output = model.predict(X_price).flatten()
    X_SVIX = np.array(X) + np.array(partial_arrays[1])
    X_SVIX_output = model.predict(X_SVIX).flatten()
    X_tail = np.array(X) + np.array(partial_arrays[2])
    X_tail_output = model.predict(X_tail).flatten()
    
    df['partial_price'] =  (X_price_output - df['est_del_option_price'])/0.01
    df['partial_SVIX'] = (X_SVIX_output - df['est_del_option_price'])/0.01
    df['partial_RT'] = (X_tail_output - df['est_del_option_price'])/0.0025 
    
    return df

# 1-month holding derivatives and error
df_1mo = add_partial_derivatives(df_1mo, model)

# compute linear approx error 
df_1mo['linear_approx_error'] = np.abs(df_1mo['del_price']*df_1mo['partial_price'] + 
    df_1mo['del_SVIX']*df_1mo['partial_SVIX'] + df_1mo['del_RT']*df_1mo['partial_RT'] - df['del_option_price'])

df_1mo.to_csv('temp.csv', index = False)

# error and partial derivatives plots
dte_bounds = [60, 90]
K_bounds = [0.95, 1.05]
df_temp = filter_by_dte(df_1mo, dte_bounds[0], dte_bounds[1])
df_temp = filter_by_moneyness(df_temp, K_bounds[0], K_bounds[1])

plt.plot(df_1mo['error'])
plt.plot(df_1mo['linear_approx_error'])







