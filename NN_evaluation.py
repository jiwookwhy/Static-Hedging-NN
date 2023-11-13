# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:48:47 2023

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

training_dfs = []

for T in [30,45,60,75,90]:
    # T = 30
    df = filter_by_dte(options, T, 400)
    df = filter_by_moneyness(df, 0.8, 1.2)
    df = add_time_T_price(df, T)
    
    df['option_price'] = 100*df['option_price']/df['settle']
    df['option_price_T'] = 100*df['option_price_T']/df['settle']
    
    # cross-validation block (comment out if not using CV)
    # start year refers to start year of the test set
    # test set runs from start_year to start_year + 5 
    # start_year is an element of [1996,2001,2006,2011,2016]
    # for VIX it is an element of [2006, 2012, 2018]
    # doing_CV = True
    # start_year = 2006
    # options = options.loc[(options.date.dt.year < start_year) | (options.date.dt.year >= start_year+5)]
    
    # normalize option price
    
    
    training_df = training_data_sampler(df, True, False)
    training_df = training_df.loc[training_df.Y < 0.05]
    training_df['T'] = T/365
    training_dfs.append(training_df)

training_df = pd.concat(training_dfs, ignore_index=True)
    
training_df.sample(n =100000).to_csv('test.csv', index = False)

# visualize data
plt.figure(figsize=(8, 6))
sns.histplot(data=training_df.sample(n =100000), x='moneyness_0', kde=True)
plt.title('Density Histogram of Moneyness')
plt.xlabel('Moneyness')
plt.ylabel('Density')

plt.figure(figsize=(8, 6))
sns.histplot(data=training_df.sample(n =100000), x='maturity_0', kde=True)
plt.title('Density Histogram of Maturity')
plt.xlabel('maturity')
plt.ylabel('Density')
plt.show()

# neural network training

X = np.array(training_df.drop(columns = ['w1', 'w2', 'w3', 'Y']))
y = np.array(training_df[['w1','w2','w3']])

plt.hist(training_df.maturity_0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model
model = Sequential()
model.add(Dense(32, input_shape=(12,), activation='relu'))  # First hidden layer with 16 units and ReLU activation
model.add(Dense(10, activation='relu'))  # Second hidden layer with 8 units and ReLU activation
model.add(Dense(3,activation='softmax'))  # Output layer with 1 unit (length 1 tensor)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using mean squared error as the loss function

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=128)  # Adjust epochs and batch_size as needed

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print("Test loss:", loss)

if doing_CV:
    model.save('./CV_models/' + ticker + '/' + ticker + '_evaluation_' + str(start_year) + '.keras')
else:
    model.save(ticker + '_evaluation_full.keras')

############################## TESTING BLOCK ##############################
from tensorflow.keras.models import load_model
# model = load_model(ticker + '_evaluation.keras')

non_training_cols = ['date', 'option_price_T_0', 'option_price_T_1', 'option_price_T_2', 'option_price_T_3', 'hedge_portfolio_T', 'w1', 'w2', 'w3', 'Y']
X_test = training_data_sampler(df, False, True).sample(1000000)
weights = model.predict(np.array(X_test.drop(columns = non_training_cols,axis = 1)))
test = append_weights(X_test, weights)
print(quick_RMSE(test.Y_hat), quick_RMSE(test.Y))

test2 = test.loc[(test.moneyness_0 >= 1.05) & (test.moneyness_0 <= 1.2)]
print(quick_RMSE(test2.Y_hat), quick_RMSE(test2.Y))


