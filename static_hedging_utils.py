#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 00:29:17 2023

@author: jiwooky
"""

import numpy as np
import pandas as pd
from math import sqrt
import statsmodels.api as sm
import matplotlib.pyplot as plt

def add_time_T_price(options, T):
    # adds the time T price of an option as a column 'option_price_T'
    # T is the number of days to look forward
    # time 0 is the time period for each row
    # date_T is defined analogously
    df = options[['date', 'exdate', 'cp_flag', 'optionid', 'option_price', 'settle', 'dte', 'forward_price', 'strike_price']]
    # note we want the closest date to date_T_temp
    df['date_T_temp'] = df['date'] + pd.Timedelta(days=T)
    
    df = df.sort_values(['date_T_temp'])
    options_T = options[['date', 'optionid', 'option_price', 'settle']].sort_values('date')
    options_T.columns = ['date_T', 'optionid', 'option_price_T', 'settle_T']
    
    df = pd.merge_asof(df, options_T, by = 'optionid',
            left_on = 'date_T_temp', right_on = 'date_T',
            tolerance=pd.Timedelta(days = 4), direction='nearest')
    
    df.drop(columns = ['date_T_temp'], inplace = True)
    
    df = df.loc[df['date_T'] <= df['exdate']]
    return df.reset_index(drop = True)

def filter_by_dte(df, lower_bound_dte, upper_bound_dte):
    if 'maturity' in df.columns:
        df = df.loc[(df.maturity >= lower_bound_dte/365) & (df.maturity <= upper_bound_dte/365)]   
    else:
        df = df.loc[(df.dte >= lower_bound_dte) & (df.dte <= upper_bound_dte)]
    return df.reset_index(drop = True)

def filter_by_moneyness(df, lower_bound_moneyness, upper_bound_moneyness):
    if 'moneyness' in df.columns:
        df = df.loc[df.moneyness >= lower_bound_moneyness] 
        df = df.loc[df.moneyness <= upper_bound_moneyness]   
    else: 
        df = df.loc[df.strike_price/df.forward_price >= lower_bound_moneyness]
        df = df.loc[df.strike_price/df.forward_price <= upper_bound_moneyness]
    return df.reset_index(drop = True)

def add_risk_factors(df, option_risk_factors):
    risk_cols = ['date', 'exdate', 'SVIX', 'LT', 'RT']
    df = df.merge(option_risk_factors[risk_cols], on = ['date', 'exdate'], how = 'left')
    
    option_risk_factors_T = option_risk_factors[risk_cols]
    option_risk_factors_T.columns = ['date_T', 'exdate', 'SVIX_T', 'LT_T', 'RT_T']
    
    df = df.merge(option_risk_factors_T, on = ['date_T', 'exdate'],  how = 'left')
    
    return df.reset_index(drop = True)
