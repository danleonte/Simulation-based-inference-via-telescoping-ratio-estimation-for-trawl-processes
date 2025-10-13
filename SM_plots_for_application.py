# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:28:29 2025

@author: leonted
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def corr_sup_ig_envelope(h, params):
    gamma, eta = params
    return np.exp(eta * (1-np.sqrt(2*h/gamma**2+1)))

# load data
folder_path = os.path.join('data')#,'temp_data_1999_01_01_2019_01_01')
data = pd.read_csv(os.path.join(folder_path,'d_h_temp_data_1999-01-01_2019-01-01_deseazonalized_both.csv'))
data.set_index('OBSERVATION_DATE', inplace=True)
# Ensure datetime index
data.index = pd.to_datetime(data.index)
ts = data["AIR_TEMPERATURE"]

# match the dates displayed in previous graphs with the indeces from samples_dict 
# this is needed because the saved samples and MAP do not have a time object
# adding one would probably simplify things
step_size = 168
first_end = 2000
ends = np.arange(first_end, len(data), step_size) #indeces at which we have samples
date_tickers_to_display = ['2014-01-01', '2015-03-26', '2016-07-07', '2017-09-27', '2019-01-01'] #same as in prevous graphs
end_indeces = [760,823,886,950,1013]


#load posterior samples and MAP
samples_dict = dict()
MAP_dict = dict()

for seq_len in (1000, 1500, 2000):   
    samples_dict[seq_len] = np.load(os.path.join(folder_path,f'results_{seq_len}.npy'))
    MAP_dict[seq_len] = np.load(os.path.join(folder_path,f'results_MAP_{seq_len}.npy'))

num_lags = 150

for seq_len in (1000, 1500, 2000):
    
    h = np.arange(num_lags+1)
    
    f, ax = plt.subplots(figsize=(12, 6))

        
    plt.title(f"ACFs with Sequence Length {seq_len}")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(True)
    plt.tight_layout()
    
    for end_index, date_ticker in zip(end_indeces, date_tickers_to_display):
        end_index = int(end_index)
        
        data_for_empirical_acf = ts.iloc[ends[end_index] - seq_len: ends[end_index]]
        acf_vals = acf(data_for_empirical_acf.values, nlags=num_lags)
        ax.plot(h, acf_vals, label=f"empirical: {date_ticker}")
        
        MAP_acf_params = MAP_dict[seq_len][end_index,:2]
        MAP_acf_vals   = corr_sup_ig_envelope(h, MAP_acf_params)
        #ax.plot(h, MAP_acf_vals, label=f"infered: {date_ticker}")

        
    plt.legend()
    plt.show()


