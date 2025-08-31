# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:44:09 2025

@author: leonted
"""

import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt


#plt.plot(low[:,-1])
#plt.plot(median[:,-1])
#plt.plot(up[:,-1])

def plot_marginal_variable_with_confidence_regions(samples_dict, marginal_variable_to_use, step_size, q ):
    
    
    mapping = {
        'mu': 2,
        'sigma': 3,
        'beta': 4
    }
    
    marginal_column_to_use = mapping[marginal_variable_to_use]

    
    f, ax = plt.subplots()
    
    len_to_use = len(samples_dict[2000])
    
    for seq_len in (1000,1500,2000):
        
        
        samples_to_use = samples_dict[seq_len]
        assert len_to_use == len(samples_to_use)
        
        #ax.plot(MLE_dict[seq_len][marginal_column_to_use], label = '')
        low, up = np.quantile(samples_to_use[:,:,marginal_column_to_use][-num_entries_to_use:], q = q, axis=1)
        
        #ax.fill_between(, low, up, 
        #                label=f"95% CI {seq_len}", alpha=0.3)
        
        # plot CI bounds as lines
        ax.plot(dates, low, lw=1, ls="--", label=f"95% CI {seq_len}", color = colors[seq_len])
        ax.plot(dates, up,  lw=1, ls="--", color = colors[seq_len])
        
        print(f'For seq_len = {seq_len} {marginal_variable_to_use}, the sum of CI widths is: ', round(np.mean(up - low),4))
        
    plt.legend()
    plt.title(marginal_variable_to_use)



def plot_MLE(MAP_dict,marginal_variable_to_use, step_size ):
    
    mapping = {
        'mu': 2,
        'sigma': 3,
        'beta': 4
    }
    
    marginal_column_to_use = mapping[marginal_variable_to_use]
    len_to_use = len(samples_dict[2000])
    
    f, ax = plt.subplots()
    
    for seq_len in (1000,1500,2000):
        
        MAP_to_use = MAP_dict[seq_len]
        assert len_to_use == len(MAP_to_use)
        

            
        ax.plot(dates,MAP_to_use[:,marginal_column_to_use][-num_entries_to_use:]
, label = f'{seq_len}')
    
    plt.legend()
    plt.title(marginal_variable_to_use)

        
        
        
if __name__ == '__main__':
    
    q = [0.025,0.975]
    colors = dict(zip((1000,1500,2000),plt.cm.Set2.colors[:3]))

    step_size = 168
    
    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data') 
    data = pd.read_csv(os.path.join(dataset_path, 'd_h_temp_data_1999-01-01_2019-01-01_deseazonalized_both.csv'))
                                    #'VIX_25_y.csv'))['resid']#'normalized_temperature_data1.csv'))
    
    first_end = 2000
    ends = np.arange(first_end, len(data), step_size)
    num_entries_to_use = 259
    dates = data.iloc[ends].OBSERVATION_DATE[-num_entries_to_use:]

    samples_dict = dict()
    MAP_dict = dict()
    
    
    for seq_len in (1000, 1500, 2000):   
        samples_dict[seq_len] = np.load(os.path.join(dataset_path,f'results_{seq_len}.npy'))
        MAP_dict[seq_len] = np.load(os.path.join(dataset_path,f'results_MAP_{seq_len}.npy'))
        
        
    plot_marginal_variable_with_confidence_regions(samples_dict, 'mu',step_size,  q)
    plot_marginal_variable_with_confidence_regions(samples_dict, 'beta', step_size, q)
    plot_marginal_variable_with_confidence_regions(samples_dict, 'sigma', step_size, q)
    
    plot_MLE(MAP_dict,'mu', step_size )
    plot_MLE(MAP_dict,'beta', step_size )
    plot_MLE(MAP_dict,'sigma', step_size )


        