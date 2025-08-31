# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:44:09 2025
@author: leonted
"""
import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])  

def plot_marginal_variable_with_confidence_regions(samples_dict, marginal_variable_to_use, step_size, q, ax):
    
    # TeX formatting for Greek letters
    tex_mapping = {
        'mu': r'$\mu$',
        'sigma': r'$\sigma$',
        'beta': r'$\beta$'
    }
    
    mapping = {
        'mu': 2,
        'sigma': 3,
        'beta': 4
    }
    
    marginal_column_to_use = mapping[marginal_variable_to_use]
    
    len_to_use = len(samples_dict[2000])
    
    # Use indices instead of dates for plotting
    x_indices = np.arange(num_entries_to_use)
    
    for seq_len in (1000,1500,2000):
        
        samples_to_use = samples_dict[seq_len]
        assert len_to_use == len(samples_to_use)
        
        low, up = np.quantile(samples_to_use[:,:,marginal_column_to_use][-num_entries_to_use:], q = q, axis=1)
        
        # Calculate mean CI width for label
        mean_ci_width = round(np.mean(up - low), 2)
        
        # plot CI bounds as lines with mean width in label
        ax.plot(x_indices, low, lw=1, #ls="--", 
                label=f"{seq_len}, width={mean_ci_width}", 
                color = colors[seq_len])
        ax.plot(x_indices, up,  lw=1, #ls="--", 
                color = colors[seq_len])
        
        print(f'For seq_len = {seq_len} {marginal_variable_to_use}, the mean CI width is: ', mean_ci_width)
    
    # Set x-ticks to show only every nth date
    n_ticks = 5  # Show fewer date labels
    tick_indices = np.linspace(0, num_entries_to_use-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    date_labels = [pd.to_datetime(dates.iloc[i]).strftime('%Y-%m-%d') for i in tick_indices]
    # Subtract one day from the first date label
    first_date = pd.to_datetime(dates.iloc[tick_indices[0]]) - pd.Timedelta(days=1)
    date_labels[0] = first_date.strftime('%Y-%m-%d')
    
    ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=10.5)
    
    # Adjust legend position based on variable for CI plots
    if marginal_variable_to_use == 'mu':
        ax.legend(loc='lower center', bbox_to_anchor=(0.65, 0.01))
    elif marginal_variable_to_use == 'beta':
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))
    else:  # sigma
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))
    ax.set_title(tex_mapping[marginal_variable_to_use])
    
def plot_MLE(MAP_dict, marginal_variable_to_use, step_size, ax):
    
    # TeX formatting for Greek letters
    tex_mapping = {
        'mu': r'$\mu$',
        'sigma': r'$\sigma$',
        'beta': r'$\beta$'
    }
    
    mapping = {
        'mu': 2,
        'sigma': 3,
        'beta': 4
    }
    
    marginal_column_to_use = mapping[marginal_variable_to_use]
    len_to_use = len(samples_dict[2000])
    
    # Use indices instead of dates for plotting
    x_indices = np.arange(num_entries_to_use)
    
    for seq_len in (1000,1500,2000):
        
        MAP_to_use = MAP_dict[seq_len]
        assert len_to_use == len(MAP_to_use)
        
        ax.plot(x_indices, MAP_to_use[:,marginal_column_to_use][-num_entries_to_use:], 
                label = f'{seq_len}', color = colors[seq_len])
    
    # Set x-ticks to show only every nth date
    n_ticks = 5  # Show fewer date labels
    tick_indices = np.linspace(0, num_entries_to_use-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    # Format dates to show only the date part
    date_labels = [pd.to_datetime(dates.iloc[i]).strftime('%Y-%m-%d') for i in tick_indices]
    # Subtract one day from the first date label
    first_date = pd.to_datetime(dates.iloc[tick_indices[0]]) - pd.Timedelta(days=1)
    date_labels[0] = first_date.strftime('%Y-%m-%d')    
    
    
    
    ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=10.5)
    
    # Adjust legend position based on variable for MAP plots
    if marginal_variable_to_use == 'mu':
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))
    elif marginal_variable_to_use == 'beta':
        ax.legend(loc='lower right')
    else:  # sigma
        ax.legend(loc='upper right')
    ax.set_title(tex_mapping[marginal_variable_to_use])
    
        
        
        
if __name__ == '__main__':
    
    q = [0.025,0.975]
    colors = {1000: '#0073C2', 1500: '#EFC000', 2000: '#CD534C'}

    step_size = 168
    
    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data') 
    data = pd.read_csv(os.path.join(dataset_path, 'd_h_temp_data_1999-01-01_2019-01-01_deseazonalized_both.csv'))
                                    #'VIX_25_y.csv'))['resid']#'normalized_temperature_data1.csv'))
    
    first_end = 2000
    ends = np.arange(first_end, len(data), step_size)
    num_entries_to_use = 254
    dates = data.iloc[ends].OBSERVATION_DATE[-num_entries_to_use:]
    samples_dict = dict()
    MAP_dict = dict()
    
    
    for seq_len in (1000, 1500, 2000):   
        samples_dict[seq_len] = np.load(os.path.join(dataset_path,f'results_{seq_len}.npy'))
        MAP_dict[seq_len] = np.load(os.path.join(dataset_path,f'results_MAP_{seq_len}.npy'))
    
    # Create combined figure for confidence intervals
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('95% Confidence Intervals', y=0.98, fontsize=14)
    
    plot_marginal_variable_with_confidence_regions(samples_dict, 'mu', step_size, q, axes[0])
    plot_marginal_variable_with_confidence_regions(samples_dict, 'beta', step_size, q, axes[1])
    plot_marginal_variable_with_confidence_regions(samples_dict, 'sigma', step_size, q, axes[2])
    
    plt.tight_layout()
    plt.savefig('CI_for_marginal_params.pdf', dpi=900, bbox_inches='tight')
    plt.show()
    
    
    # Create combined figure for MLE/MAP
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('MAP Estimates', y=0.98, fontsize=14)
    
    plot_MLE(MAP_dict, 'mu', step_size, axes[0])
    plot_MLE(MAP_dict, 'beta', step_size, axes[1])
    plot_MLE(MAP_dict, 'sigma', step_size, axes[2])
    
    plt.tight_layout()
    plt.savefig('MAP_for_marginal_params.pdf', dpi=900, bbox_inches='tight')
    plt.show()