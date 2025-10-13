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
from statsmodels.tsa.stattools import acf
plt.style.use(['science', 'no-latex'])  

def plot_marginal_variable_with_confidence_regions(samples_dict, marginal_variable_to_use, step_size, q, ax):
    
    # TeX formatting for Greek letters
    tex_mapping = {
        'acf0' : 'acf0',
        'acf1' : 'acf1',
        'mu': r'$\mu$',
        'sigma': r'$\sigma$',
        'beta': r'$\beta$'
    }
    
    mapping = {
        'acf0': 0,
        'acf1': 1,
        'mu': 2,
        'sigma': 3,
        'beta': 4
    }
    
    marginal_column_to_use = mapping[marginal_variable_to_use]
    
    len_to_use = len(samples_dict[2000])
    
    # Use indices instead of dates for plotting
    
    for seq_len in (1000,1500,2000):
        
        samples_to_use = samples_dict[seq_len]
        assert len_to_use == len(samples_to_use)
        
        low, up = np.quantile(samples_to_use[:,:,marginal_column_to_use], q = q, axis=1)
        
        # Calculate mean CI width for label
        mean_ci_width = round(np.mean(up - low), 2)
        
        # plot CI bounds as lines with mean width in label
        ax.plot(low, lw=1, #ls="--", 
                label=f"{seq_len}, width={mean_ci_width}", 
                color = colors[seq_len])
        ax.plot(up,  lw=1, #ls="--", 
                color = colors[seq_len])
        
        print(f'For seq_len = {seq_len} {marginal_variable_to_use}, the mean CI width is: ', mean_ci_width)
    
    # Set x-ticks to show only every nth date
    n_ticks = 5  # Show fewer date labels
    #tick_indices = np.linspace(0, num_entries_to_use-1, n_ticks, dtype=int)
    #ax.set_xticks(tick_indices)
    ###date_labels = [pd.to_datetime(dates.iloc[i]).strftime('%Y-%m-%d') for i in tick_indices]
    # Subtract one day from the first date label
    ###first_date = pd.to_datetime(dates.iloc[tick_indices[0]]) - pd.Timedelta(days=1)
    ###date_labels[0] = first_date.strftime('%Y-%m-%d')
    
    ###ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=11.5)
    
    # Adjust legend position based on variable for CI plots
    # Adjust legend position based on variable for CI plots
    if marginal_variable_to_use == 'acf0':
        ax.legend(loc='lower center', bbox_to_anchor=(0.65, 0.01))
    elif marginal_variable_to_use == 'acf1':
        ax.legend(loc='lower center', bbox_to_anchor=(0.65, 0.01))
    elif marginal_variable_to_use == 'mu':
        ax.legend(loc='lower center', bbox_to_anchor=(0.65, 0.01))
    elif marginal_variable_to_use == 'beta':
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))
    else:  # sigma
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))
    ax.set_title(tex_mapping[marginal_variable_to_use], fontsize=14)
    
def plot_MLE(MAP_dict, marginal_variable_to_use, step_size, ax):
    
    # TeX formatting for Greek letters
    tex_mapping = {
        'acf0' : 'acf0',
        'acf1' : 'acf1',
        'mu': r'$\mu$',
        'sigma': r'$\sigma$',
        'beta': r'$\beta$'
    }
    
    mapping = {
        'acf0': 0,
        'acf1': 1,
        'mu': 2,
        'sigma': 3,
        'beta': 4
    }
    
    marginal_column_to_use = mapping[marginal_variable_to_use]
    len_to_use = len(samples_dict[2000])
    
    # Use indices instead of dates for plotting
    #x_indices = np.arange(num_entries_to_use)
    
    for seq_len in (1000,1500,2000):
        
        MAP_to_use = MAP_dict[seq_len]
        assert len_to_use == len(MAP_to_use)
        
        ax.plot( MAP_to_use[:,marginal_column_to_use], 
                label = f'{seq_len}', color = colors[seq_len])
    
    # Set x-ticks to show only every nth date
    #n_ticks = 5  # Show fewer date labels
    #tick_indices = np.linspace(0, num_entries_to_use-1, n_ticks, dtype=int)
    #ax.set_xticks(tick_indices)
    # Format dates to show only the date part
    #date_labels = [pd.to_datetime(dates.iloc[i]).strftime('%Y-%m-%d') for i in tick_indices]
    # Subtract one day from the first date label
    #first_date = pd.to_datetime(dates.iloc[tick_indices[0]]) - pd.Timedelta(days=1)
    #date_labels[0] = first_date.strftime('%Y-%m-%d')    
    
    
    
    #ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=11.5)
    
    # Adjust legend position based on variable for MAP plots
    if marginal_variable_to_use == 'acf0':
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))
    elif marginal_variable_to_use == 'acf1':
            ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))
    elif marginal_variable_to_use == 'mu':
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))
    elif marginal_variable_to_use == 'beta':
        ax.legend(loc='lower right')
    else:  # sigma
        ax.legend(loc='upper right')
    ax.set_title(tex_mapping[marginal_variable_to_use], fontsize=14)
    
        
        
        
if __name__ == '__main__':
    
    q = [0.025,0.975]
    colors = {1000: '#0073C2', 1500: '#EFC000', 2000: '#CD534C'}

    step_size = 1500
    
    dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data','brain')#,'des') 
    data = pd.read_csv(os.path.join(dataset_path, 'application_data.csv'))
                                    #'VIX_25_y.csv'))['resid']#'normalized_temperature_data1.csv'))
                                    
    #data.set_index('Datetime',inplace=True)
    
    first_end = 2000
    ends = np.arange(first_end, len(data), step_size)
    samples_dict = dict()
    MAP_dict = dict()
    
    
    for seq_len in (1000, 1500, 2000):   
        samples_dict[seq_len] = np.load(os.path.join(dataset_path,f'results_{seq_len}.npy'))
        MAP_dict[seq_len] = np.load(os.path.join(dataset_path,f'results_MAP_{seq_len}.npy'))
    
    # Create combined figure for confidence intervals
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.25))
    fig.suptitle('95% Credible Intervals', y=0.98, fontsize=16.5)
    
    plot_marginal_variable_with_confidence_regions(samples_dict, 'mu', step_size, q, axes[0])
    plot_marginal_variable_with_confidence_regions(samples_dict, 'beta', step_size, q, axes[1])
    plot_marginal_variable_with_confidence_regions(samples_dict, 'sigma', step_size, q, axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path,'CI_for_marginal_params.pdf'), dpi=900, bbox_inches='tight')
    plt.show()
    
    
    # Create combined figure for MLE/MAP
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.25))
    fig.suptitle('MAP Estimates', y=0.98, fontsize=14)
    
    plot_MLE(MAP_dict, 'mu', step_size, axes[0])
    plot_MLE(MAP_dict, 'beta', step_size, axes[1])
    plot_MLE(MAP_dict, 'sigma', step_size, axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path,'MAP_for_marginal_params.pdf'), dpi=900, bbox_inches='tight')
    plt.show()
        
    f,ax = plt.subplots()
    
    for k in [5,50, 115, 155]:
        data_to_use = data.iloc[2000 + 1500 * (k-1) :2000  + 1500 * k].T3
        nlags = 20
        H = np.arange(0,nlags+1,1)
            
            
        def corr_sup_ig_envelope(h, params):
            gamma, eta = params
            return np.exp(eta * (1-np.sqrt(2*h/gamma**2+1)))
        
        acf_params = MAP_dict[2000][k,:2]
        theoretical_acf = corr_sup_ig_envelope(H, acf_params)
        
        ax.plot(H, acf(data_to_use, nlags = nlags), label = f'empirical {k}',linestyle='--',)
        ax.plot(H, theoretical_acf, label = f'infered {k}')
    plt.legend()
        #plt.set_title(f'{k}')
    plt.show()
##########################    
    # Choose a color palette
    import seaborn as sns

    palette = sns.color_palette("tab10", len([5, 50, 115, 155]))
    k_values = [5, 50, 115, 155]
    
    f, ax = plt.subplots(figsize=(10, 6))
    
    for idx, k in enumerate(k_values):
        color = palette[idx]  # Same color for both lines of this k
        
        # Extract data segment
        data_to_use = data.iloc[2000 + 1500 * (k-1): 2000 + 1500 * k].T3
        
        # Compute empirical ACF
        empirical_acf = acf(data_to_use, nlags=nlags)
        
        # Compute theoretical ACF
        acf_params = MAP_dict[2000][k, :2]
        theoretical_acf = corr_sup_ig_envelope(H, acf_params)
        
        # Plot both with matching colors
        ax.plot(H, empirical_acf, 
                label=f'Empirical {k}', 
                linestyle='--', color=color, alpha=0.7, linewidth=2)
        ax.plot(H, theoretical_acf, 
                label=f'Inferred {k}', 
                linestyle='-', color=color, alpha=0.9, linewidth=2)
    
    # Beautify plot
    ax.set_title("Empirical vs Inferred ACF", fontsize=14)
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("ACF", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()
    
    ###############
    from src.utils.KL_divergence import convert_3_to_4_param_nig
    import tensorflow_probability.substrates.jax as tfp
    tfp_dist = tfp.distributions
    
    f, ax = plt.subplots(figsize=(10, 6))
    
    for idx, k in enumerate(k_values):
        color = palette[idx]
    
        # Extract data segment
        data_to_use = data.iloc[2000 + 1500 * (k-1):2000 + 1500 * k].T3
        
        # KDE plot
        sns.kdeplot(
            data_to_use, ax=ax, color=color, alpha=0.7, 
            linewidth=2, linestyle='--', label=f'KDE {k}'
        )
        
        # Inferred density
        marginal_plot_x_range = np.linspace(-5, 200, 1000)
        tf_params = convert_3_to_4_param_nig(MAP_dict[2000][k, 2:])
        prob = tfp_dist.NormalInverseGaussian(
            *tf_params, validate_args=True
        ).prob(marginal_plot_x_range)
        
        ax.plot(
            marginal_plot_x_range, prob, color=color, alpha=0.9, 
            linewidth=2,  label=f'Inferred {k}'
        )
    
    # Beautify plot
    ax.set_title("Empirical KDE vs Inferred Density", fontsize=14)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()
    
    
    import jax
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow_probability.substrates.jax as tfp
    tfp_dist = tfp.distributions
    
    k_values = [5, 50, 115, 155]
    palette = sns.color_palette("tab10", len(k_values))
    
    f, ax = plt.subplots(figsize=(10, 6))
    
    for idx, k in enumerate(k_values):
        color = palette[idx]
        
        # Extract data
        data_to_use = data.iloc[2000 + 1500 * (k-1):2000 + 1500 * k].T3.values
        
        # Inferred distribution
        tf_params = convert_3_to_4_param_nig(MAP_dict[2000][k, 2:])
        nig_dist = tfp_dist.NormalInverseGaussian(*tf_params, validate_args=True)
        
        # Sample with JAX PRNGKey
        key = jax.random.PRNGKey(0)
        samples = np.array(nig_dist.sample(10**5, seed=key))
        
        # Quantiles
        q_data = np.quantile(data_to_use, np.linspace(0, 1, 500))
        q_model = np.quantile(samples, np.linspace(0, 1, 500))
        
        # QQ plot
        ax.plot(q_model, q_data, linestyle='-', color=color, alpha=0.9, linewidth=2,
                label=f'QQ {k}')
        ax.plot([q_model.min(), q_model.max()], [q_model.min(), q_model.max()],
                color=color, linestyle='--', alpha=0.5, linewidth=1)
    
    # Beautify
    ax.set_title("QQ Plots: Empirical vs Inferred NIG", fontsize=14)
    ax.set_xlabel("Model Quantiles", fontsize=12)
    ax.set_ylabel("Empirical Quantiles", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()
