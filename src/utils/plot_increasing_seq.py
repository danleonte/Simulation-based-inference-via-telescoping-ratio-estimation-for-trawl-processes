# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 00:54:01 2025

@author: dleon
"""

import scienceplots
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.weighted_GMM_marginal import estimate_jax_parameters, transform_to_constrained_jax
from src.utils.weighted_ACF_GMM import estimate_acf_parameters_transformed
from src.utils.acf_functions import get_acf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
import seaborn as sns
from src.utils.KL_divergence import convert_3_to_4_param_nig
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.numpy as jnp
tfp_dist = tfp.distributions
plt.style.use(['science'])


def load_dataset_NRE_TRE():

    trawl_process_type = 'sup_ig_nig_5p'
    use_summary_statistics = 'False'
    use_tre_TRE = True
    use_tre_NRE = False
    use_summary_statistics = False

    models_path = os.path.join(os.getcwd(), 'models')
    dataset_path = os.path.join(models_path, 'val_dataset', 'val_dataset_2000')

    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    val_x_2000 = np.load(val_x_path, mmap_mode='r')[:5].reshape(-1, 2000)
    val_thetas = np.load(val_thetas_path, mmap_mode='r')[:5].reshape(-1, 5)

    TRE_path = os.path.join(models_path, 'new_classifier',
                            'TRE_full_trawl', 'selected_models')
    NRE_path = os.path.join(models_path, 'new_classifier',
                            'NRE_full_trawl', '04_12_04_25_37', 'best_model')

    NRE_dict = dict()
    TRE_dict = dict()

    for seq_len in (1000, 1500, 2000):

        ######### TRE #########
        _, wrapper_for_approx_likelihood_just_theta_TRE = load_trained_models(
            TRE_path, val_x_2000[[0], :seq_len], trawl_process_type,
            use_tre_TRE, use_summary_statistics, f'beta_calibration_{seq_len}.pkl'
        )

        TRE_dict[seq_len] = wrapper_for_approx_likelihood_just_theta_TRE

        ######### NRE #########
        _, wrapper_for_approx_likelihood_just_theta_NRE = load_trained_models(
            NRE_path, val_x_2000[[0], :seq_len], trawl_process_type,
            use_tre_NRE, use_summary_statistics, f'no_calibration.pkl'  # doesn't matter for NRE
        )
        NRE_dict[seq_len] = wrapper_for_approx_likelihood_just_theta_NRE

    return val_x_2000, val_thetas, NRE_dict, TRE_dict


def get_MLE(trawl_with_appropriate_length, true_theta, appropriate_wrapper):

    log_like = appropriate_wrapper(
        jnp.array(trawl_with_appropriate_length)[jnp.newaxis, :])

    # Define a function that returns a scalar by indexing or using item()
    def scalar_log_like(theta):
        result = log_like(theta)
        # Extract the scalar from the (1,) shape array
        return result[0]  # or result.sum() or result.item()

    # Use value_and_grad on the scalar function
    like_and_grad = jax.jit(jax.value_and_grad(scalar_log_like))

    def minus_like_with_grad(theta_np):
        theta_jax = jnp.array(theta_np)
        value, gradient = like_and_grad(theta_jax)
        return -float(value), -np.array(gradient)

    func_to_optimize = minus_like_with_grad  # (trawl_with_appropriate_length)
    result_from_true = minimize(func_to_optimize, np.array(true_theta),
                                method='L-BFGS-B', jac=True, bounds=((10, 20), (10, 20), (-1, 1), (0.5, 1.5), (-5, 5)))

    return result_from_true


if __name__ == '__main__':

    val_x_2000, val_thetas, NRE_dict, TRE_dict = load_dataset_NRE_TRE()
    idx_to_use = -10  # 71  # 19  # 0
    add_NRE_in_plot_1 = True

    #### PLOT 1: increasing seq_len with TRE / NRE #####
    trawl_plot_1 = val_x_2000[idx_to_use][:1500]
    theta_plot_1 = val_thetas[idx_to_use]

    # TRE MLE
    TRE_jax_params = get_MLE(trawl_plot_1, theta_plot_1, TRE_dict[1500]).x
    TRE_tf_params = convert_3_to_4_param_nig(TRE_jax_params[2:])
    TRE_acf_params = TRE_jax_params[:2]

    # NRE MLE
    NRE_jax_params = get_MLE(trawl_plot_1, theta_plot_1, NRE_dict[1500]).x
    NRE_tf_params = convert_3_to_4_param_nig(NRE_jax_params[2:])
    NRE_acf_params = NRE_jax_params[:2]

    # Create figure with improved styling
    # smaller size + constrained_layout
    fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
    # gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.05)
    # Define color palette
    colors = {
        'true': '#2E86AB',
        'gmm': '#A23B72',
        'TRE': '#F18F01',
        'NRE': '#C73E1D',
        'empirical': '#6A994E',
        'trawl': '#1F77B4',
        # 'kde': '#6A994E'
    }

    # Subplot 1: Trawl Process Time Series
    ax1 = fig.add_subplot(gs[0])
    ax1.margins(x=0.02)
    time_points = np.arange(len(trawl_plot_1))
    ax1.plot(time_points, trawl_plot_1,
             color=colors['trawl'], linewidth=0.8, alpha=0.9)
    ax1.set_xlabel('Time')  # , fontsize=12)#, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12)  # , fontweight='bold')
    # fontweight='bold', pad=15 fontsize=14,)
    ax1.set_title('Trawl Process Realization',  pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Subplot 2: Marginal Distribution Comparison
    ax2 = fig.add_subplot(gs[1])
    ax2.margins(x=0.02)
    marginal_plot_x_range = np.linspace(-2.5, 6, 1000)

    # sns.kdeplot(trawl_plot_1, ax=ax2, label='kde', color=colors['kde'])

    # True distribution
    true_tf_params = convert_3_to_4_param_nig(theta_plot_1[2:])
    true_prob = tfp_dist.NormalInverseGaussian(
        *true_tf_params, validate_args=True).prob(marginal_plot_x_range)
    ax2.plot(marginal_plot_x_range, true_prob, label='True',
             color=colors['true'], linewidth=2.25, linestyle='--')

    # GMM
    marginal_jax_gmm_params = transform_to_constrained_jax(estimate_jax_parameters(
        trawl_plot_1, theta_plot_1[2:]).params)
    marginal_tf_gmm_params = convert_3_to_4_param_nig(marginal_jax_gmm_params)
    gmm_dist = tfp_dist.NormalInverseGaussian(
        *marginal_tf_gmm_params, validate_args=True)
    gmm_prob = gmm_dist.prob(marginal_plot_x_range)
    ax2.plot(marginal_plot_x_range, gmm_prob, label='GMM',
             color=colors['gmm'], linewidth=1.75)

    # TRE marginal plot
    TRE_prob = tfp_dist.NormalInverseGaussian(
        *TRE_tf_params, validate_args=True).prob(marginal_plot_x_range)
    ax2.plot(marginal_plot_x_range, TRE_prob, label='TRE',
             color=colors['TRE'], linewidth=1.75)

    # NRE marginal plot
    if add_NRE_in_plot_1:
        NRE_prob = tfp_dist.NormalInverseGaussian(
            *NRE_tf_params, validate_args=True).prob(marginal_plot_x_range)
        ax2.plot(marginal_plot_x_range, NRE_prob, label='NRE',
                 color=colors['NRE'], linewidth=1.75)

    # Add histogram of actual data
    ax2.hist(trawl_plot_1, bins=30, density=True, alpha=0.3,
             color='gray', edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Value', fontsize=12)  # , fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12)  # , fontweight='bold')
    # fontweight='bold' fontsize=14,
    ax2.set_title('Marginal Distribution Comparison',  pad=15)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Subplot 3: ACF Comparison
    ax3 = fig.add_subplot(gs[2])
    ax3.margins(x=0.02)
    nlags = 30
    H = np.arange(0, nlags+1)
    acf_func = get_acf('sup_IG')

    # Empirical ACF
    empirical_acf = compute_empirical_acf(
        trawl_plot_1, adjusted=True, nlags=nlags,)
    ax3.plot(H, empirical_acf, 'o-', label='Empirical',
             color=colors['empirical'], markersize=4.5, linewidth=1.5)

    # True ACF
    theoretical_acf = acf_func(H, theta_plot_1[:2])
    ax3.plot(H, theoretical_acf, '--', label='True',
             color=colors['true'], linewidth=2.25)

    # TRE ACF
    TRE_acf = acf_func(H, TRE_acf_params)
    ax3.plot(H, TRE_acf, label='TRE',
             color=colors['TRE'], linewidth=1.75)

    # NRE ACF
    NRE_acf = acf_func(H, NRE_acf_params)
    ax3.plot(H, NRE_acf, label='NRE',
             color=colors['NRE'], linewidth=1.75)

    # GMM ACF
    gmm_acf_params = estimate_acf_parameters_transformed(
        trawl_plot_1, nlags, 'sup_IG',
        lower_bound=10,
        upper_bound=20,
        initial_guess=theta_plot_1[:2]
    )
    gmm_acf = acf_func(H, gmm_acf_params["constrained_params"])
    ax3.plot(H, gmm_acf, label='GMM',
             color=colors['gmm'], linewidth=1.75)

    # Add confidence bands for empirical ACF
    # not done yet

    ax3.set_xlabel('Lag',)  # , fontweight='bold')  fontsize=12
    ax3.set_ylabel('Autocorrelation',  fontsize=12)  # , fontweight='bold') ,
    ax3.set_title('Autocorrelation Function Comparison',
                  pad=15)  # fontweight='bold' fontsize=14,
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_ylim(-0.1, 1.05)

    # Add overall title
    # fig.suptitle(f'Trawl Process Analysis (Sequence Length: 1500)',
    #              y=0.98)#fontweight='bold', fontsize=16,

    # Add parameter information
    # param_text = (f'True Parameters: ={theta_plot_1[0]:.2f}, λ⁻={theta_plot_1[1]:.2f}, '
    #              f'α={theta_plot_1[2]:.2f}, β={theta_plot_1[3]:.2f}, μ={theta_plot_1[4]:.2f}')
    # fig.text(0.5, 0.02, param_text, ha='center', fontsize=11,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92,
                        bottom=0.12, wspace=0.15)

    plt.show()
    plt.savefig(f'increasing_seq_{idx_to_use}.pdf',
                bbox_inches='tight', pad_inches=0.02, dpi=900)
    # plt.tsavefig('',)
    # TO ADD
    #### PLOT 2: seq_len = 1500 and multiple methods #####
