# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:29:30 2025

@author: dleon
"""
import jax
import jax.numpy as jnp
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.acf_functions import get_acf
from src.utils.weighted_GMM_acf import estimate_acf_parameters
import matplotlib.pyplot as plt
import numpy as np


def plot_acfs(trawl, theta_acf, pred_theta, config, initial_guess=None):

    if initial_guess is None:
        initial_guess = np.array(theta_acf)

    trawl_func_name = config['trawl_config']['acf']
    nlags = config['loss_config']['nr_acf_lags']
    # trawl_type  = trawl_config['trawl_type']

    empirical_acf = compute_empirical_acf(np.array(trawl), nlags=nlags)[1:]

    # theoretical and inffered acfs
    acf_func = get_acf(trawl_func_name)
    H = np.arange(1, nlags+1)
    theoretical_acf = acf_func(H, theta_acf)
    inferred_acf = acf_func(H, pred_theta)

    f, ax = plt.subplots()
    ax.plot(H, theoretical_acf, label='theoretical')
    ax.plot(H, inferred_acf, label='inferred')
    ax.plot(H, empirical_acf, label='empirical')

    try:
        gmm_params = estimate_acf_parameters(
            trawl, config, initial_guess=np.array(theta_acf))  # num_lags=num_lags, trawl_function_name=trawl_function_name)
        gmm_acf = acf_func(H, gmm_params)
        ax.plot(H, gmm_acf, label='gmm')

        plt.legend()
        return f

    except:

        plt.legend()
        return f


def plot_marginal_distributions(trawl, theta_marginal, pred_theta, config):
    pass
