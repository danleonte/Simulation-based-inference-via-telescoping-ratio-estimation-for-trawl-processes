# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:29:30 2025

@author: dleon
"""
import tensorflow_probability.substrates.jax as tfp
if True:
    from path_setup import setup_sys_path
    setup_sys_path()


import jax
import jax.numpy as jnp
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.acf_functions import get_acf
from src.utils.weighted_GMM_acf import estimate_acf_parameters
import matplotlib.pyplot as plt
import numpy as np
from src.utils.marginal_GMM import transform_to_tf_params, estimate_jax_parameters
import seaborn as sns
tfp_dist = tfp.distributions
norminvgauss = tfp_dist.NormalInverseGaussian


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
        gmm_result, e_acf_st_dev = estimate_acf_parameters(
            trawl, config, initial_guess=np.array(theta_acf))  # num_lags=num_lags, trawl_function_name=trawl_function_name)
        gmm_acf = acf_func(H, gmm_result.params)
        ax.plot(H, gmm_acf, label='gmm')

        ax.fill_between(H, empirical_acf - e_acf_st_dev / 15, empirical_acf + e_acf_st_dev / 15,
                        color='b', alpha=.2)

        plt.legend()
        return f

    except:

        plt.legend()
        return f


def plot_marginals(trawl, jax_theta_marginal, jax_pred_theta, config):

    trawl_config = config['trawl_config']
    trawl_process_type = trawl_config['trawl_process_type']

    if trawl_process_type == 'sup_ig_nig_5p':

        z = jnp.linspace(-3.75, 3.75, 200)

        mu, delta, gamma, beta = transform_to_tf_params(
            jax_theta_marginal[0], jax_theta_marginal[1], jax_theta_marginal[2])
        pred_mu, pred_delta, pred_gamma, pred_beta = transform_to_tf_params(
            jax_pred_theta[0], jax_pred_theta[1], jax_pred_theta[2])

        alpha = jnp.sqrt(gamma**2+beta**2)
        pred_alpha = jnp.sqrt(pred_gamma**2 + pred_beta**2)

        f, ax = plt.subplots()
        ax.plot(z,  norminvgauss(mu, delta, alpha, beta).prob(z), label='true')
        ax.plot(z, norminvgauss(pred_mu, pred_delta,
                pred_alpha, pred_beta).prob(z), label='pred')
        sns.kdeplot(trawl, fill=False, ax=ax, label='kde')

        try:
            gmm_result = estimate_jax_parameters(
                np.array(trawl), initial_guess=np.array(jax_theta_marginal))
            gmm_mu, gmm_delta, gmm_gamma, gmm_beta = transform_to_tf_params(
                gmm_result.params[0], gmm_result.params[1], gmm_result.params[2])
            # gmm_result['jax_mu'], gmm_result['jax_scale'],
            # gmm_result['jax_beta'])
            ax.plot(z, norminvgauss(gmm_mu.item(), gmm_delta.item(),
                    gmm_gamma.item(), gmm_beta.item()).prob(z), label='gmm')

        except:
            pass

        plt.legend()

        return f

    else:
        raise ValueError
