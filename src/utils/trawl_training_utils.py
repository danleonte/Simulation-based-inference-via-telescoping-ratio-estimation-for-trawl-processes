# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:45:23 2025

@author: dleon
"""
import jax
import jax.numpy as jnp
from functools import partial
from src.utils.acf_functions import get_acf
from src.utils.KL_divergence import vec_monte_carlo_kl_3_param_nig


if True:
    from path_setup import setup_sys_path
    setup_sys_path()


def get_pred_theta_acf_from_nn(pred_theta, trawl_func_name):

    if trawl_func_name in ('sup_IG', 'exponential'):

        return jnp.exp(pred_theta)

    else:
        raise ValueError


def get_pred_theta_marginal_from_nn(pred_theta, marginal_distr, trawl_type):

    if marginal_distr == 'NIG' and trawl_type == 'sup_ig_nig_5p':

        return pred_theta.at[:, 1].set(jnp.exp(pred_theta[:, 1]))

    else:

        raise ValueError('not yet implemented')


def apply_transformation_to_trawl():
    pass


def loss_functions_wrapper(state, config):

    learn_config = config['learn_config']
    learn_acf = learn_config['learn_acf']
    learn_marginal = learn_config['learn_marginal']

    ####
    trawl_config = config['trawl_config']
    batch_size = trawl_config['batch_size']

    ####
    loss_config = config['loss_config']
    p = loss_config['p']

    # acf hyperparams
    trawl_func_name = trawl_config['acf']
    use_acf_directly = loss_config['use_acf_directly']
    nr_acf_lags = loss_config['nr_acf_lags']
    acf_func = jax.jit(
        jax.vmap(get_acf(trawl_config['acf']), in_axes=(None, 0)))

    # marginal hyperparams
    use_kl_div = loss_config['use_kl_div']

    ###########################################################################
    #     Helper function to preict theta (potentially on log scale etc)      #
    ###########################################################################

    @partial(jax.jit, static_argnames=('train', 'learn_acf'))
    def predict_theta(params, trawl, dropout_rng, train, learn_acf):

        if learn_acf:

            trawl = (trawl - jnp.mean(trawl, axis=1, keepdims=True)) / \
                jnp.std(trawl, axis=1, keepdims=True)

        # can add another elif learn_marginal and do the transformation

        if train:

            pred_theta = state.apply_fn(
                params,
                trawl,
                train=True,
                rngs={'dropout': dropout_rng}
            )

        else:

            pred_theta = state.apply_fn(
                params,
                trawl,
                train=False
            )

        return pred_theta

    ###########################################################################
    #                      Loss function helpers                              #
    ###########################################################################

    @jax.jit
    def _acf_loss(true_theta, pred_theta):
        """Compute ACF-based loss."""
        H = jnp.arange(1, nr_acf_lags + 1)
        pred_acf = acf_func(H, pred_theta)
        theoretical_acf = acf_func(H, true_theta)
        # return jnp.mean(jnp.abs((pred_acf - theoretical_acf))**p)**(1 / p)
        l_p_norms = jnp.mean(
            jnp.abs((pred_acf - theoretical_acf))**p, axis=1)**(1/p)
        return jnp.mean(l_p_norms)

    @jax.jit
    def _direct_params_loss(true_theta, pred_theta):
        """Compute direct parameter-based loss."""
        l_p_norms = jnp.mean(
            jnp.abs(true_theta - pred_theta)**p, axis=1)**(1/p)
        return jnp.mean(l_p_norms)

    ###########################################################################
    # Allow for different loss functions for learning acf and marginal params #
    ###########################################################################

    if learn_acf:

        # subcase: when learning acf params, can either compare acfs or params

        @partial(jax.jit, static_argnames=('train'))
        def compute_loss(params, trawl, theta_acf, dropout_rng, train):

            pred_theta = predict_theta(params, trawl, dropout_rng, train, True)

            # if params are strictly positive or similar, we learn on log-scale
            pred_theta = get_pred_theta_acf_from_nn(
                pred_theta, trawl_func_name)

            if use_acf_directly:

                return _acf_loss(theta_acf, pred_theta)

            else:

                return _direct_params_loss(theta_acf, pred_theta)

        compute_loss_and_grad = jax.jit(jax.value_and_grad(
            compute_loss), static_argnames=('train',))

    elif learn_marginal:

        marginal_distr = trawl_config['marginal_distr']
        trawl_type = trawl_config['trawl_type']

        @partial(jax.jit, static_argnames=('train', 'num_KL_samples'))
        def compute_loss(params, trawl, theta_marginal, dropout_rng, train, num_KL_samples):

            pred_theta = predict_theta(params, trawl, dropout_rng, train, True)
            # check if we predict parameters on the log scale etc
            pred_theta = get_pred_theta_marginal_from_nn(
                pred_theta, marginal_distr, trawl_type)

            # KL key, if using MC approximation
            KL_key = jax.random.split(dropout_rng, batch_size)

            # if we don't use KL div, and instead compare params directly
            if not loss_config['use_kl_div']:

                return _direct_params_loss(theta_marginal, pred_theta)

            # can assume we're using KL div here
            if marginal_distr == 'NIG' and trawl_type == 'sup_ig_nig_5p':

                kl_loss = vec_monte_carlo_kl_3_param_nig(
                    theta_marginal,
                    pred_theta,
                    KL_key,
                    num_KL_samples
                )

                return jnp.mean(kl_loss)

            else:
                raise ValueError

        compute_loss_and_grad = jax.jit(jax.value_and_grad(
            compute_loss), static_argnames=('train', 'num_KL_samples'))

    ###########################################################################
    #                    Validation function                                  #
    ###########################################################################

    if learn_acf:

        @jax.jit
        def compute_validation_stats(params, val_trawls, val_thetas_acf, dropout_rng):

            # ADD DROPOUT RNG HERE
            def body_fun(i, acc, dropout_rng):

                theta_val = jax.lax.dynamic_slice_in_dim(
                    val_thetas_acf, i, 1)[0]
                trawl_val = jax.lax.dynamic_slice_in_dim(
                    val_thetas_acf, i, 1)[0]
                loss = compute_loss(params, trawl_val,
                                    theta_val, dropout_rng, False)

                return acc + jnp.array([loss, loss**2])

            total = jax.lax.fori_loop(
                0, val_trawls.shape[0], body_fun, jnp.zeros(2)
            )

            n = val_trawls.shape[0]
            mean = total[0] / n
            variance = (total[1] / n) - (mean**2)
            std = jnp.sqrt(jnp.maximum(variance, 0.0))

            return mean, std

    elif learn_marginal:

        @partial(jax.jit, static_argnames=('num_KL_samples'))
        def compute_validation_stats(params, val_trawls, val_thetas_marginal, num_KL_samples):

            def body_fun(i, acc):

                theta_val = jax.lax.dynamic_slice_in_dim(
                    val_thetas_marginal, i, 1)[0]
                trawl_val = jax.lax.dynamic_slice_in_dim(
                    val_thetas_marginal, i, 1)[0]
                loss = compute_loss(params, trawl_val, theta_val, False)

                return acc + jnp.array([loss, loss**2])

            total = jax.lax.fori_loop(
                0, val_trawls.shape[0], body_fun, jnp.zeros(2)
            )

            n = val_trawls.shape[0]
            mean = total[0] / n
            variance = (total[1] / n) - (mean**2)
            std = jnp.sqrt(jnp.maximum(variance, 0.0))

            return mean, std

    return compute_loss, compute_loss_and_grad, compute_validation_stats
