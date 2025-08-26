# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 23:04:22 2025

@author: dleon
"""

import jax
import os
import jax
import yaml
import pickle
import datetime
import time
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.random import PRNGKey
from src.utils.get_model import get_model
from src.utils.acf_functions import get_acf
import pandas as pd

nbe_type = 'sym'  # refers to the type of trained marginal NBE; can be either
# direct, kl, rev or sym; the ACF NBE is the same


acf_model_path = os.path.join(
    'models', '_other_stuff', 'summary_statistics', 'learn_acf', 'best_model')
marginal_model_path = os.path.join(
    'models', '_other_stuff', 'summary_statistics', 'learn_marginal', f'best_model_{nbe_type}')


# load config files
with open(os.path.join(acf_model_path, 'config.yaml'), 'r') as f:
    acf_config = yaml.safe_load(f)

with open(os.path.join(marginal_model_path, 'config.yaml'), 'r') as f:
    marginal_config = yaml.safe_load(f)

# load params
with open(os.path.join(acf_model_path, 'params.pkl'), 'rb') as f:
    acf_params = pickle.load(f)

with open(os.path.join(marginal_model_path, 'params.pkl'), 'rb') as f:
    marginal_params = pickle.load(f)


acf_model, _, __ = get_model(acf_config)
marginal_model, _, _ = get_model(marginal_config)


@jax.jit
def apply_acf_model(x):
    return acf_model.apply(acf_params, x)


@jax.jit
def apply_marginal_model(x):
    return marginal_model.apply(marginal_params, x)

# jit to fuse operations and reduce memory footprint


@jax.jit
def normalize_data(x):
    return (x - jnp.mean(x, axis=1, keepdims=True)) / \
        jnp.std(x, axis=1, keepdims=True)


if __name__ == '__main__':

    NBE_path = os.path.join(
        os.getcwd(), 'models', 'new_classifier', 'point_estimators', 'NBE', nbe_type)
    acf_func = jax.vmap(get_acf('sup_IG'), in_axes=(None, 0))
    num_rows_to_load = 160
    num_lags = 35

    MAE_m = dict()
    MSE_m = dict()
    L1_acf = dict()
    L2_acf = dict()

    for seq_len in (1000, 1500, 2000):

        dataset_path = os.path.join(os.getcwd(), 'models',
                                    'val_dataset', f'val_dataset_{seq_len}')
        val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
        val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

        val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
        val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

        val_x = val_x.reshape(-1, val_x.shape[-1])
        val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])

        mar_path = os.path.join(
            NBE_path, f'{nbe_type}_infered_marginal_{seq_len}.npy')
        if not os.path.exists(mar_path):
            infered_marginal = apply_marginal_model(val_x)
            infered_marginal = infered_marginal.at[:, 1].set(
                jnp.exp(infered_marginal[:, 1]))
            np.save(
                file=mar_path, arr=infered_marginal)

        else:

            infered_marginal = np.load(
                mar_path)

        acf_path = os.path.join(NBE_path, f'infered_theta_acf_{seq_len}.npy')

        if not os.path.exists(acf_path):
            val_x = normalize_data(val_x)
            infered_theta_acf = jnp.exp(apply_acf_model(val_x))
            np.save(
                file=acf_path, arr=infered_theta_acf)

        else:
            infered_theta_acf = np.load(acf_path)

        index_mar = (
            (infered_marginal[:, 0] >= -1) & (infered_marginal[:, 0] <= 1) &
            (infered_marginal[:, 1] >= 0.5) & (infered_marginal[:, 1] <= 1.5) &
            (infered_marginal[:, 2] >= -5) & (infered_marginal[:, 2] <= 5)
        )

        index_acf = (infered_theta_acf < 20) & (infered_theta_acf > 10)
        index_acf = index_acf.all(axis=1)

        df_mar = pd.DataFrame({'true_theta': list(val_thetas[index_mar]),  # exclude outputs that are out of range
                               'point_estimators':  list(infered_marginal[index_mar])
                               })

        df_acf = pd.DataFrame({'true_theta': list(val_thetas[index_acf]),  # exclude outputs that are out of range
                               'point_estimators':  list(infered_theta_acf[index_acf])
                               })

        df_mar.to_pickle(os.path.join(NBE_path, f'df_mar_{seq_len}.pkl'))
        df_acf.to_pickle(os.path.join(NBE_path, f'df_acf_{seq_len}.pkl'))

        if False:
            print('Starting KL divergence calculations')

            MAE_m[seq_len] = jnp.mean(
                jnp.abs(val_thetas[:, 2:] - infered_marginal)[index_mar], axis=0)
            MSE_m[seq_len] = jnp.mean(
                (val_thetas[:, 2:] - infered_marginal)[index_mar]**2, axis=0)**0.5

            H = np.arange(1, num_lags + 1)
            theoretical_acf = acf_func(H, val_thetas[:, :2])
            # infered_theta[:, :2])
            infered_acf = acf_func(H, infered_theta_acf)
            acf_differences = np.abs(theoretical_acf - infered_acf)

            L1_acf[seq_len] = np.mean(acf_differences[index_acf])
            L2_acf[seq_len] = np.mean(
                np.sqrt(np.mean(acf_differences[index_acf]**2, axis=1)))

            del val_x
            del val_thetas
