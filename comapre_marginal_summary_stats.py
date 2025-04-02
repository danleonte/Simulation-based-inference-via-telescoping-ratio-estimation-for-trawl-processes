# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 21:48:51 2025

@author: dleon
"""
from tqdm import tqdm
import jax.numpy as jnp
from functools import partial
from jax.random import PRNGKey
from flax.training import train_state
from src.utils.trawl_training_utils import loss_functions_wrapper
from src.utils.classifier_utils import get_projection_function
from src.utils.KL_divergence import vec_monte_carlo_kl_3_param_nig
import numpy as np
import datetime
import time
import pickle
import wandb
import yaml
import jax
import os
import matplotlib
import pandas as pd


@partial(jax.jit, static_argnames=('num_KL_samples'))
def get_marginal_loss_functions(trawl, true_marginal_theta, rng, num_KL_samples=15000):

    pred_marginal_theta = project_trawl(trawl)[:, 2:]
    batch_size = pred_marginal_theta.shape[0]

    ### MAE and MSE loss ###
    # MAE_loss = jnp.mean(
    #    jnp.abs(true_marginal_theta - pred_marginal_theta), axis=1)
    # MAE_loss = jnp.mean(MAE_loss)
    MAE_loss = jnp.mean(
        jnp.abs(true_marginal_theta - pred_marginal_theta), axis=0)

    # MSE_loss = jnp.mean(
    #    jnp.abs(true_marginal_theta - pred_marginal_theta)**2, axis=1)
    # MSE_loss = jnp.mean(MSE_loss)
    MSE_loss = jnp.mean(
        jnp.abs(true_marginal_theta - pred_marginal_theta)**2, axis=0)

    ### KL and rev KL div ###
    split_dropout_rng = jax.random.split(rng, batch_size)

    kl_samples, _ = vec_monte_carlo_kl_3_param_nig(
        true_marginal_theta,
        pred_marginal_theta,
        split_dropout_rng,
        num_KL_samples
    )

    rev_kl_samples, _ = vec_monte_carlo_kl_3_param_nig(
        pred_marginal_theta,
        true_marginal_theta,
        jax.random.split(split_dropout_rng[-1], batch_size),
        num_KL_samples
    )

    kl = jnp.mean(kl_samples)
    rev_kl = jnp.mean(rev_kl_samples)

    return MAE_loss, MSE_loss, kl, rev_kl


path_ = os.path.join('sym', 'best_model')
project_trawl = get_projection_function(path_)
trawls = np.load(os.path.join('models', 'summary_statistics', 'trawls.npy'))
thetas = np.load(os.path.join(
    'models', 'summary_statistics', 'thetas_marginal.npy'))

jax_rng = PRNGKey(1340)
result = []
for i in tqdm(range(trawls.shape[0])):
    result_to_add = get_marginal_loss_functions(trawls[i], thetas[i], jax_rng)
    result_to_add = list(
        result_to_add[:2]) + [result_to_add[2].item(), result_to_add[3].item()]
    result.append(result_to_add)
    jax_rng = jax.random.split(jax_rng)[0]


# means, st_dev = np.mean(result, axis=0), np.std(
#    result, axis=0) / result.shape[0]**0.5
# df = pd.DataFrame([np.concatenate([means,  st_dev])], columns=(
#    'MAE', 'rMSE', 'kl', 'rev_kl'))#, 's_MAE', 's_rMSE', 's_kl', 's_rev_kl'))
MAE = np.mean([i[0] for i in result], axis=0)
rMSE = np.mean([i[1] for i in result], axis=0)**0.5
kl = np.mean([i[2] for i in result])
rev = np.mean([i[3] for i in result])
means = [MAE, rMSE, kl, rev]

df = pd.DataFrame([means], columns=('MAE', 'rMSE', 'kl', 'rev_kl'))
# df.s_rMSE = df.s_rMSE**0.5

df.to_csv(os.path.join('models', 'summary_statistics',
          'learn_marginal', path_, 'stats.csv'))
