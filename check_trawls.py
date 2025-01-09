# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:14:42 2025

@author: dleon
"""

from jax.random import PRNGKey
import jax.numpy as jnp
import numpy as np
import datetime
import jax
import os
import pickle
import yaml
from src.utils.get_data_generator import get_theta_and_trawl_generator
if True:
    from path_setup import setup_sys_path
    setup_sys_path()


trial_config_file_path = os.path.join(
    'config_files', 'summary_statistics', 'trial_config.yaml')


# Load config
with open(trial_config_file_path, 'r') as f:
    config = yaml.safe_load(f)


###########################################################################
# Get params and hyperparams
learn_config = config['learn_config']
learn_acf = learn_config['learn_acf']
learn_marginal = learn_config['learn_marginal']
learn_both = learn_config['learn_both']

assert learn_acf + learn_marginal == 1 and learn_both == False


trawl_config = config['trawl_config']
batch_size = trawl_config['batch_size']

###########################################################################
# Get data generators
theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
    config)

key = jax.random.PRNGKey(np.random.randint(1, 1000))
key = jax.random.split(key, batch_size)

theta_acf, key = theta_acf_simulator(key)
theta_marginal_jax, theta_marginal_tf, key = theta_marginal_simulator(
    key)

trawl_means_, trawl_stds_ = [], []
for i in range(50):
    if i % 10 == 0:
        print(i)
    trawl, key = trawl_simulator(theta_acf, theta_marginal_tf, key)
    trawl_means_.append(jnp.mean(trawl, axis=1))
    trawl_stds_.append(jnp.std(trawl, axis=1))

trawl_means_ = np.array(trawl_means_)
trawl_stds_ = np.array(trawl_stds_)

trawl_means = np.mean(trawl_means_, axis=0)
trawl_stds = np.mean(trawl_stds_, axis=0)


theoretical_mean = theta_marginal_jax[:, 0]
theoretical_std = theta_marginal_jax[:, 1]


# CHECK MARGINAL GMM


# CHECK ACF GMM
