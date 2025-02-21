# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:58:07 2025

@author: dleon
"""
import time
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC  # NUTS
import jax.numpy as jnp
from jax.random import PRNGKey
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function
from src.model.Extended_model_nn import ExtendedModel
from netcal.presentation import ReliabilityDiagram
from scipy.optimize import minimize
import numpy as np
import datetime
import pickle
import optax
import wandb
import yaml

import jax
import os
import netcal
import matplotlib
from src.utils.summary_statistics_plotting import plot_acfs, plot_marginals
import tensorflow_probability.substrates.jax as tfp
import corner

if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt

######### Inputs ##########

folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\TRE_summary_statistics\trial_set1'
trawl_process_type = 'sup_ig_nig_5p'
use_tre = True
use_summary_statistics = True
generate_dataset = True

assert use_summary_statistics


##############################################################

################### GENERATE DATA SET #######################################
if use_tre:
    classifier_config_file_path = os.path.join(
        folder_path, 'acf', 'config.yaml')
else:
    classifier_config_file_path = os.path.join(folder_path, 'config.yaml')


with open(classifier_config_file_path, 'r') as f:
    # an arbitrary config gile; if using TRE, can
    acf_config_file = yaml.safe_load(f)


true_trawls_path = os.path.join(folder_path, 'true_trawls.npy')
true_thetas_path = os.path.join(folder_path, 'true_thetas.npy')
true_summaries_path = os.path.join(folder_path, 'true_summaries.npy')

if os.path.isfile(true_trawls_path) and os.path.isfile(true_thetas_path) and os.path.isfile(true_summaries_path):

    true_trawls = np.load(true_trawls_path)
    true_thetas = np.load(true_thetas_path)
    true_summaries = np.load(true_summaries_path)

else:

    batch_size = acf_config_file['trawl_config']['batch_size']
    key = jax.random.split(
        PRNGKey(np.random.randint(low=1, high=10**6)), batch_size)

    # Get data generators
    theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
        acf_config_file)

    true_theta_acf_, key = theta_acf_simulator(key)
    true_theta_marginal_jax_, true_theta_marginal_tf_, key = theta_marginal_simulator(
        key)
    true_trawls, key = trawl_simulator(
        true_theta_acf_, true_theta_marginal_tf_, key)

    true_thetas = jnp.concatenate(
        [true_theta_acf_, true_theta_marginal_jax_], axis=1)

    np.save(file=true_trawls_path, arr=true_trawls)
    np.save(file=true_thetas_path, arr=true_thetas)

    if use_summary_statistics:

        project_trawl = get_projection_function()

        true_summaries = project_trawl(true_trawls)

        np.save(file=true_summaries_path, arr=true_summaries)


#####################    LOAD MODELS    #######################################
true_iteration = 19
true_trawl = true_trawls[true_iteration, :]
true_theta = true_thetas[true_iteration, :]
true_s = true_summaries[true_iteration, :]
approximate_log_likelihood_to_evidence, approximate_log_posterior, _ = \
    load_trained_models(folder_path, true_trawls[:, ::-1], trawl_process_type,  # [::-1] not necessary, it s just a dummy, but just to make sure we don t pollute wth true values of some sort
                        use_tre, use_summary_statistics)

assert not _


def model():
    eta = numpyro.sample("eta", dist.Uniform(10, 20))
    gamma = numpyro.sample("gamma", dist.Uniform(10, 20))
    mu = numpyro.sample("mu", dist.Uniform(-1, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0.5, 1.5))
    beta = numpyro.sample("beta", dist.Uniform(-5, 5))

    params = jnp.array([eta, gamma, mu, sigma, beta])
    numpyro.factor("likelihood", approximate_log_likelihood_to_evidence(true_s[jnp.newaxis, :],
                                                                        params[jnp.newaxis, :])[0])  # Include log-likelihood in inference


def model2():
    eta = numpyro.sample("eta", dist.Uniform(10, 20))
    gamma = numpyro.sample("gamma", dist.Uniform(10, 20))
    mu = numpyro.sample("mu", dist.Uniform(-1, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0.5, 1.5))
    beta = numpyro.sample("beta", dist.Uniform(-5, 5))

    params = jnp.array([eta, gamma, mu, sigma, beta])[jnp.newaxis, :]
    batch_size = params.shape[0]  # Should be `num_chains`
    x_tiled = jnp.tile(true_s, (batch_size, 1))
    numpyro.factor("likelihood", approximate_log_likelihood_to_evidence(x_tiled,
                                                                        params))  # Include log-likelihood in inference


num_samples = 100000
num_warmup = 3000
num_chains = 20  # Vectorized MCMC

rng_key = jax.random.PRNGKey(42)
chain_keys = jax.random.split(rng_key, num_chains)

hmc_kernel = HMC(
    model,
    step_size=0.1,            # Initial step size (will be adapted)
    adapt_step_size=True,     # Enables step size adaptation
    adapt_mass_matrix=True,   # Enables mass matrix adaptation
    dense_mass=True,          # Uses a dense mass matrix (full covariance)
)

hmc_kernel2 = HMC(
    model2,
    step_size=0.1,            # Initial step size (will be adapted)
    adapt_step_size=True,     # Enables step size adaptation
    adapt_mass_matrix=True,   # Enables mass matrix adaptation
    dense_mass=True,          # Uses a dense mass matrix (full covariance)
)

mcmc = MCMC(
    hmc_kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=1,
    # chain_method = 'vectorized',
    progress_bar=False
)

mcmc2 = MCMC(
    hmc_kernel2,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    chain_method='vectorized',
    progress_bar=False
)

start_time = time.time()
mcmc.run(rng_key)  # chain_method="vectorized")
end_time = time.time()

print(start_time - end_time)
start_time2 = time.time()
mcmc2.run(chain_keys)  # chain_method="vectorized")
end_time2 = time.time()
print(start_time2 - end_time2)
