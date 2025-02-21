# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:07:43 2025

@author: dleon
"""

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
import arviz as az

numpyro.set_host_device_count(3)
# Define the model


def model():
    x1 = numpyro.sample("x1", dist.Uniform(10, 20))
    x2 = numpyro.sample("x2", dist.Uniform(10, 20))
    x3 = numpyro.sample("x3", dist.Uniform(-1, 1))
    x4 = numpyro.sample("x4", dist.Uniform(0.5, 1.5))
    x5 = numpyro.sample("x5", dist.Uniform(-5, 5))

    params = jnp.array([x1, x2, x3, x4, x5])
    numpyro.factor("likelihood", approximate_log_likelihood_to_evidence(
        params)[0])  # Include log-likelihood in inference


rng_key = random.PRNGKey(0)
nuts_kernel = NUTS(model, adapt_step_size=True)  # Adaptive HMC
mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=500000, num_chains=2)
mcmc.run(rng_key)

# Extract and visualize posterior samples
posterior_samples = mcmc.get_samples(group_by_chain=True)
az_data = az.from_numpyro(mcmc)
az.plot_trace(az_data)
