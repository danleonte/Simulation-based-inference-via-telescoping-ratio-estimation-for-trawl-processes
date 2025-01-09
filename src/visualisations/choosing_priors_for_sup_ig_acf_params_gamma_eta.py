# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:20:07 2024

@author: dleon
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import jax
import jax.numpy as jnp
from src.utils.acf_functions import corr_sup_ig_envelope
from src.utils.get_transformed_distr import get_transformed_beta_distr
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

total_nr_lags = 30
lags = jnp.arange(start=1, stop=total_nr_lags + 1)  # assuming tau = 1


gamma_key = jax.random.PRNGKey(10)
eta_key = jax.random.PRNGKey(20)


distr_name = input("Enter prior distribution: beta or uniform \n")


if distr_name == 'beta':

    gamma_hyperparams = jnp.array([5., 15., 2., 2.])
    eta_hyperparams = jnp.array([5., 10., 2., 2.])

    gamma_sampler = get_transformed_beta_distr(gamma_hyperparams)
    eta_sampler = get_transformed_beta_distr(eta_hyperparams)


elif distr_name == 'uniform':

    gamma_hyperparams = jnp.array([5., 15.])  # jnp.array([5., 15.])
    eta_hyperparams = jnp.array([10., 20.])  # jnp.array([5., 15.])

    gamma_sampler = tfp.distributions.Uniform(
        low=gamma_hyperparams[0], high=gamma_hyperparams[1])
    eta_sampler = tfp.distributions.Uniform(
        low=eta_hyperparams[0], high=eta_hyperparams[1])

else:
    raise ValueError('Input must be either beta or uniform\n')

# get samples
gamma_samples = gamma_sampler.sample(500, gamma_key)
eta_samples = eta_sampler.sample(500, eta_key)

acf_params = jnp.vstack([gamma_samples, eta_samples])

result = jnp.array([corr_sup_ig_envelope(lags[i], acf_params) for i in lags])


# Convert the data into a DataFrame for easier plotting with seaborn
data = {
    "lags": np.repeat(lags, result.shape[1]),
    "values": np.array(result).flatten()
}
df = pd.DataFrame(data)

# Create the boxplot using seaborn
fig, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(data=df, x="lags", y="values", ax=ax,
            showmeans=True, palette="coolwarm", whis=[5, 95])

# Set labels and title
ax.set_xlabel("Lags")
ax.set_ylabel("ACFs")
ax.set_title("Boxplot for each lag of the theoretical ACF Quantiles")
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
