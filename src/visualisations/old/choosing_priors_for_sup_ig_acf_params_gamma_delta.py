# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:06:48 2024

@author: dleon
"""
# set path, othterwise import such as
# from src.module_name import X won't work;
# run only once
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import jax
import jax.numpy as jnp
from src.utils.acf_functions import corr_supp_ig_envelope
from src.utils.get_transformed_distr import get_transformed_beta_distr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

total_nr_lags = 30
lags = jnp.arange(start=1, stop=total_nr_lags + 1)  # assuming tau = 1

gamma_hyperparams = jnp.array([10, 25, 1.25, 1.25])
delta_hyperparams = jnp.array([0.1, 5, 1.25, 1.25])

gamma_key = jax.random.PRNGKey(10)
delta_key = jax.random.PRNGKey(20)

gamma_sampler = get_transformed_beta_distr(gamma_hyperparams)
gamma_samples = gamma_sampler.sample(100, gamma_key)

delta_sampler = get_transformed_beta_distr(delta_hyperparams)
delta_samples = delta_sampler.sample(100, delta_key)

acf_params = jnp.vstack([gamma_samples, delta_samples])

result = jnp.array([corr_supp_ig_envelope(lags[i], acf_params) for i in lags])


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
