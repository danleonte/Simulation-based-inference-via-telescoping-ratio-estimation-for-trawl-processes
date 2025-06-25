# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 23:03:04 2025
@author: dleon
"""
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import scienceplots
import numpy as np
plt.style.use(['science'])

tfd = tfp.distributions
norminvgauss = tfd.NormalInverseGaussian

# only param is beta
jax_beta = jnp.array((-1., -1, 0, 0, 1, 1))
jax_gamma = jnp.array((2, 3, 2, 3, 2, 3.))
jax_alpha = jnp.sqrt(jax_beta**2+jax_gamma**2)
jax_scale = jax_gamma**3 / jax_alpha**2
jax_loc = - jax_beta * jax_scale / jax_gamma

# check means and st dev are 0 and 1 respectively
distributions = norminvgauss(
    loc=jax_loc, scale=jax_scale, tailweight=jax_alpha, skewness=jax_beta)
jnp.allclose(jnp.array(distributions.mean()), jnp.zeros(
    len(jax_beta)), rtol=1e-03, atol=1e-03)
jnp.allclose(jnp.array(distributions.stddev()),
             jnp.ones(len(jax_beta)), rtol=1e-03, atol=1e-03)

x = jnp.linspace(-3, 3, 500)
y = distributions.prob(x[:, jnp.newaxis])
y = jnp.array(y).transpose()

# Create single figure
plt.figure()
ax = plt.gca()

# Define colors for the 6 combinations
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i in range(len(jax_beta)):
    # Create label with just the values
    label = '{:.0f}, {:.0f}'.format(jax_beta[i], jax_gamma[i])
    ax.plot(x, y[i], color=colors[i], label=label, linewidth=1.5)

# Add axis labels
ax.set_xlabel(r'$x$')
ax.set_ylabel('Probability Density')
ax.set_title(r'NIG density for different values of $\beta$ and $\gamma$')

# Add grid for readability
ax.grid(True, alpha=0.3)

# Set axis limits to match the image better
ax.set_xlim(-3, 3)
ax.set_ylim(0, 0.5)

# Legend with column headers
plt.legend(loc='upper right', fontsize=8, title=r'$\beta$, $\gamma$')
plt.tight_layout()
plt.savefig('nig_beta_gamma_distributions.pdf', bbox_inches='tight')
plt.show()
