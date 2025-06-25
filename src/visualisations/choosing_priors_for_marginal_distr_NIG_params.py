# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:29:28 2024
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
jax_beta = jnp.linspace(-5, 5, 10)
jax_gamma = 1 + jnp.abs(jax_beta)/5
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

# Use colormap for natural transition (inverted)
colors = plt.cm.coolwarm(np.linspace(1, 0, len(jax_beta)))

for i in range(len(jax_beta)):
    # Plot all lines but only label every 3rd for cleaner legend
    label = f'{jax_beta[i]:.1f}' if i % 3 == 0 else None
    ax.plot(x, y[i], color=colors[i], label=label)

# Add axis labels
ax.set_xlabel('Value')
ax.set_ylabel('Probability Density')
ax.set_title('Normal Inverse Gaussian Distributions')

# Add grid for readability
ax.grid(True, alpha=0.3)

plt.legend(title=r'$\beta$')
plt.tight_layout()
plt.savefig('nig_distributions.pdf', bbox_inches='tight')
plt.show()

# for gamma fixed and increasing beta, the distributions
# get super close to dirac delta measures at 0, but one on the positive part
# and one of the negative part; this might make the neural network
# take a weird turn, for very similar values of hte time series (close to 0)
# might have to output very different values of beta (- large, + large)
# can use sampling distribution != prior, but then we only get the posterior distribution up to a constant
# i don t know if this will introduce artefacts
# with prior = sampling distribution, could set it to U(-5,5) or Beta(1.5,1.5) scaled to (-5,5)
# the issue is that we will have to do a coverage check, and am thinking that MALA might go off bounds
# bc we ll have to do this automatically, for many  true samples which might be in the corner actually
# mght be better to take a prior that's not uniform in this case, or just use a transformation e..g tanh? Both?
# what do you advise?
# alternatively, could rescale beta from a bounded interval to (-infiniy, infinity)
