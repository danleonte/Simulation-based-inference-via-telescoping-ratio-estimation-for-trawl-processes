# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:11:16 2024

@author: dleon
"""

import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions
norminvgauss = tfd.NormalInverseGaussian


@jax.jit
def compute_matrix_from_first_column_4(s_i1):
    nr_trawls = len(s_i1)

    #  a[0] -a[1] ,a[1] -a[2], ... , a[k-2] - a[k-1] , 0
    differences = jnp.append(jnp.diff(s_i1[::-1])[::-1], 0)

    left_column = jnp.array(s_i1)[:, jnp.newaxis]
    right_column = jnp.zeros((nr_trawls, 1))
    # we reconstruct the elements on the secondary diagonal at the end

    middle_matrix = jnp.tile(differences[:, jnp.newaxis], (1, nr_trawls - 2))
    whole_matrix = jnp.concatenate(
        [left_column, middle_matrix, right_column], axis=1)
    whole_matrix_reversed = jnp.triu(jnp.fliplr(whole_matrix), k=0)

    diag_elements = jnp.diag_indices_from(whole_matrix_reversed)
    whole_matrix_reversed = whole_matrix_reversed.at[diag_elements].set(s_i1)

    slice_areas_matrix = jnp.fliplr(whole_matrix_reversed)
    return slice_areas_matrix


# first argument should be static when (vmapped and then) jitted
def slice_sample_sup_ig_nig_trawl(nr_trawls, tau, envelope_params,
                                  distr_params, key):
    """
    Args:

        nr_trawls: sequence length               int
        tau: should be 1.0,                      int
        envelope_params: (gamma,delta)           jnp.array
        distr_params: (mu, scale, alpha, beta)   jnp.array
        key: jax randomness seed                 PRNGkey

    Return one path of the trawl process.
    """

    acf_gamma, acf_delta = envelope_params
    h = jnp.arange(0.0, nr_trawls, 1.0) * tau
    const_h = (2 * h / acf_gamma**2 + 1) ** 0.5
    top = 1 - jnp.exp(acf_gamma * acf_delta * (1 - const_h))

    # top = jnp.exp(lambda_ * jnp.arange(-nr_trawls+1, 1., 1.)
    #              * tau)  # /lambda_: noramlize the area to 1
    s_i1 = jnp.append(jnp.diff(top), top[0])

    areas = compute_matrix_from_first_column_4(s_i1)

    indicator_small_areas = areas < 5 * 10 ** (-8)
    # some not so small number
    areas = jnp.where(indicator_small_areas, 1.0, areas)

    # take out the zeros, i.e. below the second diagonal
    areas = (jnp.fliplr(areas))[jnp.triu_indices(nr_trawls, k=0)]

    # sample rows with TF parameterization
    mu, scale, alpha, beta = distr_params  # tf params
    sampler = norminvgauss(
        loc=mu * areas, scale=scale * areas, tailweight=alpha, skewness=beta
    )

    key, subkey = jax.random.split(key)
    sampled_values_rows = sampler.sample(seed=subkey)

    # back in matrix form
    slice_values = jnp.zeros([nr_trawls, nr_trawls])
    a = jnp.fliplr(
        slice_values.at[jnp.triu_indices(nr_trawls, k=0)].set(
            sampled_values_rows)
    )
    a = jnp.where(indicator_small_areas, 0.0, a)

    a = jnp.cumsum(a[::-1], axis=0)[::-1]
    return (
        jnp.bincount(
            jnp.sum(jnp.indices([nr_trawls, nr_trawls]), axis=0).flatten(),
            a.flatten(),
            length=nr_trawls,
        ),
        key,
    )
