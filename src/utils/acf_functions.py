# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:11:48 2024

@author: dleon
"""
import jax
import jax.numpy as jnp


@jax.jit
def corr_exponential_envelope(h, params):
    u = params[0]
    return jnp.exp(-u * h)


@jax.jit
def corr_gamma_envelope(h, params):
    H, delta = params
    return (1+h/delta)**(-H)


@jax.jit
def corr_sup_ig_envelope(h, params):
    gamma, eta = params
    return jnp.exp(eta * (1-jnp.sqrt(2*h/gamma**2+1)))

    # previous implementation had gamma, delta
    # where gamma is the same and eta = delta * gamma, delta = eta / gamma
    # gamma, delta = params
    # return jnp.exp(delta * gamma * (1-jnp.sqrt(2*h/gamma**2+1)))


def get_acf(acf_type):

    if acf_type == 'exponential':

        return corr_exponential_envelope

    elif acf_type == 'gamma':

        return corr_gamma_envelope

    elif acf_type == 'sup_IG':

        return corr_sup_ig_envelope

    else:

        raise ValueError(f'acf_type {acf_type} not implemented yet')
