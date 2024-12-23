# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:34:48 2024

@author: dleon
"""

import jax
import jax.numpy as jnp


def corr_exponential_envelope(h, params):
    u = params[0]
    return jnp.exp(-u * h)


def corr_gamma_envelope(h, params):
    H, delta = params
    return (1+h/delta)**(-H)


def corr_supp_ig_envelope(h, params):
    gamma, delta = params
    return jnp.exp(delta * gamma * (1-jnp.sqrt(2*h/gamma**2+1)))
