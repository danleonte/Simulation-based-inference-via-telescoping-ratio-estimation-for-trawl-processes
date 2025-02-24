# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:25:24 2024

@author: dleon
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from src.utils.acf_functions import get_acf


acf_func = jax.jit(
    jax.vmap(get_acf('sup_IG'), in_axes=(None, 0)))


def modify_x(x, theta, tre_indicator, trawl_process_type):

    # trawl_process_type = classifier_config['trawl_config']['trawl_process_type']

    if trawl_process_type == 'sup_ig_nig_5p':

        gamma_acf, eta_acf, mu, scale, beta = jnp.transpose(theta)

        if tre_indicator == "beta":
            modified_x = (x - mu)/scale

        elif tre_indicator == "sigma":
            modified_x = (x-mu)

        elif tre_indicator in ("mu", "acf", "nre"):
            modified_x = x

        else:
            raise ValueError

        return modified_x


def chop_theta(theta, tre_type, trawl_process_type):

    if trawl_process_type == 'sup_ig_nig_5p':

        if tre_type in ("beta", "nre"):
            modified_theta = theta

        elif tre_type == "sigma":
            modified_theta = theta[:, :4]

        elif tre_type == 'mu':
            modified_theta = theta[:, :3]

        elif tre_type == 'acf':

            modified_theta = theta[:, :2]
        else:
            raise ValueError

        return modified_theta


# Approach 1: String in constructor
class ExtendedModel(nn.Module):
    base_model: nn.Module
    # String parameter in constructor ;can be one of mu, sigma, beta, acf or nre
    trawl_process_type: str
    tre_type: str
    use_summary_statistics: bool

    def setup(self):
        pass  # This is required in Flax when using only static fields

    def __call__(self, x, theta, train: bool = False):

        H = jnp.arange(1, 35)

        if not self.use_summary_statistics:
            x = modify_x(x, theta, self.tre_type, self.trawl_process_type)

        # only after modifying x
        theta = chop_theta(theta, self.tre_type, self.trawl_process_type)
        r = acf_func(H, x[:, :2]) - acf_func(H, theta)
        r = jnp.sqrt(jnp.sum(r**2, axis=1, keepdims=True))

        return self.base_model(x=r, train=train)
