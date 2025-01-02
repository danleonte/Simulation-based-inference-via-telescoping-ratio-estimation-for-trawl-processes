# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:25:24 2024

@author: dleon
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


def modify_x(x, theta, tre_indicator):

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


def chop_theta(theta, tre_indicator):

    if tre_indicator in ("beta", "nre"):
        modified_theta = theta

    elif tre_indicator == "sigma":
        modified_theta = theta[:, :4]

    elif tre_indicator == 'mu':
        modified_theta = theta[:, :3]

    elif tre_indicator == 'acf':

        modified_theta = theta[:, :2]
    else:
        raise ValueError

    return modified_theta


# Approach 1: String in constructor
class ExtendedModel_for_sup_ig_nig_trawl(nn.Module):
    base_model: nn.Module
    # String parameter in constructor ;can be one of mu, sigma, beta, acf or nre
    tre_indicator: str
    summary_statistics_indicator: bool

    def __call__(self, x, theta):

        if not self.summary_statistics:
            x = modify_x(x, theta, self.tre_indicator)

        theta = chop_theta(theta, self.tre_indicator)  # only after modifying x

        return self.base_model(x, theta)
