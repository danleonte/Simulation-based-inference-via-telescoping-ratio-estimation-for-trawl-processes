# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:14:40 2024

@author: dleon
"""
import jax
import jax.numpy as jnp
from functools import partial


# Corrected update_step function
@jax.jit
def update_step(state, grads):
    """
    Update model parameters using gradients.
    """
    new_state = state.apply_gradients(grads=grads)  # Fix: Provide only grads
    return new_state

# @jax.jit
# def update_step(state, grads):
#    """
#    Update model parameters using gradients.
#    """
#    updates, new_opt_state = state.tx.update(grads, state.opt_state)
#    new_state = state.apply_gradients(
#        grads=updates, opt_state=new_opt_state)
#    return new_state


def summary_stats_loss_fn(pred_theta, theta_acf, theta_marginal_jax):
    """
    Compute total loss, ACF loss, and marginal loss.

    Args:
        trawl: Input to the model.
        theta_acf: Ground truth ACF parameters.
        theta_marginal_jax: Ground truth marginal parameters.

    Returns:
        total_loss: Combined loss.
        (acf_loss, marginal_loss): Individual losses as auxiliary output.
    """
    # Split prediction into acf and marginal parts
    pred_theta_acf = pred_theta[:, :theta_acf.shape[-1]]
    pred_theta_marginal = pred_theta[:, theta_acf.shape[-1]:]

    # Calculate losses
    acf_loss = jnp.mean((pred_theta_acf - theta_acf) ** 2)
    marginal_loss = jnp.mean((pred_theta_marginal - theta_marginal_jax) ** 2)

    # Total loss
    total_loss = acf_loss + marginal_loss
    return total_loss, (acf_loss, marginal_loss)
