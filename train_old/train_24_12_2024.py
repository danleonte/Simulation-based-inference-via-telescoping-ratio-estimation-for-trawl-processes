# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:43:15 2024

@author: dleon
"""


import yaml
import wandb
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from flax.training import train_state
import optax

if True:
    from path_setup import setup_sys_path
    setup_sys_path()

from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.get_model import get_model
from src.utils.utils import update_step, summary_stats_loss_fn


def train_and_evaluate(config_file_path):

    # Load config
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(project="SBI_trawls", config=config)

    # Get params and hyperparams
    trawl_config = config['trawl_config']
    batch_size = trawl_config['batch_size']

    # Get data generators
    theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
        config)

    # Create model and initialize parameters
    model, params, carry, key = get_model(config)
    key = jax.random.split(PRNGKey(config['prng_key']), batch_size)

    # Initialize optimizer
    lr = config["optimizer"]["lr"]
    if config['optimizer']['name'] == 'adam':
        optimizer = optax.adam(lr)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        # jax.experimental.optimizers.adam(config["learning_rate"]),
        params=params,
        tx=optimizer
    )

    # add if statements here and compute_loss_and_grad_both_thetas
    # all with the same name, and different loss functions
    # if statements inside this functioon maybe?

    ########################  Loss  functions  ################################

    @jax.jit
    def compute_loss_and_grad(params, trawl, carry, theta_acf, theta_marginal_jax):
        """
        Compute loss and gradients.
        """
        def loss_fn_wrapped(params, carry):
            pred_theta = model.apply(params, trawl, carry)
            total_loss, (acf_loss, marginal_loss) = summary_stats_loss_fn(
                pred_theta, theta_acf, theta_marginal_jax)
            return total_loss, (acf_loss, marginal_loss)

        (total_loss, (acf_loss, marginal_loss)), grads = jax.value_and_grad(
            lambda params: loss_fn_wrapped(params, carry), has_aux=True
        )(params)
        return total_loss, acf_loss, marginal_loss, grads

    ###########################################################################

    # Training loop
    for iteration in range(config["train_config"]["n_iterations"]):

        theta_acf, key = theta_acf_simulator(key)
        theta_marginal_jax, theta_marginal_tf, key = theta_marginal_simulator(
            key)
        trawl, key = trawl_simulator(theta_acf, theta_marginal_tf, key)

        # ADD IF STATEMENTS HERE

        # Compute loss and gradients
        total_loss, acf_loss, marginal_loss, grads = compute_loss_and_grad(
            state.params, trawl, carry, theta_acf, theta_marginal_jax)

        # Update model parameters
        state = update_step(state, grads)

        # Validation and logging
        wandb.log({
            "acf_loss": acf_loss.item(),
            "marginal_loss": marginal_loss.item(),
            "total_loss": total_loss.item(),
        })

    wandb.finish()


if __name__ == "__main__":
    import glob
    # Loop over configs
    for config_file_path in glob.glob("config_files/*.yaml"):
        train_and_evaluate(config_file_path)
