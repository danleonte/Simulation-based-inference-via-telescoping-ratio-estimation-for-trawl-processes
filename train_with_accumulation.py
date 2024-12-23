# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:52:50 2024

@author: dleon
"""
import yaml
import wandb
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
# from flax.training import train_state
import optax

# Training step with gradient accumulation


def train_epoch(state, batch_size, num_acc_steps):
    pass

    for i in range(num_acc_steps):
        # Initialize gradient accumulator
        grad_accumulator = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        accumulated_loss = 0

        # Accumulate gradients over num_acc_steps micro-batches


def train_and_evaluate(config_path):

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(project="SBI_trawls", config=config)

    # Initialize data generator

    # Create model

    # Initialize parameters and optimizer

    # Training loop
    for iteration in range(config["train_config"]["n_iterations"]):
        pass

    # Validation and logging
    wandb.log({
        "epoch": iteration,
    })

    wandb.finish()


if __name__ == "__main__":
    import glob
    # Loop over configs
    for config_file in glob.glob("configs/*.yaml"):
        train_and_evaluate(config_file)
