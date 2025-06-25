# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:41:54 2024

@author: dleon
"""


import optax
from flax.training import train_state
from jax.random import PRNGKey
import jax.numpy as jnp
import numpy as np
import datetime
import jax
import wandb
import os
import pickle
import yaml

if True:
    from path_setup import setup_sys_path
    setup_sys_path()


from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.get_model import get_model
from src.utils.utils import update_step, summary_stats_loss_fn


def train_classifier(config_file_path):

    # Load config
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    ###########################################################################
    # Get params and hyperparams
    tre_config = config['tre_config']
    use_tre = tre_config['use_tre']
    tre_type = tre_config['tre_type']
    use_summary_statistics = tre_config['use_summary_statistics']
    replace_acf = tre_config['replace_full_trawl_with_acf']

    # Initialize wandb
    group_name = (f"classifier_tre_{tre_type}_" if use_tre else "nre_") + (
        "summary_" if use_summary_statistics else 'full_trawl_')
    if (not use_summary_statistics and replace_acf):
        group_name += "replace_acf"

    wandb.init(project="SBI_trawls", group=group_name, config=config)

    trawl_config = config['trawl_config']
    batch_size = trawl_config['batch_size']

    ###########################################################################
    # Get data generators
    theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
        config)

    # Generate validation data
    val_batches = config["val_config"]["val_n_batches"]
    val_freq = config["val_config"]["val_freq"]
    val_data = []

    # Generate fixed validation set
    # Different seed for validation
    val_key = jax.random.split(PRNGKey(config['prng_key'] + 10), batch_size)
    for _ in range(val_batches):
        theta_acf_val, val_key = theta_acf_simulator(val_key)
        theta_marginal_jax_val, theta_marginal_tf_val, val_key = theta_marginal_simulator(
            val_key)
        trawl_val, val_key = trawl_simulator(
            theta_acf_val, theta_marginal_tf_val, val_key)

        # if using summary statistics, project the trawl
        if use_summary_statistics:
            trawl_val = project(trawl_val)

        theta_val = jnp.concatenate(
            [theta_acf_val, theta_marginal_jax_val], axis=1)

        val_data.append((trawl_val, theta_val))

    # Convert validation data to JAX arrays
    val_trawls = jnp.stack([x[0] for x in val_data])
    val_thetas = jnp.stack([x[1] for x in val_data])

    print(f'{val_batches} batches simulated for the validation dataset.')

    ###########################################################################
    # Create directory for validation data and model checkpoints
    base_checkpoint_dir = os.path.join("models", 'classifier')
    experiment_dir = os.path.join(base_checkpoint_dir,  wandb.run.name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save validation dataset
    val_data_dir = os.path.join(experiment_dir, "validation_data")
    os.makedirs(val_data_dir, exist_ok=True)

    # Convert to numpy and save
    np.save(os.path.join(val_data_dir, "val_trawls.npy"), np.array(val_trawls))
    np.save(os.path.join(val_data_dir, "val_thetas.npy"), np.array(val_thetas))

    # If using summary statistics, replace val_trawls with these
    # Elif full trawl and TRE and replace_acf, replace val_trawls with the acf

    ###########################################################################
    # Create model and initialize parameters
    base_model = get_model(config)
    model = 0
    # initialize model here
    key = jax.random.split(PRNGKey(config['prng_key']), batch_size)

    # Initialize optimizer
    lr = config["optimizer"]["lr"]

    # Create learning rate schedule
    schedule_fn = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales={150: lr/10}  # At step 100, multiply lr by 0.1
    )

    if config['optimizer']['name'] == 'adam':
        optimizer = optax.adam(schedule_fn)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        # jax.experimental.optimizers.adam(config["learning_rate"]),
        params=params,
        tx=optimizer
    )
    ###########################################################################
    # Loss functions

    @jax.jit
    def compute_loss(params, trawl, theta, Y):
        """Base loss function without gradients."""
        pred_Y = model.apply(params, trawl, theta)
        return optax.losses.sigmoid_binary_cross_entropy(logits=pred_Y, labels=Y)

    # get grads for training
    compute_loss_and_grad = jax.jit(jax.value_and_grad(compute_loss))

    @jax.jit
    def compute_validation_loss(params, val_trawls, val_thetas):
        pass

    # Initialize best validation loss tracking
    best_val_loss = float('inf')
    best_iteration = -1
    best_model_path = os.path.join(experiment_dir, "best_model")
    ###########################################################################

    # Training loop
    for iteration in range(config["train_config"]["n_iterations"]):

        # Generate data and shuffle
        # data A
        theta_acf_a, key = theta_acf_simulator(key)
        theta_marginal_jax_a, theta_marginal_tf_a, key = theta_marginal_simulator(
            key)
        theta_a = jnp.concatenate([theta_acf_a, theta_marginal_jax_a], axis=1)

        trawl_a, key = trawl_simulator(theta_acf_a, theta_marginal_tf_a, key)
        if use_summary_statistics:
            trawl_a = project(trawl_a)

        # data B
        theta_acf_b, key = theta_acf_simulator(key)
        theta_marginal_jax_b, theta_marginal_tf_b, key = theta_marginal_simulator(
            key)
        # trawl_b, key = trawl_simulator(theta_acf_b, theta_marginal_tf_b, key)
        theta_b = jnp.concatenate([theta_acf_b, theta_marginal_jax_b], axis=1)

        # shuffle
        # jnp.vstack([theta_a, theta_a, theta_b, theta_b])
        theta = jnp.vstack([theta_a, theta_b])
        # jnp.vstack([trawl_a, trawl_b, trawl_a, trawl_b])
        trawl = jnp.vstack([trawl_a, trawl_a])
        Y = jnp.vstack([jnp.ones(batch_size), jnp.zeros(batch_size)])
        # jnp.vstack([jnp.ones(batch_size), jnp.zeros(
        # batch_size), jnp.zeros(batch_size), jnp.ones(batch_size)])

        loss, grads = compute_loss_and_grad(params, trawl, theta, Y)

        # Update model parameters
        state = update_step(state, grads)

        # Validation and log in

        metrics = {
            'bce_loss': loss.item()
        }

        wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    import glob
    # Loop over configs
    for config_file_path in glob.glob("config_files/classifier/*.yaml"):
        train_classifier(config_file_path)
