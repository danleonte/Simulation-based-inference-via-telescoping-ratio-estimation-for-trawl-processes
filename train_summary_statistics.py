# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:40:41 2025

@author: dleon
"""


if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import os
import jax
import yaml
import wandb
import optax
import pickle
import datetime
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.random import PRNGKey
from flax.training import train_state
from src.utils.get_model import get_model
from src.utils.acf_functions import get_acf
from src.utils.summary_statistics_plotting import plot_acfs, plot_marginals
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.trawl_training_utils import loss_functions_wrapper


# config_file_path = 'config_files/summary_statistics/LSTM/marginal\\config3.yaml'


def train_and_evaluate(config_file_path):

    try:

        # Load config file
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        ###########################################################################
        # Get general params and hyperparams, check if we learn the acf or marginal
        # distribution parameters
        learn_config = config['learn_config']
        learn_acf = learn_config['learn_acf']
        learn_marginal = learn_config['learn_marginal']
        learn_both = learn_config['learn_both']

        assert learn_acf + learn_marginal == 1 and learn_both == False

        # Initialize wandb
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        config_name = os.path.basename(config_file_path).replace('.yaml', '')
        run_name = f"{timestamp}_{config_name}"
        run_name = 'acf' if learn_acf else 'marginal'

        group_name = (
            "acf" if learn_acf else
            "marginal" if learn_marginal else
            "both")
        wandb.init(project="SBI_trawls", group=group_name,
                   name=run_name, config=config)

        ###########################################################################
        # Get params and hyperparams for the data generating process
        trawl_config = config['trawl_config']
        batch_size = trawl_config['batch_size']

        # Get data generators
        theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
            config)

        # Generate validation data
        val_batches = config["val_config"]["val_n_batches"]
        val_freq = config["val_config"]["val_freq"]
        val_data = []

        # Generate fixed validation set
        # Different seed for validation
        val_key = jax.random.split(
            PRNGKey(config['prng_key'] + 10), batch_size)
        for _ in range(val_batches):
            theta_acf_val, val_key = theta_acf_simulator(val_key)
            theta_marginal_jax_val, theta_marginal_tf_val, val_key = theta_marginal_simulator(
                val_key)
            trawl_val, val_key = trawl_simulator(
                theta_acf_val, theta_marginal_tf_val, val_key)

            if learn_acf:
                val_data.append((trawl_val, theta_acf_val))
            elif learn_marginal:
                val_data.append((trawl_val, theta_marginal_jax_val))

        # Convert validation data to JAX arrays
        # Saves it in the format [#batches, batch_size, vector_dimension]
        # It's a bit weird, might change it in the future
        # then need to also change the validaton loss function
        val_trawls = jnp.stack([x[0] for x in val_data])
        val_thetas = jnp.stack([x[1] for x in val_data])

        print(f'{val_batches} batches simulated for the validation dataset.')

        ###########################################################################
        # Create directory for validation data and model checkpoints
        base_checkpoint_dir = os.path.join("models", 'summary_statistics')
        checkpoint_subdir = "learn_acf" if learn_acf else "learn_marginal"
        experiment_dir = os.path.join(
            base_checkpoint_dir, checkpoint_subdir, wandb.run.name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Save validation dataset
        val_data_dir = os.path.join(experiment_dir, "validation_data")
        os.makedirs(val_data_dir, exist_ok=True)

        # Convert to numpy and save
        np.save(os.path.join(val_data_dir, "val_trawls.npy"),
                np.array(val_trawls))
        np.save(os.path.join(val_data_dir, "val_thetas.npy"),
                np.array(val_thetas))

        ###########################################################################
        # Create model and initialize parameters
        model, params, key = get_model(config)
        # for simulating data during training
        key = jax.random.split(PRNGKey(config['prng_key']+2351), batch_size)
        dropout_key = jax.random.PRNGKey(
            config['prng_key'] + 29354)  # for dropout

        # Initialize optimizer
        lr = config["optimizer"]["lr"]
        total_steps = config["train_config"]["n_iterations"]
        warmup_steps = 500
        decay_steps = total_steps - warmup_steps

        schedule_fn = optax.join_schedules([
            # Constant learning rate for warmup_steps
            optax.constant_schedule(lr),
            # Cosine decay for the remaining steps
            optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=decay_steps,
                alpha=0.05
            )
        ], boundaries=[warmup_steps])

        if config['optimizer']['name'] == 'adam':
            if 'weight_decay' in config['optimizer']:
                # AdamW = Adam with weight decay
                optimizer = optax.adamw(
                    learning_rate=schedule_fn,
                    weight_decay=config['optimizer']['weight_decay']
                )
            else:
                # Regular Adam if no weight_decay specified
                optimizer = optax.adam(learning_rate=schedule_fn)

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )

        ###########################################################################
        # Get params and hyperparams for the loss function
        loss_config = config['loss_config']
        num_KL_samples = loss_config['num_KL_samples']

        # Loss functions
        predict_theta, compute_loss, compute_loss_and_grad, \
            compute_validation_stats = loss_functions_wrapper(state, config)

        # Initialize best validation loss tracking
        best_val_loss = float('inf')
        best_iteration = -1
        best_model_path = os.path.join(experiment_dir, "best_model")
        ###########################################################################

        # Training loop
        for iteration in range(config["train_config"]["n_iterations"]):

            theta_acf, key = theta_acf_simulator(key)
            theta_marginal_jax, theta_marginal_tf, key = theta_marginal_simulator(
                key)
            trawl, key = trawl_simulator(theta_acf, theta_marginal_tf, key)

            dropout_key, dropout_subkey_to_use = jax.random.split(dropout_key)

            # Compute loss and gradients
            if learn_acf:
                loss, grads = compute_loss_and_grad(
                    params, trawl, theta_acf, dropout_subkey_to_use, True, num_KL_samples)

            elif learn_marginal:
                loss, grads = compute_loss_and_grad(
                    params, trawl, theta_marginal_jax, dropout_subkey_to_use, True, num_KL_samples)

                if iteration == 1000:
                    num_KL_samples *= 2

                if iteration == 3000:
                    num_KL_samples *= 2

                if iteration == 5500:
                    num_KL_samples *= 2

                if iteration = 9000:
                    num_KL_samples *= 2

            # Update model parameters
            state = state.apply_gradients(grads=grads)
            params = state.params

            # Logging and then validation
            loss_name = 'acf_loss' if learn_acf else 'marginal_loss'
            train_loss, val_loss = 'train_' + loss_name, 'val_' + loss_name
            metrics = {
                train_loss: loss.item()
            }

            # Compute validation loss periodically
            if iteration % val_freq == 0:

                # if learn_acf:
                #
                #    val_loss, val_loss_std = compute_validation_stats(
                #        params, val_trawls, val_thetas, num_KL_samples)
                #
                # elif learn_marginal:

                val_loss, val_loss_std, dropout_key = compute_validation_stats(
                    params, val_trawls, val_thetas, dropout_key, num_KL_samples)

                # Log metrics under the same group for better visualization
                metrics.update({
                    "val_metrics/val_loss": val_loss.item(),
                    "val_metrics/val_loss_upper": val_loss.item() + 1.96 * val_loss_std.item() / val_trawls.shape[0]**0.5,
                    "val_metrics/val_loss_lower": val_loss.item() - 1.96 * val_loss_std.item() / val_trawls.shape[0]**0.5,
                })

                # Save just the parameters instead of full state
                params_filename = os.path.join(
                    experiment_dir, f"params_iter_{iteration}.pkl")
                with open(params_filename, 'wb') as f:
                    pickle.dump(state.params, f)

                # Keep track of best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iteration = iteration

                pred_theta = predict_theta(
                    params, trawl, dropout_key, False)

                if learn_acf:

                    for i in range(5):
                        fig_ = plot_acfs(
                            trawl[i], theta_acf[i], pred_theta[i], config)
                        wandb.log({f"Acf plot {i}": wandb.Image(fig_)})

                elif learn_marginal:

                    for i in range(5):
                        fig_ = plot_marginals(
                            theta_marginal_jax[i], pred_theta[i], config)
                        wandb.log({f"Acf plot {i}": wandb.Image(fig_)})

            wandb.log(metrics)

        # Save best model info
        best_model_info_path = os.path.join(
            val_data_dir, "best_model_info.txt")
        with open(best_model_info_path, 'w') as f:
            f.write(f"Best model iteration: {best_iteration}\n")
            f.write(f"Best validation loss: {best_val_loss:.6f}\n")

        config_save_path = os.path.join(val_data_dir, "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)

    finally:
        # At the very end of the function
        wandb.finish()


if __name__ == "__main__":
    import glob
    # Loop over configs
    for config_file_path in glob.glob("config_files/summary_statistics/LSTM/marginal/*.yaml"):
        train_and_evaluate(config_file_path)
