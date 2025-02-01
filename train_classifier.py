# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:00:29 2025

@author: dleon
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:41:54 2024

@author: dleon
"""


from src.utils.classifier_utils import get_projection_function
from src.utils.trawl_training_utils import loss_functions_wrapper
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.summary_statistics_plotting import plot_acfs, plot_marginals
from src.utils.acf_functions import get_acf
from src.utils.get_model import get_model
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
from statsmodels.tsa.stattools import acf as compute_empirical_acf
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

#############################


# training: BCE loss
# validation: S, B
# utils, after training: produce calibration plots, with a table of calibration metrics
# do MALA


# tre:
# if using summary_statistics, chop the theta
# if not using summary_statisics and using the full trawl instead, demean and
# standardize the time series on top of chopping the theta


# classifier_config_file_path = 'config_files/classifier\\classifier_config.yaml'

def train_classifier(classifier_config_file_path):

    try:

        # Load config file
        with open(classifier_config_file_path, 'r') as f:
            classifier_config = yaml.safe_load(f)

        ###########################################################################
        # Get params and hyperparams
        tre_config = classifier_config['tre_config']
        use_tre = tre_config['use_tre']
        tre_type = tre_config['tre_type']
        use_summary_statistics = tre_config['use_summary_statistics']
        replace_acf = tre_config['replace_full_trawl_with_acf']
        nlags = tre_config['nlags']

        if use_summary_statistics:
            project_trawl = get_projection_function()

        #################          Initialize wandb           #################
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        project_name = "SBI_trawls_classifier_" + \
            ('tre_' + tre_type if use_tre else 'nre')
        run_name = f"{timestamp}"

        group_name = (
            'with_summary_statistics' if use_summary_statistics else
            'with_full_trawl'
        )

        wandb.init(project=project_name, group=group_name,
                   name=run_name, config=classifier_config)

        #######################################################################
        #                      Generate validation data                       #
        #######################################################################
        # Get params and hyperparams for the data generating process
        trawl_config = classifier_config['trawl_config']
        batch_size = trawl_config['batch_size']

        # Get data generators
        theta_acf_simulator, theta_marginal_simulator, trawl_simulator = get_theta_and_trawl_generator(
            classifier_config)

        # Generate validation data
        val_batches = classifier_config["val_config"]["val_n_batches"]
        val_freq = classifier_config["val_config"]["val_freq"]
        val_data = []

        # Generate fixed validation set
        # Different seed for validation
        val_key = jax.random.split(
            PRNGKey(classifier_config['prng_key'] + 10), batch_size)
        for _ in range(val_batches):
            theta_acf_val, val_key = theta_acf_simulator(val_key)
            theta_marginal_jax_val, theta_marginal_tf_val, val_key = theta_marginal_simulator(
                val_key)
            trawl_val, val_key = trawl_simulator(
                theta_acf_val, theta_marginal_tf_val, val_key)

            ########################################
            if use_summary_statistics:
                trawl_val = project_trawl(trawl_val)

            elif (not use_summary_statistics) and replace_acf and use_tre and tre_type == 'acf':

                trawl_val = jnp.array([compute_empirical_acf(np.array(trawl_), nlags=nlags)[1:]
                                       for trawl_ in trawl_val])

                ########################################

            val_data.append((trawl_val, jnp.concatenate(
                [theta_acf_val, theta_marginal_jax_val], axis=1)))

        # Convert validation data to JAX arrays
        # Saves it in the format [#batches, batch_size, vector_dimension]
        # It's a bit weird, might change it in the future
        # then need to also change the validaton loss function
        val_trawls = jnp.stack([x[0] for x in val_data])
        val_thetas = jnp.stack([x[1] for x in val_data])

        print(f'{val_batches} batches simulated for the validation dataset.')
        #######################################################################
        # Create directory for validation data and model checkpoints
        base_checkpoint_dir = os.path.join("models", 'classifier')
        checkpoint_subdir = 'summary_statistics' if use_summary_statistics else 'full_trawl'

        if use_tre:
            # TRE
            checkpoint_subdir = 'TRE_' + checkpoint_subdir
            checkpoint_subdir = os.path.join(checkpoint_subdir, tre_type)

        else:
            # NRE
            checkpoint_subdir = 'NRE_' + checkpoint_subdir

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

        #######################################################################
        #                        Get model                                    #
        #######################################################################
        # Create model and initialize parameters
        model, params, key = get_model(classifier_config)
        # for simulating data during training
        key = jax.random.split(
            PRNGKey(classifier_config['prng_key']+352), batch_size)
        dropout_key = jax.random.PRNGKey(
            classifier_config['prng_key'] + 22454)  # for dropout

        # load extended model
        if (not use_summary_statistics) and use_tre:

            ######  EXTENDED MODEL HERE ########
            # CHECK KEYS ARE UPDATED

            # Initialize optimizer
        lr = classifier_config["optimizer"]["lr"]
        total_steps = classifier_config["train_config"]["n_iterations"]
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

        if classifier_config['optimizer']['name'] == 'adam':
            if 'weight_decay' in classifier_config['optimizer']:
                # AdamW = Adam with weight decay
                optimizer = optax.adamw(
                    learning_rate=schedule_fn,
                    weight_decay=classifier_config['optimizer']['weight_decay']
                )
            else:
                # Regular Adam if no weight_decay specified
                optimizer = optax.adam(learning_rate=schedule_fn)

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
        # Initialize best validation loss tracking
        best_val_loss = float('inf')
        best_iteration = -1
        best_model_path = os.path.join(experiment_dir, "best_model")

        #######################################################################
        #                      Loss functions                                 #
        #######################################################################

        @partial(jax.jit, static_argnames=('train'))
        def compute_loss(params, trawl, theta, Y, dropout_rng, train):
            """Base loss function without gradients."""
            pred_Y = state.apply_fn(
                params,
                trawl,
                theta,
                train=train,
                rngs={'dropout': dropout_rng}
            )
            return optax.losses.sigmoid_binary_cross_entropy(logits=pred_Y, labels=Y)

        compute_loss_and_grad = jax.jit(jax.value_and_grad(
            compute_loss), static_argnames=('train',))

        @jax.jit
        def compute_validation_loss(params, val_trawls, val_thetas):

            ADD S AND B METRICS

            def body_fun(i, acc):
                theta_val = jax.lax.dynamic_slice_in_dim(val_thetas, i, 1)[0]
                trawl_val = jax.lax.dynamic_slice_in_dim(val_trawls, i, 1)[0]
                loss = compute_loss(params, trawl_val,
                                    theta_val, jax.random.PRNGKey(0), False)
                return acc + jnp.array([loss, loss**2])

            # Run the loop with just the accumulator
            total = jax.lax.fori_loop(
                0, val_trawls.shape[0], body_fun, jnp.zeros(2))

            n = val_trawls.shape[0]
            mean = total[0] / n
            variance = (total[1] / n) - (mean**2)
            std = jnp.sqrt(jnp.maximum(variance, 0.0))

            return mean, std

        #######################################################################
        #                         Training loop                               #
        #######################################################################

        for iteration in range(classifier_config["train_config"]["n_iterations"]):

            dropout_key, dropout_subkey_to_use = jax.random.split(dropout_key)

            # Generate data and shuffle
            # data A
            theta_acf_a, key = theta_acf_simulator(key)
            theta_marginal_jax_a, theta_marginal_tf_a, key = theta_marginal_simulator(
                key)
            theta_a = jnp.concatenate(
                [theta_acf_a, theta_marginal_jax_a], axis=1)

            trawl_a, key = trawl_simulator(
                theta_acf_a, theta_marginal_tf_a, key)

            # data B
            theta_acf_b, key = theta_acf_simulator(key)
            theta_marginal_jax_b, theta_marginal_tf_b, key = theta_marginal_simulator(
                key)
            # trawl_b, key = trawl_simulator(theta_acf_b, theta_marginal_tf_b, key)
            theta_b = jnp.concatenate(
                [theta_acf_b, theta_marginal_jax_b], axis=1)

            if use_summary_statistics:
                trawl_a = project_trawl(trawl_a)

            elif (not use_summary_statistics) and replace_acf and use_tre and tre_type == 'acf':

                trawl_a = jnp.array([compute_empirical_acf(np.array(trawl_), nlags=nlags)[1:]
                                     for trawl_ in trawl_a])

            theta = jnp.vstack([theta_a, theta_b])
            trawl = jnp.vstack([trawl_a, trawl_a])
            Y = jnp.vstack([jnp.ones(batch_size), jnp.zeros(batch_size)])

            loss, grads = compute_loss_and_grad(
                state.params, trawl, theta, Y, dropout_subkey_to_use, True)

            # Update model parameters
            state = state.apply_gradients(grads=grads)
            params = state.params

            ###################################################################
            #               Validation  inside the training loop              #
            ###################################################################
            # Compute validation loss periodically
            if iteration % val_freq == 0:

                TO ADD

            # Save best model info
            best_model_info_path = os.path.join(
                val_data_dir, "best_model_info.txt")
            with open(best_model_info_path, 'w') as f:
                f.write(f"Best model iteration: {best_iteration}\n")
                f.write(f"Best validation loss: {best_val_loss:.6f}\n")

            config_save_path = os.path.join(val_data_dir, "config.yaml")
            with open(config_save_path, 'w') as f:
                yaml.dump(classifier_config, f)

    finally:
        # At the very end of the function
        wandb.finish()
