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

from netcal.presentation import ReliabilityDiagram
from src.model.Extended_model_nn import ExtendedModel
from src.utils.classifier_utils import get_projection_function
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.get_model import get_model
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from flax.training import train_state
from jax.random import PRNGKey
from functools import partial
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
import optax
import wandb
import yaml
import jax
import os
import netcal
if True:
    from path_setup import setup_sys_path
    setup_sys_path()

################################################
###########################################
###########################################


# training: BCE loss
# validation: S, B
# utils, after training: produce calibration plots, with a table of calibration metrics
# do MALA


# tre:
# if using summary_statistics, chop the theta
# if not using summary_statisics and using the full trawl instead, demean and
# standardize the time series on top of chopping the theta


# classifier_config_file_path = 'config_files/classifier\\classifier_config1.yaml'

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
            theta_val = jnp.concatenate(
                [theta_acf_val, theta_marginal_jax_val], axis=1)
            val_data.append((trawl_val, theta_val))

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

            to_double_check = True
            assert not to_double_check

            ######  EXTENDED MODEL HERE ########
            # CHECK KEYS ARE UPDATED
            model = ExtendedModel(base_model=model,  trawl_process_type=trawl_config['trawl_process_type'],
                                  tre_type=tre_type, use_summary_statistics=use_summary_statistics)

            # Initialize parameters
            # don't use val_key afterwards
            params = model.init(val_key[0], trawl_val, theta_val)

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
            if Y.ndim > 1:
                Y = Y.squeeze(-1)
            if pred_Y.ndim > 1:
                pred_Y = pred_Y.squeeze(-1)

            bce_loss = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(
                logits=pred_Y, labels=Y))

            # half of them are 0s, half of them are 1, so we have to x2
            S = 2 * jnp.mean(pred_Y * Y)
            classifier_output = jax.nn.sigmoid(pred_Y)
            B = 2 * jnp.mean(classifier_output)
            accuracy = jnp.mean(pred_Y == Y)

            return bce_loss, (S, B, accuracy, classifier_output)

        compute_loss_and_grad = jax.jit(jax.value_and_grad(
            compute_loss, has_aux=True), static_argnames=('train',))

        ################### helper for validations ############################

        @jax.jit
        def process_sample(params, trawl_val, theta_val):
            """JIT-compiled function to process a single validation sample."""
            batch_size = theta_val.shape[0]

            # Shuffle
            trawl_val = jnp.vstack([trawl_val, theta_val])  # normal, normal
            theta_val = jnp.vstack(
                [theta_val, jnp.roll(theta_val, -1)])  # normal, shuffled
            Y_val = jnp.vstack(
                [jnp.ones([batch_size, 1]), jnp.zeros([batch_size, 1])])  # 1, then 0

            # Compute loss, S, B, accuracy, classifier output
            bce_loss, (S, B, accuracy, classifier_output) = compute_loss(
                params, trawl_val, theta_val, Y_val, jax.random.PRNGKey(
                    0), False
            )

            # Return values for accumulation
            return jnp.array([bce_loss, bce_loss**2, S, S**2, B, B**2, accuracy, accuracy**2]), classifier_output

        def compute_validation_loss(params, val_trawls, val_thetas):
            num_samples = val_trawls.shape[0]

            # Initialize accumulators
            total = jnp.zeros(8)

            # Store classifier outputs dynamically
            all_classifier_outputs = []

            for i in range(num_samples):
                theta_val = val_thetas[i]
                trawl_val = val_trawls[i]

                # **Call JIT-compiled function**
                sample_stats, classifier_output = process_sample(
                    params, trawl_val, theta_val)

                # Accumulate statistics
                total += sample_stats

                # Store classifier outputs dynamically
                all_classifier_outputs.append(classifier_output)

            # Convert classifier outputs to JAX array
            all_classifier_outputs = jnp.concatenate(
                all_classifier_outputs, axis=0)

            # Compute means & standard deviations efficiently
            means = total[::2] / num_samples
            variances = (total[1::2] / num_samples) - (means**2)
            stds = jnp.sqrt(jnp.maximum(variances, 0.0))

            # Unpack values
            mean_loss, mean_S, mean_B, mean_accuracy = means
            std_loss, std_S, std_B, std_accuracy = stds

            return mean_loss, std_loss, mean_S, std_S, mean_B, std_B, mean_accuracy, std_accuracy, all_classifier_outputs

            #######################################################################
        #                         Training loop                               #
        #######################################################################

        for iteration in range(classifier_config["train_config"]["n_iterations"]):

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
            Y = jnp.concatenate([jnp.ones(batch_size), jnp.zeros(batch_size)])

            dropout_key, dropout_subkey_to_use = jax.random.split(dropout_key)
            (bce_loss, (S, B, accuracy, _)), grads = compute_loss_and_grad(
                state.params, trawl, theta, Y, dropout_subkey_to_use, True)

            # Update model parameters
            state = state.apply_gradients(grads=grads)
            params = state.params

            metrics = {
                'bce_loss': bce_loss.item()
            }
            ###################################################################
            #               Validation  inside the training loop              #
            ###################################################################
            # Compute validation loss periodically
            if iteration % val_freq == 0:

                val_bce, val_std_bce, val_S, val_std_S, val_B, val_std_B, val_acc, val_std_acc, all_classifier_outputs = compute_validation_loss(
                    params, val_trawls, val_thetas)

                # metrics.update({
                #    "val_metrics/val_bce_loss": val_loss.item(),
                #    "val_metrics/val_bce_upper": val_loss.item() + 1.96 * val_loss_std.item() / val_trawls.shape[0]**0.5,
                #    "val_metrics/val_bce_lower": val_loss.item() - 1.96 * val_loss_std.item() / val_trawls.shape[0]**0.5,
                # })

                metrics.update({
                    "val_bce": val_bce.item(),
                    "val_S": val_S.item(),
                    "val_B": val_B.item()
                })

                # Save just the parameters instead of full state
                params_filename = os.path.join(
                    experiment_dir, f"params_iter_{iteration}.pkl")
                with open(params_filename, 'wb') as f:
                    pickle.dump(state.params, f)

                # Keep track of best model
                if val_bce < best_val_loss:
                    best_val_loss = val_bce
                    best_iteration = iteration

                ################## diagnosing classifiers #################
                Y_calibration = jnp.hstack(
                    [jnp.ones([batch_size]), jnp.zeros([batch_size])])
                Y_calibration = np.concatenate(
                    [Y_calibration]*len(val_trawls))

                all_classifier_outputs = np.array(all_classifier_outputs)

                # uncalibrated reliability diagrams
                diagram_eq = ReliabilityDiagram(
                    20, equal_intervals=False)
                diagram_eq = diagram_eq.plot(
                    all_classifier_outputs, Y_calibration)

                diagram_un = ReliabilityDiagram(
                    20, equal_intervals=True)
                diagram_un = diagram_un.plot(
                    all_classifier_outputs, Y_calibration)

                hist_beta, ax = plt.subplots()
                ax.hist(
                    all_classifier_outputs[Y_calibration == 1], label='Y=1', alpha=0.5, density=True)
                ax.hist(
                    all_classifier_outputs[Y_calibration == 0], label='Y=0', alpha=0.5, density=True)
                ax.set_title(
                    r'Histogram of $c(\mathbf{x},\mathbf{\theta})$ classifier')
                ax.legend(loc='upper center')

                wandb.log({f"Diagram eq": wandb.Image(diagram_eq)})
                wandb.log({f"Diagram uneq": wandb.Image(diagram_un)})
                wandb.log({f"Histogram": wandb.Image(hist_beta)})

            wandb.log(metrics)

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


if __name__ == "__main__":
    import glob
    # Loop over configs
    for config_file_path in glob.glob("config_files/classifier/*.yaml"):
        train_classifier(config_file_path)
