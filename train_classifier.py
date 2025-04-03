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

# Set the backend before importing pyplot


# matplotlib.use('Agg')
import jax.numpy as jnp
from functools import partial
from jax.random import PRNGKey
from flax.training import train_state
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.get_model import get_model
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function, tre_shuffle
from src.model.Extended_model_nn import ExtendedModel
from netcal.presentation import ReliabilityDiagram
import numpy as np
import datetime
import time
import pickle
import optax
import wandb
import yaml
import jax
import os
import netcal
import matplotlib
if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt


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


# classifier_config_file_path = 'config_files/classifier\\TRE_summary_statistics/beta/classifier_config1.yaml'

def check_if_run_stopped():
    """Check if the current wandb run was manually stopped from the UI."""
    try:
        api = wandb.Api()
        run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
        return run.state in ["finished", "failed", "crashed", "killed"]
    except Exception as e:
        print(f"Warning: Failed to check run status. Error: {e}")
        return False  # Assume it's running if there's an issue


def try_to_close_wandb():
    if wandb.run is not None:
        try:
            wandb.finish()
        except:
            pass
        # Small delay to ensure wandb is fully cleaned up
        time.sleep(5)


def train_classifier(classifier_config):

    try:

        # Check what we're doing: TRE or NRE, with full trawls or summary stats
        tre_config = classifier_config['tre_config']
        use_tre = tre_config['use_tre']
        tre_type = tre_config['tre_type']
        use_summary_statistics = tre_config['use_summary_statistics']
        replace_acf = tre_config['replace_full_trawl_with_acf']
        nlags = tre_config['nlags'],
        summary_stats_type = ''

        if use_summary_statistics:
            project_trawl = get_projection_function(
                tre_config['summary_stats_type'])
            summary_stats_type = '_' + tre_config['summary_stats_type']

        #################          Initialize wandb           #################
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        project_name = "classifier_" + \
            ('tre_' + tre_type if use_tre else 'nre') + \
            (
                '_with_summary_statistics' + summary_stats_type if use_summary_statistics else
                '_with_full_trawl'
            )
        run_name = f"{timestamp}"

        # group_name = (
        #    'with_summary_statistics' if use_summary_statistics else
        #    'with_full_trawl'
        # )

        wandb.init(project=project_name,  # group=group_name,
                   name=run_name, config=classifier_config)

        # Create directory for validation data and model checkpoints

        base_checkpoint_dir = os.path.join("models", 'classifier')
        checkpoint_subdir = 'summary_statistics' + \
            summary_stats_type if use_summary_statistics else 'full_trawl'

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

        val_trawls_path = os.path.join(
            base_checkpoint_dir, 'val_trawls_class.npy')
        val_thetas_path = os.path.join(
            base_checkpoint_dir, 'val_thetas_class.npy')

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
        val_key = jax.random.split(
            PRNGKey(classifier_config['prng_key'] + 10), batch_size)

        if not (os.path.isfile(val_trawls_path) and os.path.isfile(val_thetas_path)):

            # Generate fixed validation set
            val_data = []

            for _ in range(val_batches):
                theta_acf_val, val_key = theta_acf_simulator(val_key)
                theta_marginal_jax_val, theta_marginal_tf_val, val_key = theta_marginal_simulator(
                    val_key)
                trawl_val, val_key = trawl_simulator(
                    theta_acf_val, theta_marginal_tf_val, val_key)

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

            # Convert to numpy and save
            np.save(val_trawls_path, np.array(val_trawls))
            np.save(val_thetas_path, np.array(val_thetas))

        #######################################################################
        val_trawls = []
        _val_trawls = jnp.load(val_trawls_path)
        val_thetas = jnp.load(val_thetas_path)

        for _ in range(val_batches):

            trawl_val = _val_trawls[_]

            if use_summary_statistics:
                trawl_val = project_trawl(trawl_val)

            elif (not use_summary_statistics) and replace_acf and use_tre and tre_type == 'acf':

                trawl_val = jnp.array([compute_empirical_acf(np.array(trawl_), nlags=nlags)[1:]
                                       for trawl_ in trawl_val])

            val_trawls.append(trawl_val)

        val_trawls = jnp.array(val_trawls)
        del _val_trawls
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

        if use_tre:

            # CAN T LEARN THE ACF WITHOUT THE SUMMARY STATISTICS
            # SO WE RE NOT  CONSIDERING THAT CASE HERE
            # DO NOT INITIALIZE MODEL
            del params
            model = get_model(classifier_config, False)
            assert tre_type in ('beta', 'mu', 'sigma', 'acf')
            # if tre_type == 'acf' and (not use_summary_statistics):
            #    assert replace_acf

            ######  EXTENDED MODEL HERE ########
            # CHECK KEYS ARE UPDATED
            model = ExtendedModel(base_model=model,  trawl_process_type=trawl_config['trawl_process_type'],
                                  tre_type=tre_type, use_summary_statistics=use_summary_statistics)

            # Initialize parameters
            # don't use val_key afterwards
            params = model.init(val_key[0], val_trawls[0], val_thetas[0])

            # Initialize optimizer
        lr = classifier_config["optimizer"]["lr"]
        if 'alpha' in classifier_config["optimizer"].keys():
            alpha = classifier_config["optimizer"]["alpha"]
        else:
            alpha = 0.025

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
                alpha=alpha
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
        os.makedirs(best_model_path, exist_ok=True)

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
            accuracy = jnp.mean(
                (classifier_output > 0.5).astype(jnp.float32) == Y)

            return bce_loss, (S, B, accuracy, classifier_output)

        compute_loss_and_grad = jax.jit(jax.value_and_grad(
            compute_loss, has_aux=True), static_argnames=('train',))

        ################### helper for validations ############################

        @jax.jit
        def process_sample(params, trawl_val, theta_val):
            """JIT-compiled function to process a single validation sample."""
            batch_size = theta_val.shape[0]

            # Shuffle
            # trawl_val = jnp.vstack([trawl_val, trawl_val])  # normal, normal
            # theta_val = jnp.vstack(
            #    [theta_val, jnp.roll(theta_val, -1,axis=0)])  # normal, shuffled
            # Y_val = jnp.vstack(
            #    [jnp.ones([batch_size, 1]), jnp.zeros([batch_size, 1])])  # 1, then 0

            trawl_val, theta_val, Y_val = tre_shuffle(
                trawl_val, theta_val, jnp.roll(theta_val, -1, axis=0), classifier_config)

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

                sample_stats, classifier_output = process_sample(
                    params, trawl_val, theta_val)

                # Accumulate statistics
                total += sample_stats
                all_classifier_outputs.append(classifier_output)

            all_classifier_outputs = jnp.concatenate(
                all_classifier_outputs, axis=0)

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

            # Check if this run has been terminated from the wandb UI
            # if wandb.run.terminated:
            #    print(f"Run {wandb.run.name} was terminated from the wandb UI")
            #    break

            # if check_if_run_stopped():
            #    print(
            #        f"Run {wandb.run.name} was stopped from the wandb UI. Moving to next config.")
            #    # Force clean exit
            #    wandb.finish()  # Ensure wandb is closed properly
            #    return None  # Signal to main loop to continue to next config

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

            trawl, theta, Y = tre_shuffle(
                trawl_a, theta_a, theta_b, classifier_config)

            # theta = jnp.vstack([theta_a, theta_b])
            # trawl = jnp.vstack([trawl_a, trawl_a])
            # Y = jnp.concatenate([jnp.ones(batch_size), jnp.zeros(batch_size)])

            ############ annoying code to deal with W&B run stops ##############
            # if check_if_run_stopped():
            #    print(
            #        f"Run {wandb.run.name} was stopped. Skipping gradient computation.")
            #    wandb.finish()
            #    return None  # Signal to main loop to continue to next config

            ###################################################################

            try:
                dropout_key, dropout_subkey_to_use = jax.random.split(
                    dropout_key)
                (bce_loss, (S, B, accuracy, _)), grads = compute_loss_and_grad(
                    state.params, trawl, theta, Y, dropout_subkey_to_use, True)

                # Update model parameters
                state = state.apply_gradients(grads=grads)
                params = state.params

            except KeyboardInterrupt:
                # Handle keyboard interrupt during JAX computation
                print("Keyboard interrupt during computation. Exiting gracefully...")
                wandb.finish()
                return None
            except Exception as e:
                # For other exceptions, check if it's because the run was stopped
                if check_if_run_stopped():
                    print(f"Run was stopped during computation. Error: {e}")
                    wandb.finish()
                    return None
                else:
                    # If it's a legitimate error, re-raise it
                    raise

            metrics = {
                'acc': accuracy.item(),
                'bce_loss': bce_loss.item()
            }
            ###################################################################
            #               Validation  inside the training loop              #
            ###################################################################
            # Compute validation loss periodically
            if iteration > 5000 and iteration % val_freq == 0:
                # Check if run stopped before starting validation
                # if check_if_run_stopped():
                #    print(
                #        f"Run {wandb.run.name} was stopped before validation. Exiting.")
                #    wandb.finish()
                #    return None

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
                    "val_B": val_B.item(),
                    "val_acc": val_acc.item()
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

                if iteration > 5000 and (iteration % (2 * val_freq) == 0):
                    # Check if run stopped before plotting
                    # if check_if_run_stopped():
                    #    print(
                    #        f"Run {wandb.run.name} was stopped before plotting. Exiting.")
                    #    wandb.finish()
                    #    return None
                    print('plotting reliability diagrams')

                    Y_calibration = jnp.hstack(
                        [jnp.ones([batch_size]), jnp.zeros([batch_size])])
                    Y_calibration = np.concatenate(
                        [Y_calibration]*len(val_trawls))

                    all_classifier_outputs = np.array(
                        all_classifier_outputs)

                    # Reliability diagram with equal intervals
                    # diagram_eq = ReliabilityDiagram(
                    #    15, equal_intervals=False)
                    # fig_eq = diagram_eq.plot(
                    #    all_classifier_outputs, Y_calibration).get_figure()
                    # fig_eq.canvas.draw()  # Force render
                    # wandb.log({"Diagram eq": wandb.Image(fig_eq)},
                    #          step=iteration)  # Add step
                    # plt.close(fig_eq)

                    # Reliability diagram with unequal intervals
                    diagram_un = ReliabilityDiagram(
                        15, equal_intervals=True)
                    fig_un = diagram_un.plot(
                        all_classifier_outputs, Y_calibration).get_figure()
                    fig_un.canvas.draw()  # Force render
                    wandb.log({"Diagram uneq": wandb.Image(
                        fig_un)}, step=iteration)  # Add step
                    plt.close(fig_un)

                    # Histogram
                    # hist_beta, ax = plt.subplots()
                    # ax.hist(
                    #    all_classifier_outputs[Y_calibration == 1], label='Y=1', alpha=0.5, density=True)
                    # ax.hist(
                    #    all_classifier_outputs[Y_calibration == 0], label='Y=0', alpha=0.5, density=True)
                    # ax.set_title(
                    #    r'Histogram of $c(\mathbf{x},\mathbf{\theta})$ classifier')
                    # ax.legend(loc='upper center')
                    # hist_beta.canvas.draw()  # Force render
                    # wandb.log({"Histogram": wandb.Image(hist_beta)},
                    #          step=iteration)  # Add step
                    # plt.close(hist_beta)

                    # Log metrics to W&B
            try:
                wandb.log(metrics)
            except Exception as e:
                print(f"Warning: Failed to log metrics to wandb: {e}")
                # Check if run was stopped
                if check_if_run_stopped():
                    print(f"Run {wandb.run.name} was stopped. Exiting.")
                    wandb.finish()
                    return None

        # Save best model info
        best_model_info_path = os.path.join(
            best_model_path, "best_model_info.txt")
        with open(best_model_info_path, 'w') as f:
            f.write(f"Best model iteration: {best_iteration}\n")
            f.write(f"Best validation loss: {best_val_loss:.6f}\n")

        config_save_path = os.path.join(best_model_path, "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(classifier_config, f)

        return True

    except Exception as e:
        print(f"Run failed with error: {e}")
        return False
    finally:
        # Ensure wandb is properly finished
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to finish wandb run. Error: {e}")


if __name__ == "__main__":
    from copy import deepcopy

    # Load config file
    classifier_config_file_path = r'config_files/classifier/TRE_summary_statistics/acf/base_acf_config_Dense.yaml'

    with open(classifier_config_file_path, 'r') as f:
        base_config = yaml.safe_load(f)
        model_name = base_config['model_config']['model_name']

    configurations = []

    if model_name == 'LSTMModel':
        assert model_name == 'LSTMModel'

        for lstm_hidden_size in (128, 32):
            for num_lstm_layers in (3, 2):
                for linear_layer_sizes in ([32, 16, 8, 4], [64, 32, 16, 8, 4], [128, 48, 32, 15, 8, 4, 2]):
                    for mean_aggregation in (False,):  # True):
                        for dropout_rate in (0.05,):  # 0.2):
                            for lr in (0.00025,):  # 0.0005):

                                if (num_lstm_layers <= 2 or lstm_hidden_size < 64) and (linear_layer_sizes[0] <= 2 * lstm_hidden_size) and (dropout_rate <= 0.15 or lstm_hidden_size >= 64):

                                    config_to_use = deepcopy(base_config)
                                    config_to_use['model_config'] = {'model_name': model_name,
                                                                     'lstm_hidden_size': lstm_hidden_size,
                                                                     'num_lstm_layers': num_lstm_layers,
                                                                     'linear_layer_sizes': linear_layer_sizes,
                                                                     'mean_aggregation': mean_aggregation,
                                                                     'final_output_size': 1,
                                                                     'dropout_rate': dropout_rate,
                                                                     'with_theta': True
                                                                     }
                                    config_to_use['optimizer']['lr'] = lr
                                    config_to_use['prng_key'] = np.random.randint(
                                        1, 10**5)

                                    configurations.append(config_to_use)

    elif model_name == 'DenseModel':

        for linear_layer_sizes in ([64, 32, 16, 8], [48, 24, 12, 6], [128, 64, 32, 16, 4],
                                   [24, 12, 8, 6], [32, 24, 16, 8, 4], [
                                       16, 8, 4], [12, 8, 4, 2],
                                   [64, 24, 12, 6, 2], [48, 24, 12, 6, 2]):

            for dropout_rate in (0.025,  0.15):

                for lr in (0.001,  0.00005):

                    for alpha in (0.05, 0.005):

                        config_to_use = deepcopy(base_config)

                        config_to_use['optimizer']['lr'] = lr
                        config_to_use['optimizer']['alpha'] = alpha
                        config_to_use['prng_key'] = np.random.randint(
                            1, 10**5)

                        config_to_use['model_config'] = {'model_name': model_name,
                                                         'linear_layer_sizes': linear_layer_sizes,
                                                         'final_output_size': 1,
                                                         'dropout_rate': dropout_rate,
                                                         'with_theta': True
                                                         }
                        config_to_use['tre_config']['summary_stats_type'] = 'best_model_direct'

                        configurations.append(config_to_use)


############# RUN THROUGH TEH CONFIGURATIONS WHILE DEALING WITH STOPPED RUNS ############
    # Run through all configurations with improved handling
    for config_idx, config in enumerate(configurations):
        print(f"Starting configuration {config_idx+1}/{len(configurations)}")

        # Make sure wandb is clean before starting
        try_to_close_wandb()

        try:
            success = train_classifier(config)

            # Check if training was stopped early (None return value)
            if success is None:
                print(
                    f"Configuration {config_idx+1} was manually stopped. Moving to next configuration.")
                # Extra delay after a manual stop to ensure clean startup for next run
                time.sleep(5)
                continue

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected in main loop.")
            # Try to clean up wandb
            try:
                try_to_close_wandb()

                continue

            except:
                pass

        except Exception as e:
            print(f"Error with configuration {config_idx+1}: {e}")
            print("Continuing to next configuration")

        # Short delay before next configuration
        time.sleep(3)

    print("All configurations completed or program interrupted")
