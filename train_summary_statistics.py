# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:42:34 2024

@author: dleon
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:43:15 2024

@author: dleon
"""


from src.utils.utils import update_step, summary_stats_loss_fn
from src.utils.get_model import get_model
from src.utils.get_data_generator import get_theta_and_trawl_generator
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
from src.utils.acf_functions import get_acf, plot_theoretical_empirical_inferred_acf
from src.utils.KL_divergence import vec_monte_carlo_kl_3_param_nig
from functools import partial
if True:
    from path_setup import setup_sys_path
    setup_sys_path()
# config_file_path = 'config_files/summary_statistics/Transformer\\config1.yaml'


def train_and_evaluate(config_file_path):

    # Load config
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    ###########################################################################
    # Get params and hyperparams
    learn_config = config['learn_config']
    learn_acf = learn_config['learn_acf']
    learn_marginal = learn_config['learn_marginal']
    learn_both = learn_config['learn_both']

    assert learn_acf + learn_marginal == 1 and learn_both == False

    # Initialize wandb
    group_name = (
        "acf" if learn_acf else
        "marginal" if learn_marginal else
        "both")
    wandb.init(project="SBI_trawls", group=group_name, config=config)

    trawl_config = config['trawl_config']
    batch_size = trawl_config['batch_size']

    # more hyperparams for the loss function
    loss_config = config['loss_config']
    p = loss_config['p']

    if learn_acf:

        use_acf_directly = loss_config['use_acf_directly']
        nr_acf_lags = loss_config['nr_acf_lags']
        acf_func = jax.jit(
            jax.vmap(get_acf(trawl_config['acf']), in_axes=(None, 0)))

    elif learn_marginal:

        use_kl_div = loss_config['use_kl_div']
        num_KL_samples = loss_config['num_KL_samples']
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
    np.save(os.path.join(val_data_dir, "val_trawls.npy"), np.array(val_trawls))
    np.save(os.path.join(val_data_dir, "val_thetas.npy"), np.array(val_thetas))

    ###########################################################################
    # Create model and initialize parameters
    model, params, key = get_model(config)
    key = jax.random.split(PRNGKey(config['prng_key']+2351), batch_size)

    # Initialize optimizer
    lr = config["optimizer"]["lr"]
    total_steps = config["train_config"]["n_iterations"]
    warmup_steps = 250
    decay_steps = total_steps - warmup_steps

    # Create learning rate schedule
    # schedule_fn = optax.piecewise_constant_schedule(
    #    init_value=lr,
    #    boundaries_and_scales={250: lr/10}  # At step 100, multiply lr by 0.1
    # )
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

    # dropout_seed = np.random.randint(low=1, high=10**4)
    # dropout_rng = jax.random.PRNGKey(dropout_seed)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    ###########################################################################
    # Loss functions

    @partial(jax.jit, static_argnames=('num_KL_samples',))
    def compute_loss(params, trawl, theta_acf_or_marginal_jax, dropout_rng, p, key, num_KL_samples=10**3):
        """Base loss function for training with dropout RNG handling.
        Args:

            params: weights and biases of the model
            trawl: realisation of the trawl process, jnp array of shape [batch_dim, seq_len]
            theta_acf_or_marginal_jax: 
            p: integer that specifies which p norm we use, if we use the p norm. see below
            key: jax.random.PRNGKey, without a batch size, as per flax's convetion

        Returns:

            to add

        Based on the config file, the loss can be computed in one of multiple ways.
        We can either compare 
        """
        pred_theta = state.apply_fn(
            params,
            trawl,
            train=True,  # Hardcoded since this is only for training
            rngs={'dropout': dropout_rng}
        )
        if learn_acf and use_acf_directly:

            # predict log_acf_params
            pred_theta = jnp.exp(pred_theta)

            # compute L^p distance between true and inferred acf
            H = jnp.arange(1, nr_acf_lags+1)
            pred_acf = acf_func(H, pred_theta)
            theoretical_acf = acf_func(
                H, theta_acf_or_marginal_jax)  # this is theta_acf
            loss = jnp.mean(
                jnp.abs((pred_acf - theoretical_acf))**p)**(1/p)

        elif learn_marginal and use_kl_div:

            # predict mu, log_scale, beta

            pred_mu, pred_log_scale, pred_beta = pred_theta[:, [
                0]], pred_theta[:, [1]], pred_theta[:, [2]]
            pred_theta = jnp.concatenate(
                [pred_mu, jnp.exp(pred_log_scale), pred_beta], axis=1)

            # compute KL divergence between true and infered NIG distr
            KL_key = jax.random.split(dropout_rng, batch_size)

            loss = vec_monte_carlo_kl_3_param_nig(theta_acf_or_marginal_jax,
                                                  pred_theta, KL_key, num_KL_samples)

            loss = jnp.mean(loss)

        else:

            # predict actual params

            loss = jnp.mean(
                jnp.abs(theta_acf_or_marginal_jax - pred_theta)**p)**(1/p)
        return loss

    # get grads for training
    # key and num_KL_samples are only used when we learn the marginal params
    # via minimized MC approximation of the KL divergence
    compute_loss_and_grad = jax.jit(jax.value_and_grad(
        compute_loss), static_argnames=('num_KL_samples',))

    @partial(jax.jit, static_argnames=('num_KL_samples',))
    def compute_validation_stats(params, val_trawls, val_thetas, p, key, num_KL_samples):
        """Compute mean and std of validation loss."""
        def body_fun(i, acc):

            theta_val = jax.lax.dynamic_slice_in_dim(val_thetas, i, 1)[0]
            trawl_val = jax.lax.dynamic_slice_in_dim(val_trawls, i, 1)[0]
            if learn_acf:
                trawl_val = (trawl_val - jnp.mean(trawl_val, axis=1,
                             keepdims=True))/jnp.std(trawl_val, axis=1, keepdims=True)
            # No dropout during validation
            pred_theta = state.apply_fn(params, trawl_val, train=False)
            # this is actually on the log scale for acf

            if learn_acf and use_acf_directly:

                pred_theta = jnp.exp(pred_theta)

                H = jnp.arange(1, nr_acf_lags+1)
                pred_acf = acf_func(H, pred_theta)
                theoretical_acf = acf_func(H, theta_val)  # this is theta_acf
                loss = jnp.mean(
                    jnp.abs((pred_acf - theoretical_acf))**p)**(1/p)

            elif learn_marginal and use_kl_div:

                raise ValueError('not yet implemented')

            else:
                loss = jnp.mean(jnp.abs(theta_val - pred_theta)**p)**(1/p)

            return acc + jnp.array([loss, loss**2])

        total = jax.lax.fori_loop(
            0, val_trawls.shape[0], body_fun, jnp.zeros(2)
        )

        n = val_trawls.shape[0]
        mean = total[0] / n
        variance = (total[1] / n) - (mean**2)
        std = jnp.sqrt(jnp.maximum(variance, 0.0))

        return mean, std

    # Initialize best validation loss tracking
    best_val_loss = float('inf')
    best_iteration = -1
    best_model_path = os.path.join(experiment_dir, "best_model")
    ###########################################################################

    # Training loop
    for iteration in range(config["train_config"]["n_iterations"]):

        # Split RNG for simulation and dropout
        if key.ndim > 1:
            dropout_key = jax.random.split(key[0])[0]
        else:
            dropout_key = jax.random.split(key)[0]

        theta_acf, key = theta_acf_simulator(key)
        theta_marginal_jax, theta_marginal_tf, key = theta_marginal_simulator(
            key)
        trawl, key = trawl_simulator(theta_acf, theta_marginal_tf, key)

        # Compute loss and gradients
        if learn_acf:

            standardized_trawl = (
                trawl - jnp.mean(trawl, axis=1, keepdims=True))/jnp.std(trawl, axis=1, keepdims=True)

            loss, grads = compute_loss_and_grad(
                state.params,
                standardized_trawl,
                theta_acf,
                dropout_key,
                p,
                key)  # Use fresh dropout key
        elif learn_marginal:
            loss, grads = compute_loss_and_grad(
                state.params,
                trawl,
                theta_marginal_jax,
                dropout_key,
                p,
                key)  # Use fresh dropout key

        # Update model parameters
        state = update_step(state, grads)

        # Validation and logging
        loss_name = 'acf_loss' if learn_acf else 'marginal_loss'
        train_loss, val_loss = 'train_' + loss_name, 'val_' + loss_name
        metrics = {
            train_loss: loss.item()
        }

        # Compute validation loss periodically
        if iteration % val_freq == 0:

            val_loss, val_loss_std = compute_validation_stats(
                state.params, val_trawls, val_thetas, p)

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

            # upload plots as well

            if learn_acf:  # TO CHANGE

                pred_theta = state.apply_fn(
                    state.params,
                    standardized_trawl,
                    train=False
                )

                pred_theta = jnp.exp(pred_theta)
                for i in range(5):
                    fig_ = plot_theoretical_empirical_inferred_acf(
                        standardized_trawl[i], theta_acf[i], pred_theta[i], 'sup_IG', nr_acf_lags)

                    wandb.log({f"Acf plot {i}": wandb.Image(fig_)})

            elif learn_marginal:
                pass

        wandb.log(metrics)

    # Final best model summary
    print(f"\nTraining completed. Best model:")
    print(f"Iteration: {best_iteration}")
    print(f"Validation Loss: {best_val_loss:.6f}")

    summary_filename = os.path.join(val_data_dir, "training_summary.txt")
    with open(summary_filename, 'w') as f:
        f.write("Training Summary\n")
        f.write("================\n\n")
        f.write(f"Best Model Information:\n")
        f.write(f"Iteration: {best_iteration}\n")
        f.write(f"Validation Loss: {best_val_loss:.6f}\n")
        # f.write(
        #    f"Validation Loss Confidence Interval: [{val_loss_ci_lower:.6f}, {val_loss_ci_upper:.6f}]\n")
        f.write(
            f"\nTraining completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    wandb.finish()

    save_config_path = os.path.join(val_data_dir, 'config')
    with open(save_config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


if __name__ == "__main__":
    import glob
    # Loop over configs
    for config_file_path in glob.glob("config_files/summary_statistics/LSTM/*.yaml"):
        train_and_evaluate(config_file_path)

    # config_file_path = 'config_files/summary_statistics/Transformer\\config1.yaml'


# TO DO: Log in loss per parameter, without names, just numbers
# add gradient acumulation maybe, maybe not really
# play with optimization techniques
# add clasifier
# add metrics to classifier

# for acf, display a boxplot of absolute / percentage wise differences in true and infered acfs
