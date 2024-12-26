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
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.get_model import get_model
from src.utils.utils import update_step, summary_stats_loss_fn
if True:
    from path_setup import setup_sys_path
    setup_sys_path()


def train_and_evaluate(config_file_path):

    # Load config
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(project="SBI_trawls", config=config)

    ###########################################################################
    # Get params and hyperparams
    learn_config = config['learn_config']
    learn_acf = learn_config['learn_acf']
    learn_marginal = learn_config['learn_marginal']
    learn_both = learn_config['learn_both']

    assert learn_acf + learn_marginal == 1 and learn_both == False

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

        if learn_acf:
            val_data.append((trawl_val, theta_acf_val))
        elif learn_marginal:
            val_data.append((trawl_val, theta_marginal_jax_val))

    # Convert validation data to JAX arrays
    val_trawls = jnp.stack([x[0] for x in val_data])
    val_thetas = jnp.stack([x[1] for x in val_data])

    print(f'{val_batches} batches simulated for the validation dataset.')

    ###########################################################################
    # Create directory for validation data and model checkpoints
    base_checkpoint_dir = "models"
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
    def compute_loss(params, trawl, theta_acf_or_marginal_jax, p=1):
        """Base loss function without gradients."""
        pred_theta = model.apply(params, trawl)
        loss = jnp.mean(
            jnp.abs(theta_acf_or_marginal_jax - pred_theta)**p)**(1/p)
        return loss

    # get grads for training
    compute_loss_and_grad = jax.jit(jax.value_and_grad(compute_loss))

    @jax.jit
    def compute_validation_loss(params, val_trawls, val_thetas):
        """Compute mean and standard deviation of validation loss."""
        def body_fun(i, acc):
            trawl_val = jax.lax.dynamic_slice_in_dim(val_trawls, i, 1)[0]
            theta_val = jax.lax.dynamic_slice_in_dim(val_thetas, i, 1)[0]
            loss = compute_loss(params, trawl_val, theta_val)
            return acc + (loss, loss**2)

        total_loss, total_loss_sq = jax.lax.fori_loop(
            0, val_trawls.shape[0], body_fun, (jnp.array(0.), jnp.array(0.))
        )

        mean_loss = total_loss / val_trawls.shape[0]
        std_loss = jnp.sqrt(
            (total_loss_sq / val_trawls.shape[0]) - (mean_loss**2)
        )
        return mean_loss, std_loss

    # Initialize best validation loss tracking
    best_val_loss = float('inf')
    best_iteration = -1
    best_model_path = os.path.join(experiment_dir, "best_model")
    ###########################################################################

    # @jax.jit
    # def compute_loss_and_grad_acf_and_marginal(params, trawl, theta_acf, theta_marginal_jax):
    #    """
    #    Compute loss and gradients.
    #    """
    #    def loss_fn_wrapped(params):
    #        pred_theta = model.apply(params, trawl)
    #        total_loss, (acf_loss, marginal_loss) = summary_stats_loss_fn(
    #            pred_theta, theta_acf, theta_marginal_jax)
    #        return total_loss, (acf_loss, marginal_loss)
    #
    #    (total_loss, (acf_loss, marginal_loss)), grads = jax.value_and_grad(
    #        lambda params: loss_fn_wrapped(params), has_aux=True
    #    )(params)
    #    return total_loss, acf_loss, marginal_loss, grads

    ###########################################################################

    # Training loop
    for iteration in range(config["train_config"]["n_iterations"]):

        theta_acf, key = theta_acf_simulator(key)
        theta_marginal_jax, theta_marginal_tf, key = theta_marginal_simulator(
            key)
        trawl, key = trawl_simulator(theta_acf, theta_marginal_tf, key)

        # Compute loss and gradients
        if learn_acf:
            loss, grads = compute_loss_and_grad(
                state.params, trawl, theta_acf)

        elif learn_marginal:
            loss, grads = compute_loss_and_grad(
                state.params, trawl, theta_marginal_jax)

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
            val_mean_loss, val_std_loss = compute_validation_loss(
                state.params, val_trawls, val_thetas
            )
            val_loss_ci_lower = val_mean_loss - 1.96 * val_std_loss
            val_loss_ci_upper = val_mean_loss + 1.96 * val_std_loss

            metrics.update({
                val_loss: val_mean_loss.item(),
                val_loss+"_ci_lower": val_loss_ci_lower.item(),
                val_loss+"_ci_upper": val_loss_ci_upper.item()
            })

            ###################################################################
            # WHY DOES THIS HANG???
            # Save regular checkpoints
            # checkpoints.save_checkpoint(
            #    ckpt_dir=checkpoint_dir,
            #    target=state,
            #    step=iteration,
            #    prefix="checkpoint_iter_",
            #    overwrite=False
            # )

            # Use this to load checkpoint
            # state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=iteration_number)
            ###################################################################

            # Save just the parameters instead of full state
            params_filename = os.path.join(
                experiment_dir, f"params_iter_{iteration}.pkl")
            with open(params_filename, 'wb') as f:
                pickle.dump(state.params, f)

            # Keep track of best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iteration

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


if __name__ == "__main__":
    import glob
    # Loop over configs
    for config_file_path in glob.glob("config_files/*.yaml"):
        train_and_evaluate(config_file_path)


# TO DO: Log in loss per parameter, without names, just numbers
# add gradient acumulation maybe, maybe not really
# play with optimization techniques
# add clasifier
# add metrics to classifier

# for acf, display a boxplot of absolute / percentage wise differences in true and infered acfs
