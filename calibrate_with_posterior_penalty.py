from src.utils.plot_calibration_map import plot_calibration_map
from sklearn.linear_model import LogisticRegression
from jax.scipy.special import logit
from jax.nn import sigmoid
import pandas as pd
import distrax

import optax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.get_model import get_model
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.classifier_utils import get_projection_function, tre_shuffle
from src.utils.monotone_spline_post_training_calibration import fit_spline
from netcal.presentation import ReliabilityDiagram
from src.model.Extended_model_nn import ExtendedModel, VariableExtendedModel
from src.utils.chebyshev_utils import interpolation_points_domain, integrate_from_sampled
import numpy as np
import datetime
import pickle
import yaml
import jax
import os
import netcal
import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
if True:
    from path_setup import setup_sys_path
    setup_sys_path()
    import matplotlib.pyplot as plt


def create_parameter_sweep_fn(apply_fn, tre_type, bounds, N):
    """
    Create a parameter sweep function with model function captured in closure.

    Args:
        apply_fn: Model application function (expects batch inputs)
        tre_type: Parameter to vary ('beta', 'sigma', 'mu')
        bounds: Parameter bounds (min, max)
        N: Number of parameter values to evaluate

    Returns:
        A JIT-compiled function that takes (thetas, x_cache) and returns evaluations
    """
    # Determine which parameter to vary
    param_idx = {'beta': -1, 'sigma': -2, 'mu': -3}[tre_type]

    # Parameter values to evaluate (computed once)
    param_values = interpolation_points_domain(N, bounds[0], bounds[1])

    # Define the inner processing function with apply_fn in closure
    def process_param(p_val, thetas, x_cache):
        batch_size = thetas.shape[0]
        modified = thetas.at[:, param_idx].set(jnp.full(batch_size, p_val))
        results, _ = apply_fn(modified, x_cache)
        return results

    # Create the vectorized version (done once)
    vectorized_process = jax.vmap(process_param, in_axes=(0, None, None))

    # Define and return the JIT-compiled sweep function
    @jax.jit  # No static_argnums needed
    def parameter_sweep(thetas, x_cache):
        """
        Evaluate parameter sweep across the batch.

        Args:
            thetas: Batch of thetas [batch_size, param_dim]
            x_cache: Cached representation

        Returns:
            Evaluations with shape [batch_size, N]
        """
        all_results = vectorized_process(param_values, thetas, x_cache)
        return jnp.transpose(all_results).squeeze()

    return parameter_sweep


def model_apply_wrapper(model, params):

    # Define JIT-ed apply functions

    @jax.jit
    def apply_model_with_x(x, theta):
        """Apply model with a new x input, returning output and x_cache."""
        return model.apply(params, x, theta)

    @jax.jit
    def apply_model_with_x_cache(theta, x_cache):
        """Apply model with cached x representation, returning output and updated x_cache."""
        return model.apply(params, None, theta, x_cache=x_cache)

    return apply_model_with_x, apply_model_with_x_cache


def calibrate_with_posterior_penalty(trained_classifier_path, seq_len, tre_type, N):

    dummy_x = jnp.ones([1, seq_len])
    trawl_process_type = 'sup_ig_nig_5p'
    model, params, prior, bounds = load_one_tre_model_only_and_prior_and_bounds(
        trained_classifier_path, dummy_x, trawl_process_type, tre_type)

    apply_model_with_x, apply_model_with_x_cache = model_apply_wrapper(
        model, params)
    # vmapped_apply_model_with_x_cache = jax.jit(jax.vmap(apply_model_with_x_cache, in_axes = (0, None)))

    # dataset paths: part 1 - trawls and thetas
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        trained_classifier_path))),  f'cal_dataset_{seq_len}')
    # load validation dataset
    cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
    cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

    assert os.path.isfile(cal_x_path) and os.path.isfile(
        cal_thetas_path) and os.path.isfile(cal_Y_path)

    # load theta and Y
    cal_thetas_array = np.load(cal_thetas_path, mmap_mode='r')
    cal_Y_array = np.load(cal_Y_path, mmap_mode='r')

    # dataset paths: evaluated functions
    classifier_evaluations_path = os.path.join(
        dataset_path, f'{tre_type}_{N}.npy')

    # DO EVALUATIONS for CLENSHAW QUADARATURE
    if not os.path.isfile(classifier_evaluations_path):

        cal_x_array = np.load(cal_x_path, mmap_mode='r')
        nr_batches = cal_x_array.shape[0]
        batch_size = cal_x_array.shape[1]

        if tre_type == 'acf':
            full_shape = (nr_batches, batch_size, N, N)

        else:
            full_shape = (nr_batches, batch_size, N)

        with np.lib.format.open_memmap(classifier_evaluations_path, mode='w+',
                                       dtype=np.float32, shape=full_shape) as results:

            for i in range(nr_batches):

                cal_x_batch = jnp.array(cal_x_array[i])
                cal_thetas_batch = jnp.array(cal_thetas_array[i])
                _, x_cache = apply_model_with_x(cal_x_batch, cal_thetas_batch)

                if tre_type == 'acf':

                    raise ValueError

                else:

                    parameter_sweep_func = create_parameter_sweep_fn(
                        apply_model_with_x_cache, tre_type, bounds, N)
                    results[i] = np.array(
                        parameter_sweep_func(cal_thetas_batch, x_cache))

                    # Optionally flush to ensure writing to disk
                    if i % 20 == 0:
                        print(i)
                        results.flush()

        del results

    # EVALUATIONS ARE ALREADY DONE
    else:

        # evaluations for clenshaw quadrature
        sweeped_values = np.load(classifier_evaluations_path, mmap_mode='r')

        # BCE calculation
        log_r = np.load(log_r_path)
        pred_prob_Y = np.load(pred_prob_Y_path)
        Y = np.load(Y_path)
        # [nr_batches,batch_size, N] for mu, sigma, beta or
        # [nr_batches, batch_size, N,N] for acf


def get_x_cache(model, x_batch):

    pass


if __name__ == '__main__':

    trained_classifier_path = 'D:\\sbi_ambit\\SBI_for_trawl_processes_and_ambit_fields\\models\\new_classifier\\TRE_full_trawl\\beta\\04_12_04_26_56'
    seq_len = 1500
    dummy_x = jnp.ones([1, seq_len])
    trawl_process_type = 'sup_ig_nig_5p'
    tre_type = 'beta'
    N = 100
    model, params, prior = load_one_tre_model_only_and_prior(
        trained_classifier_path, dummy_x, trawl_process_type, tre_type)
