from jax.scipy.special import logit
from jax.nn import sigmoid
import pandas as pd
# import distrax

# import optax
import jax.numpy as jnp
from jax.random import PRNGKey
from functools import partial
# from statsmodels.tsa.stattools import acf as compute_empirical_acf
from src.utils.get_model import get_model
from src.utils.get_trained_models import load_one_tre_model_only_and_prior_and_bounds
from src.utils.get_data_generator import get_theta_and_trawl_generator
from src.utils.reconstruct_beta_calibration import beta_calibrate_log_r
# from src.utils.classifier_utils import get_projection_function, tre_shuffle
# from src.utils.monotone_spline_post_training_calibration import fit_spline
# from netcal.presentation import ReliabilityDiagram
from src.model.Extended_model_nn import ExtendedModel, VariableExtendedModel
from src.utils.chebyshev_utils import interpolation_points_domain, vec_polyfit_domain, vec_sample_from_coeff, sample_from_coeff, integrate_from_sampled, polyfit_domain, chebval_ab_for_one_x
from src.utils.chebyshev_utils import vec_chebval_ab_for_multiple_x_per_envelope_and_multple_envelopes as vec_chebval
import numpy as np
import datetime
import pickle
import yaml
import jax
import os
# import netcal
import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
if True:
    from path_setup import setup_sys_path
    setup_sys_path()


def create_parameter_sweep_fn_for_2nd_acf_params(apply_fn, N):
    """
    Create a parameter sweep function with model function captured in closure.

    Args:
        apply_fn: Model application function (expects batch inputs), cached version
        bounds: Parameter bounds (min, max)
        N: Number of parameter values to evaluate

    Returns:
        A JIT-compiled function that takes (thetas, x_cache) and returns evaluations
    """
    # vary 2nd parameter, which has index 1
    param_idx = 1
    bounds = (10, 20)  # BOUNDS FOR 2ND ACF PARAMETER
    tre_type = 'acf'

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


@partial(jax.jit, static_argnames=('nr_samples',))
def estimate_first_density(x_cache_to_use, key, true_theta, nr_samples):

    thetas = jnp.zeros((N, 5))
    thetas = thetas.at[:, 0].set(
        interpolation_points_domain(N, bounds[0], bounds[1]))
    x_cache_to_use_expanded = jnp.broadcast_to(
        x_cache_to_use, (N, x_cached_shape))  # x_cache_size
    log_f_x_y = evaluate_at_chebyshev_knots(thetas, x_cache_to_use_expanded)

    if apply_calibration:
        log_f_x_y = beta_calibrate_log_r(log_f_x_y,
                                         calibration_params['params'])

    f_x_y = jnp.exp(log_f_x_y)
    f_x = vec_integrate_from_sampled(f_x_y)
    cheb_coeff_f_x = polyfit_domain(f_x, bounds[0], bounds[1])

    key, subkey = jax.random.split(key)
    samples = sample_from_coeff(
        cheb_coeff_f_x, subkey, bounds[0], bounds[1],  nr_samples)

    prob_at_samples = chebval_ab_for_one_x(samples, cheb_coeff_f_x, a, b)

    prob_at_true_value = chebval_ab_for_one_x(
        true_theta[0], cheb_coeff_f_x, a, b)

    # also return proability at true value and samples

    return samples, prob_at_samples, prob_at_true_value, key


@partial(jax.jit, static_argnames=('root_nr_samples'))
def do_acf_sampling(theta_first_component_to_use, x_cache_to_use, vec_key, root_nr_samples):

    bounds = (10, 20)
    a, b = bounds

    split_keys = jax.vmap(lambda k: jax.random.split(k, num=2))(vec_key)
    next_vec_key = split_keys[:, 0]  # Use first split from each key

    thetas = jnp.zeros((root_nr_samples, 5)).at[:, 0].set(
        theta_first_component_to_use)
    x_cache_to_use_expanded = jnp.broadcast_to(
        x_cache_to_use, (root_nr_samples, x_cached_shape))  # x_cache_size

    log_prob_envelope = evaluate_at_chebyshev_knots(
        thetas, x_cache_to_use_expanded)

    if apply_calibration:
        log_prob_envelope = beta_calibrate_log_r(log_prob_envelope,
                                                 calibration_params['params'])

    coeff = vec_polyfit_domain(jnp.exp(log_prob_envelope), a, b)
    conditional_samples = vec_sample_from_coeff(
        coeff, vec_key, a, b, 1)

    # approximate density at posterior samples
    cond_prob_at_samples = vec_chebval(conditional_samples, coeff, a, b)
    # true_cond_prob = chebval_ab_for_one_x(true_theta[1],coeff,a,b)

    return conditional_samples, cond_prob_at_samples, next_vec_key


@jax.jit
def get_cond_prob_at_true_value(true_theta, x_cache_to_use):

    log_prob_envelope = evaluate_at_chebyshev_knots(
        true_theta[jnp.newaxis, :], x_cache_to_use)

    if apply_calibration:
        log_prob_envelope = beta_calibrate_log_r(log_prob_envelope,
                                                 calibration_params['params'])

    coeff = polyfit_domain(jnp.exp(log_prob_envelope), a, b)
    cond_prob_at_true_sample = chebval_ab_for_one_x(true_theta[1], coeff, a, b)
    return cond_prob_at_true_sample


if __name__ == '__main__':

    tre_type = 'acf'
    # trained_classifier_path = f'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/new_classifier/TRE_full_trawl/selected_models/{tre_type}'
    trained_classifier_path = f'D:\\sbi_ambit\\SBI_for_trawl_processes_and_ambit_fields\\models\\new_classifier\\TRE_full_trawl\\selected_models\\{tre_type}'
    seq_len = 1500
    dummy_x = jnp.ones([1, seq_len])
    trawl_process_type = 'sup_ig_nig_5p'
    N = 128
    num_rows_to_load = 782  # nr data points is 64 * num_rows_to_load
    # num_envelopes_to_build_at_once = 5
    bounds = (10, 20)
    a, b = bounds
    root_nr_samples = 71
    nr_samples = root_nr_samples**2
    beta_calibration_indicator = True
    assert beta_calibration_indicator

    apply_calibration = True

    # for sampling of the 1st component
    key = jax.random.PRNGKey(np.random.randint(1, 100000))
    # for sampling 2nd component, conditional on 1st componentt
    vec_key = jax.random.PRNGKey(np.random.randint(1, 100000))
    vec_key = jax.random.split(vec_key, root_nr_samples)

    # Create a partial function with fixed a, b

    def integrate_partial(samples):
        return integrate_from_sampled(samples, a=a, b=b)
    vec_integrate_from_sampled = jax.jit(jax.vmap(integrate_partial))

    rank_list = []

    # get calibratiton

    if beta_calibration_indicator:

        calibratiton_file_name = f'beta_calibration_{seq_len}.pkl'

    # else:
    #
    #    calibratiton_file_name = 'no_calibration.pkl'

    with open(os.path.join(trained_classifier_path, calibratiton_file_name), 'rb') as file:
        calibration_params = pickle.load(file)

    assert tre_type in trained_classifier_path
    model, params, _, __bounds = load_one_tre_model_only_and_prior_and_bounds(
        trained_classifier_path, dummy_x, trawl_process_type, tre_type)

    # HARD CODED BOUNDS

    # LOAD DATA
    # Load dataset
    dataset_path = os.path.join(os.getcwd(), 'models',
                                'val_dataset', f'val_dataset_{seq_len}')
    val_x_path = os.path.join(dataset_path, 'val_x_joint.npy')
    val_thetas_path = os.path.join(dataset_path, 'val_thetas_joint.npy')

    # Load first few rows of val_x with memory mapping
    val_x = np.load(val_x_path, mmap_mode='r')[:num_rows_to_load]
    val_thetas = np.load(val_thetas_path)[:num_rows_to_load]

    val_x = val_x.reshape(-1, seq_len)
    val_thetas = val_thetas.reshape(-1, val_thetas.shape[-1])

    # LOAD FUNCTIONS
    apply_model_with_x, apply_model_with_x_cache = model_apply_wrapper(
        model, params)
    evaluate_at_chebyshev_knots = create_parameter_sweep_fn_for_2nd_acf_params(
        apply_model_with_x_cache,  N+1)

    _, __ = apply_model_with_x(
        jnp.array(val_x[[0]]), jnp.array(val_thetas[[0]]))
    x_cached_shape = __.shape[-1]

    # for i, (batch_thetas, batch_x) in enumerate(zip(theta_batches, x_batches)):
    for i, (theta_to_use, x_to_use) in enumerate(zip(val_thetas, val_x)):
        if i % 50 == 0:
            print(i)

        # get x_cache
        _, x_cache_to_use = apply_model_with_x(
            jnp.array(x_to_use.reshape(1, -1)), jnp.array(theta_to_use.reshape(1, -1)))

        # sample 1st component of the acf
        samples_1_comp, prob_at_1_comp, prob_at_1_true, key = estimate_first_density(
            x_cache_to_use, key, theta_to_use, nr_samples)
        # break it into batches to make avoid memory issues
        samples_1_comp_batches = np.array_split(
            samples_1_comp, root_nr_samples)

        samples_2_comp = []
        prob_at_2_comp = []

        for theta_first_component_to_use in samples_1_comp_batches:

            conditional_samples, cond_prob_at_samples, vec_key = do_acf_sampling(theta_first_component_to_use,
                                                                                 x_cache_to_use, vec_key,  root_nr_samples)

            # append conditional samples and conditional probabilities
            samples_2_comp.append(conditional_samples)
            prob_at_2_comp.append(cond_prob_at_samples)

        #
        # jnp.array(samples_2_comp).squeeze()
        samples_2_comp = jnp.vstack(samples_2_comp).squeeze()
        # jnp.vstack(prob_at_2_comp)
        # jnp.array(prob_at_2_comp).squeeze()
        prob_at_2_comp = jnp.vstack(prob_at_2_comp).squeeze()

        prob_at_2_true = get_cond_prob_at_true_value(
            jnp.array(theta_to_use), x_cache_to_use)
        # to compute the trtue conditional probability, multiply them etc

        true_probability = prob_at_1_true * prob_at_2_true
        prob_at_samples = prob_at_1_comp * prob_at_2_comp
        rank_list.append(np.mean(true_probability <
                                 prob_at_samples))

    results_path = os.path.join(os.getcwd(), 'models', 'new_classifier', 'TRE_full_trawl',
                                'selected_models', 'per_classifier_coverage_check', str(tre_type))
    os.makedirs(results_path, exist_ok=True)
    if apply_calibration:
        file_path = f'{tre_type}_cal_ranks_seq_len_{seq_len}_N_{N}.npy'
    else:
        file_path = f'{tre_type}_uncal_ranks_seq_len_{seq_len}_N_{N}.npy'

    np.save(file=file_path, arr=rank_list)
