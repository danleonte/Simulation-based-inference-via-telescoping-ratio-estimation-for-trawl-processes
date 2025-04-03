import os
import glob
import yaml
import pickle
import pandas as pd
import gc  # Import garbage collector
import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad
from scipy.optimize import minimize
from posterior_sampling_utils import run_mcmc_for_trawl, save_results, create_and_save_plots
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models


def minus_like_with_grad_wrapper(trawl_to_use):

    def like(theta):
        return approximate_log_likelihood_to_evidence(trawl_to_use[jnp.newaxis, :], theta[jnp.newaxis, :])[0, 0]

    # Use value_and_grad to compute both function value and gradient
    like_and_grad = jit(value_and_grad(like))

    def minus_like_with_grad(theta_np):
        theta_jax = jnp.array(theta_np)
        value, gradient = like_and_grad(theta_jax)
        return -float(value), -np.array(gradient)

    return minus_like_with_grad


def get_MLE(trawl_subfolder):

    trawl_idx = int(os.path.basename(trawl_subfolder).removeprefix("trawl_"))
    result_file_path = os.path.join(trawl_subfolder, 'results.pkl')
    if not os.path.isfile(result_file_path):
        return None

    with open(result_file_path, 'rb') as f:
        results = pickle.load(f)

    log_likelihoods = results['log_likelihood_samples']
    max_index = np.unravel_index(
        np.argmax(log_likelihoods), log_likelihoods.shape)
    max_mcmc_value = log_likelihoods[max_index].item()

    # THE NAMES OF GAMMA AND ETA ARE SWAPPED IN MODEL VEC
    # SWAP GAMMA AND ETA IN MCMC

    diff = results['posterior_samples']['sigma'].shape[1] - \
        results['log_likelihood_samples'].shape[1]
    max_index = (max_index[0], max_index[1]+diff)
    mcmc_starting_point = jnp.array([results['posterior_samples'][key][max_index] for key in
                                     ['eta', 'gamma', 'mu', 'sigma', 'beta']])

    true_trawl, true_theta = jnp.array(
        results['true_trawl']), jnp.array(results['true_theta'])

    # SANITY CHECK
    assert 0.9999 * max_mcmc_value < approximate_log_likelihood_to_evidence(
        true_trawl[jnp.newaxis, :], mcmc_starting_point[jnp.newaxis, :]).item(), trawl_subfolder
    # BECAUSE WE HAD TO SWAP ETA AND GAMMA HERE

    do_bfgs = False
    if do_bfgs:
        func_to_optimize = minus_like_with_grad_wrapper(true_trawl)

        # Use method that accepts gradients
        result_from_mcmc = minimize(func_to_optimize, np.array(mcmc_starting_point),
                                    method='L-BFGS-B', jac=True, bounds=((10, 20), (10, 20), (-1, 1), (0.5, 1.5), (-5, 5)))

        result_from_true = minimize(func_to_optimize, np.array(true_theta),
                                    method='L-BFGS-B', jac=True, bounds=((10, 20), (10, 20), (-1, 1), (0.5, 1.5), (-5, 5)))

        likelihoods = [max_mcmc_value, -
                       result_from_mcmc.fun, - result_from_true.fun]
        thetas = [mcmc_starting_point, result_from_mcmc.x, result_from_true.x]

        best_index = np.argmax(likelihoods)
        return trawl_idx, true_theta, thetas[best_index], likelihoods[best_index]

    else:
        return trawl_idx, true_theta, mcmc_starting_point, max_mcmc_value


if __name__ == '__main__':

    # folder_path = r'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/classifier/NRE_full_trawl/uncalibrated'
    folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\NRE_full_trawl\uncalibrated'

    # r'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/classifier/TRE_full_trawl/beta_calibrated/'
    # Get all matching folders
    trawl_subfolders = [
        f for f in glob.glob(os.path.join(folder_path, 'mcmc_results_sup_ig_nig_5p', "trawl_*"))
        if os.path.isdir(f)
    ]

    # Set up model configuration
    use_tre = 'TRE' in folder_path
    if not (use_tre or 'NRE' in folder_path):
        raise ValueError("Path must contain 'TRE' or 'NRE'")

    use_summary_statistics = 'summary_statistics' in folder_path
    if not (use_summary_statistics or 'full_trawl' in folder_path):
        raise ValueError(
            "Path must contain 'full_trawl' or 'summary_statistics'")

    if use_tre:
        classifier_config_file_path = os.path.join(
            folder_path, 'acf', 'config.yaml')
    else:
        classifier_config_file_path = os.path.join(folder_path, 'config.yaml')

    with open(classifier_config_file_path, 'r') as f:
        a_classifier_config = yaml.safe_load(f)
        trawl_process_type = a_classifier_config['trawl_config']['trawl_process_type']
        seq_len = a_classifier_config['trawl_config']['seq_len']

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(
        os.path.dirname(folder_path)), 'cal_dataset')
    cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
    cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

    cal_Y = jnp.load(cal_Y_path)
    true_trawls = jnp.load(cal_x_path)[:, cal_Y == 1].reshape(-1, seq_len)
    true_thetas = jnp.load(cal_thetas_path)
    true_thetas = true_thetas[:, cal_Y == 1].reshape(-1, true_thetas.shape[-1])
    del cal_Y

    # Load approximate likelihood function
    approximate_log_likelihood_to_evidence, _, _ = load_trained_models(
        folder_path, true_trawls[[0], ::-1], trawl_process_type,
        use_tre, use_summary_statistics
    )

    idx_list = []
    true_theta_list = []
    MLE_list = []
    likelihood_list = []
    count = 0
    for trawl_subfolder in trawl_subfolders:

        count += 1
        if count % 25 == 1:
            print(count)
            gc.collect()

        result_to_add = get_MLE(trawl_subfolder)

        if result_to_add == None:
            pass
        else:
            idx_list.append(result_to_add[0])
            true_theta_list.append(result_to_add[1])
            MLE_list.append(result_to_add[2])
            likelihood_list.append(result_to_add[3])

    df = pd.DataFrame({
        'idx': idx_list,
        'true_theta': true_theta_list,
        'MLE': MLE_list,
        'likelihoods': likelihood_list
    })

    results_path = os.path.join(folder_path, 'results')
    os.makedirs(results_path,  exist_ok=True)
    df.to_pickle(os.path.join(results_path, 'MLE_results.pkl'))
