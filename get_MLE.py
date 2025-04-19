import os
import glob
import yaml
import pickle
import pandas as pd
import gc  # Import garbage collector
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, value_and_grad
from scipy.optimize import minimize
from posterior_sampling_utils import run_mcmc_for_trawl, save_results, create_and_save_plots
from src.utils.get_trained_models import load_trained_models_for_posterior_inference as load_trained_models
from tqdm import tqdm


def minus_like_with_grad_wrapper(trawl_to_use):

    log_like = wrapper_for_approx_likelihood_just_theta(
        jnp.array(trawl_to_use)[jnp.newaxis, :])

    # Define a function that returns a scalar by indexing or using item()
    def scalar_log_like(theta):
        result = log_like(theta)
        # Extract the scalar from the (1,) shape array
        return result[0]  # or result.sum() or result.item()

    # Use value_and_grad on the scalar function
    like_and_grad = jit(value_and_grad(scalar_log_like))

    def minus_like_with_grad(theta_np):
        theta_jax = jnp.array(theta_np)
        value, gradient = like_and_grad(theta_jax)
        return -float(value), -np.array(gradient)

    return minus_like_with_grad


def get_MLE(trawl_to_use, theta_to_use):

    # SANITY CHECK
    # assert 0.9999 * max_mcmc_value < approximate_log_likelihood_to_evidence(
    #    true_trawl[jnp.newaxis, :], mcmc_starting_point[jnp.newaxis, :]).item(), trawl_subfolder
    # BECAUSE WE HAD TO SWAP ETA AND GAMMA HERE

    func_to_optimize = minus_like_with_grad_wrapper(trawl_to_use)
    result_from_true = minimize(func_to_optimize, np.array(theta_to_use),
                                method='L-BFGS-B', jac=True, bounds=((10, 20), (10, 20), (-1, 1), (0.5, 1.5), (-5, 5)))

    return result_from_true


if __name__ == '__main__':

    # folder_path = r'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/classifier/NRE_full_trawl/uncalibrated'
    folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\new_classifier\TRE_full_trawl\selected_models'
    seq_len = 1000
    num_rows_to_load = 125
    num_trawls_to_use = 500
    calibration_filename = 'spline_calibration_1000.npy'

    # r'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/classifier/TRE_full_trawl/beta_calibrated/'
    # Get all matching folders
    # trawl_subfolders = [
    #    f for f in glob.glob(os.path.join(folder_path, 'mcmc_results_sup_ig_nig_5p', "trawl_*"))
    #    if os.path.isdir(f)
    # ]

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
        # seq_len = a_classifier_config['trawl_config']['seq_len']

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(
        os.path.dirname(folder_path)), f'cal_dataset_{seq_len}')
    cal_x_path = os.path.join(dataset_path, 'cal_x.npy')
    cal_thetas_path = os.path.join(dataset_path, 'cal_thetas.npy')
    cal_Y_path = os.path.join(dataset_path, 'cal_Y.npy')

    # Load first few rows of cal_x with memory mapping
    cal_x = np.load(cal_x_path, mmap_mode='r')[:num_rows_to_load]

    # Also load just the first few rows of cal_Y to match
    cal_Y = np.load(cal_Y_path)[:num_rows_to_load]

    # Load cal_thetas (adjust if it also needs to be limited)
    cal_thetas = np.load(cal_thetas_path)[:num_rows_to_load]

    # Now create the mask using the truncated cal_Y
    mask = cal_Y == 1

    # Apply the mask and reshape as before
    true_trawls = cal_x[:, mask].reshape(-1, seq_len)
    true_thetas = cal_thetas[:, mask].reshape(-1, cal_thetas.shape[-1])

    # Load approximate likelihood function
    _, wrapper_for_approx_likelihood_just_theta = load_trained_models(
        folder_path, true_trawls[[0], ::-1], trawl_process_type,
        use_tre, use_summary_statistics, calibration_filename
    )

    results_list = []
    for idx in tqdm(range(num_trawls_to_use)):

        results_list.append(get_MLE(true_trawls[idx], true_thetas[idx]))

    idx_list = []
    true_theta_list = []
    MLE_list = []
    log_likelihood_list = []

    for idx in range(num_trawls_to_use):

        idx_list.append(idx)
        true_theta_list.append(true_thetas[idx])
        MLE_list.append(np.array(results_list[idx].x))
        log_likelihood_list.append(-results_list[idx].fun)

    df = pd.DataFrame({
        'idx': idx_list,
        'true_theta': true_theta_list,
        'MLE': MLE_list,
        'log_likelihood_list': log_likelihood_list
    })

    results_path = os.path.join(folder_path, f'results_{seq_len}')
    os.makedirs(results_path,  exist_ok=True)
    df.to_pickle(os.path.join(
        results_path, f'MLE_results_{calibration_filename[:-4]}.pkl'))
