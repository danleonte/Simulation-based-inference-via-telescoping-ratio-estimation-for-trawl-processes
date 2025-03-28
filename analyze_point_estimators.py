if True:
    from path_setup import setup_sys_path
    setup_sys_path()

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from src.utils.acf_functions import get_acf
from src.utils.KL_divergence import vec_monte_carlo_kl_3_param_nig


def compare_point_estimators(folder_path, num_lags=50):

    results_path = os.path.join(folder_path, 'results')

    # makes jax arrays into np arrays
    df = pd.read_pickle(os.path.join(results_path, 'MLE_results.pkl'))

    true_theta = np.array([np.array(i) for i in df.true_theta.values])
    MLE_theta = np.array([np.array(i) for i in df.MLE.values])

    # acf errors

    H = np.arange(1, num_lags + 1)
    # this should be num_trawls, num_lags shaped
    theoretical_acf = acf_func(H, true_theta[:, :2])
    infered_acf = acf_func(H, MLE_theta[:, :2])
    acf_differences = np.abs(theoretical_acf - infered_acf)
    # might want to convert these to integrals or to sums
    med_L1_acf = np.median(np.mean(acf_differences, axis=1))
    L1_acf = np.mean(np.mean(acf_differences, axis=1))
    L2_acf = np.mean(np.sqrt(np.mean(acf_differences**2, axis=1)))
    rMSE_acf = np.mean(np.mean(acf_differences**2, axis=1))**0.5

    # marginal errors
    marginal_medAEs = np.median(
        np.abs((true_theta - MLE_theta)[:, 2:]), axis=0)
    marginal_MAEs = np.mean(np.abs((true_theta - MLE_theta)[:, 2:]), axis=0)
    marginal_MSEs = np.mean(
        np.square((true_theta - MLE_theta)[:, 2:]), axis=0)**0.5

    print('Starting KL divergence calculations')

    batched_true_thetas = [jnp.array(true_theta[i:i + 100, 2:])
                           for i in range(0, len(true_theta), 100)]
    batched_inferred_thetas = [
        jnp.array(MLE_theta[i:i + 100, 2:]) for i in range(0, len(infered_acf), 100)]

    forward_kl = []
    rev_kl = []

    for i in range(len(batched_true_thetas)):

        num_samples = 6000
        params1 = batched_true_thetas[i]
        params2 = batched_inferred_thetas[i]
        vec_key = jax.random.split(jax.random.PRNGKey(12414), len(params1))

        forward_kl.append(vec_monte_carlo_kl_3_param_nig(
            params1, params2, vec_key, num_samples)[0].mean().item())
        rev_kl.append(vec_monte_carlo_kl_3_param_nig(
            params2, params1, vec_key, num_samples)[0].mean().item())

    acf_estimation_error = {
        'med_L1_acf': med_L1_acf,
        'L1_acf': L1_acf,
        'L2_acf': L2_acf,
        'rMSE_acf': rMSE_acf
    }

    marginal_estimation_error = {
        'marginal_medAE': marginal_medAEs,
        'marginal_MAEs': marginal_MAEs,
        'marginal_MSEs': marginal_MSEs,
        'forward_kl': np.mean(forward_kl),
        'rev_kl': np.mean(rev_kl),
    }

    np.save(os.path.join(results_path, 'acf_estimation_error.npy'),
            acf_estimation_error)
    np.save(os.path.join(results_path, 'marginal_estimation_error.npy'),
            marginal_estimation_error)
    # np.load(os.path.join(results_path,'acf_estimation_error.npy'),allow_pickle=True)


if __name__ == '__main__':

    # folder_path = r'/home/leonted/SBI/SBI_for_trawl_processes_and_ambit_fields/models/classifier/TRE_full_trawl/beta_calibrated/results'
    folder_path = r'D:\sbi_ambit\SBI_for_trawl_processes_and_ambit_fields\models\classifier\TRE_full_trawl\beta_calibrated'
    acf_func = jax.vmap(get_acf('sup_IG'), in_axes=(None, 0))
